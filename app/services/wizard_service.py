"""Wizard service: orchestration layer for the guided GoalGroup creation wizard."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud import wizards as crud_wizard
from app.models.goal_group_wizard import GoalGroupWizard, WizardStatus
from app.models.plan import Plan, PlanStatus

if TYPE_CHECKING:
    from app.models.goal_group import GoalGroup

logger = logging.getLogger(__name__)

WIZARD_TTL_HOURS = 24

_TERMINAL = frozenset({WizardStatus.confirmed, WizardStatus.cancelled, WizardStatus.failed})


def _now_utc() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _default_expires_at() -> datetime:
    return _now_utc() + timedelta(hours=WIZARD_TTL_HOURS)


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------


async def create_wizard(db: AsyncSession, *, go_getter_id: int) -> GoalGroupWizard:
    """Create a new wizard in collecting_scope state.

    Raises ValueError if the go_getter already has an active (non-terminal) wizard.
    """
    existing = await crud_wizard.get_active_for_go_getter(db, go_getter_id)
    if existing:
        raise ValueError(
            f"An active wizard already exists for this go_getter (id={existing.id}). "
            "Complete or cancel it before starting a new one."
        )
    wizard = await crud_wizard.create(
        db, go_getter_id=go_getter_id, expires_at=_default_expires_at()
    )
    return wizard


async def set_scope(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    *,
    title: str,
    description: Optional[str],
    start_date,
    end_date,
) -> GoalGroupWizard:
    """Save scope fields, transition to collecting_targets.

    Raises ValueError if the date range is less than 7 days.
    """
    _assert_not_terminal(wizard)
    span = (end_date - start_date).days
    if span < 7:
        raise ValueError(f"end_date must be at least 7 days after start_date (got {span} days).")
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        group_title=title,
        group_description=description,
        start_date=start_date,
        end_date=end_date,
        status=WizardStatus.collecting_targets,
    )
    return wizard


async def set_targets(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    *,
    target_specs: list[dict],
) -> GoalGroupWizard:
    """Save target specs, transition to collecting_constraints.

    Raises ValueError if any target_id doesn't belong to the wizard's go_getter.
    """
    _assert_not_terminal(wizard)
    from app.models.target import Target

    normalized: list[dict] = []
    for spec in target_specs:
        target_id = spec.get("target_id")
        result = await db.execute(select(Target).where(Target.id == target_id))
        target = result.scalar_one_or_none()
        if target is None:
            raise ValueError(f"Target {target_id} not found.")
        if target.go_getter_id != wizard.go_getter_id:
            raise ValueError(
                f"Target {target_id} does not belong to go_getter {wizard.go_getter_id}."
            )
        # P1: normalize subcategory_id from DB so feasibility checks use the
        # authoritative value, not whatever the client sent.
        normalized.append(
            {
                "target_id": target_id,
                "subcategory_id": target.subcategory_id,
                "priority": spec.get("priority", 3),
            }
        )

    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        target_specs=normalized,
        status=WizardStatus.collecting_constraints,
    )
    return wizard


async def set_constraints(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    *,
    constraints: dict,
) -> GoalGroupWizard:
    """Save constraints, then trigger plan generation + feasibility check."""
    _assert_not_terminal(wizard)
    # Store constraints with string keys (JSON serialisation)
    str_constraints = {str(k): v for k, v in constraints.items()}
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        constraints=str_constraints,
        status=WizardStatus.generating_plans,
    )
    await _generate_and_check(db, wizard)
    return wizard


async def run_feasibility(db: AsyncSession, wizard: GoalGroupWizard) -> GoalGroupWizard:
    """Run feasibility check on the current wizard state."""
    _assert_not_terminal(wizard)
    from app.services.feasibility_service import check_feasibility, enrich_with_llm

    risks = await check_feasibility(db, wizard)
    risks = await enrich_with_llm(risks)

    passed = not any(r.is_blocker for r in risks)
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        feasibility_risks=[r.to_dict() for r in risks],
        feasibility_passed=1 if passed else 0,
        status=WizardStatus.feasibility_check,
    )
    return wizard


async def adjust(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    *,
    patch: dict,
) -> GoalGroupWizard:
    """Apply partial updates to target_specs / constraints and re-generate + re-check."""
    _assert_not_terminal(wizard)

    updates: dict = {"status": WizardStatus.adjusting}

    if "target_specs" in patch and patch["target_specs"] is not None:
        from app.models.target import Target as _Target

        validated: list[dict] = []
        for spec in patch["target_specs"]:
            target_id = spec.get("target_id")
            t_result = await db.execute(select(_Target).where(_Target.id == target_id))
            target = t_result.scalar_one_or_none()
            if target is None:
                raise ValueError(f"Target {target_id} not found.")
            if target.go_getter_id != wizard.go_getter_id:
                raise ValueError(
                    f"Target {target_id} does not belong to go_getter {wizard.go_getter_id}."
                )
            validated.append(
                {
                    "target_id": target_id,
                    "subcategory_id": target.subcategory_id,
                    "priority": spec.get("priority", 3),
                }
            )
        updates["target_specs"] = validated

    if "constraints" in patch and patch["constraints"] is not None:
        new_constraints = {str(k): v for k, v in patch["constraints"].items()}
        updates["constraints"] = new_constraints

    wizard = await crud_wizard.update_wizard(db, wizard, **updates)

    # Cancel any previously generated draft plans
    await _cancel_draft_plans(db, wizard)
    wizard = await crud_wizard.update_wizard(
        db, wizard, draft_plan_ids=[], status=WizardStatus.generating_plans
    )

    await _generate_and_check(db, wizard)
    return wizard


async def confirm(db: AsyncSession, wizard: GoalGroupWizard) -> tuple[GoalGroup, list[dict]]:
    """Atomically create the GoalGroup and activate all draft plans.

    Returns (goal_group, superseded_plans) where superseded_plans is a list of
    {"plan_id", "title", "target_id"} dicts for any previously-active plans that
    were marked completed to make room for the new wizard plans.

    Raises ValueError if the wizard has unresolved blocker risks.
    """
    _assert_not_terminal(wizard)
    if wizard.feasibility_passed is None:
        raise ValueError("Feasibility check has not been run yet.")
    if wizard.feasibility_passed == 0:
        raise ValueError(
            "Cannot confirm: wizard has blocking feasibility issues. "
            "Fix the issues or call adjust first."
        )
    if not wizard.draft_plan_ids:
        raise ValueError("No draft plans found. Run constraints step first.")
    # P1: reject confirm when generation_errors exist — some targets have no plan
    if wizard.generation_errors:
        raise ValueError(
            "Cannot confirm: plan generation failed for one or more targets. "
            "Call adjust to retry or remove the failing targets."
        )

    from app.crud.goal_groups import create as crud_create_group, get_active_for_go_getter
    from app.models.goal_group import GoalGroupStatus
    from app.models.target import Target

    # Enforce one-active-group invariant (service layer)
    existing_group = await get_active_for_go_getter(db, wizard.go_getter_id)
    if existing_group:
        raise ValueError(
            f"GoGetter already has an active GoalGroup (id={existing_group.id}). "
            "Complete or archive it before confirming the wizard."
        )

    # Create the GoalGroup
    group = await crud_create_group(
        db,
        go_getter_id=wizard.go_getter_id,
        title=wizard.group_title or "My Goal Group",
        description=wizard.group_description,
        start_date=wizard.start_date,
        end_date=wizard.end_date,
    )

    # Activate all draft plans and link them + their targets to the group
    superseded_plans: list[dict] = []
    target_ids_linked: set[int] = set()
    for plan_id in wizard.draft_plan_ids:
        result = await db.execute(select(Plan).where(Plan.id == plan_id))
        plan = result.scalar_one_or_none()
        if plan is None:
            logger.warning("Wizard %d: draft plan %d not found during confirm", wizard.id, plan_id)
            continue
        if plan.status == PlanStatus.draft:
            # P0: deactivate any live active plan for this target at confirm time,
            # not during draft generation, so existing plans are never mutated
            # before the user confirms the wizard.
            existing_active = await db.execute(
                select(Plan).where(
                    Plan.target_id == plan.target_id,
                    Plan.status == PlanStatus.active,
                )
            )
            for old_plan in existing_active.scalars().all():
                superseded_plans.append(
                    {
                        "plan_id": old_plan.id,
                        "title": old_plan.title,
                        "target_id": old_plan.target_id,
                    }
                )
                old_plan.status = PlanStatus.completed
                db.add(old_plan)

            plan.status = PlanStatus.active
            plan.group_id = group.id
            db.add(plan)

            if plan.target_id not in target_ids_linked:
                t_result = await db.execute(select(Target).where(Target.id == plan.target_id))
                target = t_result.scalar_one_or_none()
                if target:
                    target.group_id = group.id
                    db.add(target)
                target_ids_linked.add(plan.target_id)

    await db.flush()

    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        goal_group_id=group.id,
        status=WizardStatus.confirmed,
    )
    logger.info(
        "Wizard %d confirmed: created GoalGroup %d with plans %s (superseded: %s)",
        wizard.id,
        group.id,
        wizard.draft_plan_ids,
        [p["plan_id"] for p in superseded_plans],
    )
    return group, superseded_plans


async def cancel_wizard(db: AsyncSession, wizard: GoalGroupWizard) -> None:
    """Cancel the wizard and all draft plans generated within it."""
    if wizard.status in _TERMINAL:
        return
    await _cancel_draft_plans(db, wizard)
    await crud_wizard.update_wizard(db, wizard, status=WizardStatus.cancelled)


# ---------------------------------------------------------------------------
# Graph-node service functions (thin, called by wizard_graph.py nodes)
# ---------------------------------------------------------------------------


async def save_constraints_to_db(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    constraints: dict,
) -> GoalGroupWizard:
    """Save constraints to DB and set status=generating_plans. Does not trigger generation."""
    _assert_not_terminal(wizard)
    str_constraints = {str(k): v for k, v in constraints.items()}
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        constraints=str_constraints,
        status=WizardStatus.generating_plans,
    )
    return wizard


async def run_web_research_step(db: AsyncSession, wizard: GoalGroupWizard) -> GoalGroupWizard:
    """Load GoGetter.grade and run web research for all targets (best-effort)."""
    from app.models.go_getter import GoGetter

    gg_result = await db.execute(select(GoGetter).where(GoGetter.id == wizard.go_getter_id))
    go_getter = gg_result.scalar_one_or_none()
    if go_getter is None:
        return wizard
    await _run_web_research(db, wizard, go_getter.grade)
    return wizard


async def generate_plans_parallel(
    db: AsyncSession,
    wizard: GoalGroupWizard,
) -> tuple[list[int], list[dict]]:
    """Generate draft plans for all targets in parallel using independent DB sessions.

    Each target gets its own AsyncSessionLocal session that commits after plan creation,
    so partial success is possible: some plans may succeed while others fail.
    Returns (plan_ids, errors).
    """
    from app.services import plan_generator
    from app.models.target import Target
    from app.models.go_getter import GoGetter
    from app.database import AsyncSessionLocal

    gg_result = await db.execute(select(GoGetter).where(GoGetter.id == wizard.go_getter_id))
    go_getter = gg_result.scalar_one_or_none()
    if go_getter is None:
        return [], [{"error": "GoGetter not found"}]

    target_specs = wizard.target_specs or []
    reference_materials = wizard.reference_materials or {}

    async def _generate_one(spec: dict) -> tuple[int | None, dict | None]:
        from app.models.goal_group_wizard import GoalGroupWizard as _WizardModel

        target_id = spec.get("target_id")
        subcategory_id = spec.get("subcategory_id")
        async with AsyncSessionLocal() as gen_db:
            t_result = await gen_db.execute(select(Target).where(Target.id == target_id))
            target = t_result.scalar_one_or_none()
            if target is None:
                return None, {"target_id": target_id, "error": "Target not found"}
            if target.go_getter_id != wizard.go_getter_id:
                return None, {
                    "target_id": target_id,
                    "error": f"Target does not belong to go_getter {wizard.go_getter_id}",
                }

            constraint = {}
            if wizard.constraints and subcategory_id is not None:
                constraint = (
                    wizard.constraints.get(str(subcategory_id))
                    or wizard.constraints.get(subcategory_id)
                    or {}
                )
            daily_minutes = constraint.get("daily_minutes") if constraint else None
            preferred_days = constraint.get("preferred_days") if constraint else None

            try:
                new_plan = await plan_generator.generate_plan(
                    db=gen_db,
                    target=target,
                    pupil_name=go_getter.name,
                    grade=go_getter.grade,
                    start_date=wizard.start_date,
                    end_date=wizard.end_date,
                    daily_study_minutes=daily_minutes,
                    preferred_days=preferred_days,
                    initial_status=PlanStatus.draft,
                    deactivate_existing=False,
                    reference_materials=reference_materials.get(str(target_id)),
                    wizard_id=wizard.id,
                )
                # Atomically register plan_id on the wizard row before committing.
                # SELECT FOR UPDATE serializes concurrent appends from parallel tasks
                # so draft_plan_ids is crash-safe: if the process dies after this
                # commit, the ID is already persisted and cleanup can find the plan.
                w_result = await gen_db.execute(
                    select(_WizardModel).where(_WizardModel.id == wizard.id).with_for_update()
                )
                w = w_result.scalar_one_or_none()
                if w is not None:
                    existing_ids = list(w.draft_plan_ids or [])
                    if new_plan.id not in existing_ids:
                        existing_ids.append(new_plan.id)
                        w.draft_plan_ids = existing_ids
                        gen_db.add(w)
                await gen_db.commit()
                return new_plan.id, None
            except Exception as exc:
                logger.exception(
                    "Wizard %d: plan generation failed for target %d: %s",
                    wizard.id,
                    target_id,
                    exc,
                )
                return None, {"target_id": target_id, "error": str(exc)}

    results = await asyncio.gather(*[_generate_one(spec) for spec in target_specs])

    plan_ids: list[int] = []
    errors: list[dict] = []
    for plan_id, error in results:
        if plan_id is not None:
            plan_ids.append(plan_id)
        if error is not None:
            errors.append(error)

    return plan_ids, errors


async def save_plan_gen_results(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    plan_ids: list[int],
    errors: list[dict],
) -> GoalGroupWizard:
    """Persist draft_plan_ids and generation_errors to DB."""
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        draft_plan_ids=plan_ids,
        generation_errors=errors if errors else None,
    )
    return wizard


async def run_feasibility_step(db: AsyncSession, wizard: GoalGroupWizard) -> GoalGroupWizard:
    """Run feasibility check + LLM enrichment and write results to DB.

    Re-checks terminal state right before writing status so that a concurrent
    mid-generation cancel (which writes directly to DB) cannot be overwritten
    by the feasibility status update.
    """
    from app.services.feasibility_service import check_feasibility, enrich_with_llm

    risks = await check_feasibility(db, wizard)
    risks = await enrich_with_llm(risks)
    passed = not any(r.is_blocker for r in risks)

    # Reload wizard to detect any concurrent status change (e.g. a direct-DB
    # cancel that happened while the LLM calls above were in flight).
    await db.refresh(wizard)
    if wizard.status in _TERMINAL:
        logger.info(
            "run_feasibility_step: wizard %d became terminal (%s) during LLM work, "
            "skipping status write",
            wizard.id,
            wizard.status.value,
        )
        return wizard

    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        feasibility_risks=[r.to_dict() for r in risks],
        feasibility_passed=1 if passed else 0,
        status=WizardStatus.feasibility_check,
    )
    return wizard


async def save_adjust_patch(
    db: AsyncSession,
    wizard: GoalGroupWizard,
    *,
    patch: dict,
) -> GoalGroupWizard:
    """Save patch fields to wizard and cancel old draft plans. Does not run generation."""
    _assert_not_terminal(wizard)

    updates: dict = {"status": WizardStatus.adjusting}

    if "target_specs" in patch and patch["target_specs"] is not None:
        from app.models.target import Target as _Target

        validated: list[dict] = []
        for spec in patch["target_specs"]:
            target_id = spec.get("target_id")
            t_result = await db.execute(select(_Target).where(_Target.id == target_id))
            target = t_result.scalar_one_or_none()
            if target is None:
                raise ValueError(f"Target {target_id} not found.")
            if target.go_getter_id != wizard.go_getter_id:
                raise ValueError(
                    f"Target {target_id} does not belong to go_getter {wizard.go_getter_id}."
                )
            validated.append(
                {
                    "target_id": target_id,
                    "subcategory_id": target.subcategory_id,
                    "priority": spec.get("priority", 3),
                }
            )
        updates["target_specs"] = validated

    if "constraints" in patch and patch["constraints"] is not None:
        new_constraints = {str(k): v for k, v in patch["constraints"].items()}
        updates["constraints"] = new_constraints

    wizard = await crud_wizard.update_wizard(db, wizard, **updates)

    await _cancel_draft_plans(db, wizard)
    wizard = await crud_wizard.update_wizard(
        db, wizard, draft_plan_ids=[], status=WizardStatus.generating_plans
    )
    return wizard


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_not_terminal(wizard: GoalGroupWizard) -> None:
    if wizard.status in _TERMINAL:
        raise ValueError(
            f"Wizard {wizard.id} is in terminal state '{wizard.status.value}' and cannot be modified."
        )


async def _cancel_draft_plans(db: AsyncSession, wizard: GoalGroupWizard) -> None:
    """Cancel all draft plans belonging to this wizard.

    Uses two complementary queries:
    1. Plans explicitly tracked in wizard.draft_plan_ids (the normal case).
    2. Any plans with wizard_id == wizard.id (catches orphans committed after a
       mid-generation crash before save_plan_gen_results could write their IDs).
    """
    ids_to_cancel: set[int] = set(wizard.draft_plan_ids or [])

    # Sweep for orphans using the wizard_id backlink
    orphan_result = await db.execute(
        select(Plan).where(
            Plan.wizard_id == wizard.id,
            Plan.status == PlanStatus.draft,
        )
    )
    for orphan in orphan_result.scalars().all():
        ids_to_cancel.add(orphan.id)

    if not ids_to_cancel:
        return

    for plan_id in ids_to_cancel:
        result = await db.execute(select(Plan).where(Plan.id == plan_id))
        plan = result.scalar_one_or_none()
        if plan and plan.status == PlanStatus.draft:
            plan.status = PlanStatus.cancelled
            db.add(plan)
    await db.flush()


async def _run_web_research(
    db: AsyncSession, wizard: GoalGroupWizard, grade: str
) -> tuple[dict, dict]:
    """Run web research for all targets concurrently (best-effort).

    Returns (reference_materials, search_errors) dicts keyed by str(target_id).
    Saves results to wizard immediately. Never raises.
    """
    from app.services.web_research_service import search_study_materials
    from app.models.target import Target

    target_specs = wizard.target_specs or []
    targets: dict[int, Target] = {}
    for spec in target_specs:
        tid = spec.get("target_id")
        if tid is None:
            continue
        t_result = await db.execute(select(Target).where(Target.id == tid))
        t = t_result.scalar_one_or_none()
        if t and t.go_getter_id == wizard.go_getter_id:
            targets[tid] = t

    ordered_ids = [s["target_id"] for s in target_specs if s.get("target_id") in targets]
    if not ordered_ids:
        return {}, {}

    coros = [
        search_study_materials(
            subject=targets[tid].subject,
            grade=grade,
            description=targets[tid].description or targets[tid].title or "",
        )
        for tid in ordered_ids
    ]
    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    reference_materials: dict = {}
    search_errors: dict = {}
    for tid, res in zip(ordered_ids, raw_results, strict=False):
        if isinstance(res, Exception):
            logger.warning("Web research failed for target %d: %s", tid, res)
            search_errors[str(tid)] = str(res)
        elif res:
            reference_materials[str(tid)] = res

    await crud_wizard.update_wizard(
        db,
        wizard,
        reference_materials=reference_materials or None,
        search_errors=search_errors or None,
    )
    return reference_materials, search_errors


async def _generate_and_check(db: AsyncSession, wizard: GoalGroupWizard) -> None:
    """Generate draft plans for all target_specs, then run feasibility.

    Updates wizard in-place via crud_wizard.update_wizard.
    """
    from app.services import plan_generator
    from app.services.feasibility_service import check_feasibility, enrich_with_llm
    from app.models.target import Target
    from app.models.go_getter import GoGetter

    # Load the go_getter for pupil_name and grade
    gg_result = await db.execute(select(GoGetter).where(GoGetter.id == wizard.go_getter_id))
    go_getter = gg_result.scalar_one_or_none()
    if go_getter is None:
        await crud_wizard.update_wizard(
            db,
            wizard,
            generation_errors=[{"error": "GoGetter not found"}],
            status=WizardStatus.failed,
        )
        return

    target_specs = wizard.target_specs or []
    draft_plan_ids: list[int] = []
    errors: list[dict] = []

    wizard = await crud_wizard.update_wizard(
        db, wizard, status=WizardStatus.generating_plans, generation_errors=None
    )

    # Run web research per target (parallel, best-effort)
    research_results, _ = await _run_web_research(db, wizard, go_getter.grade)

    for spec in target_specs:
        target_id = spec.get("target_id")
        subcategory_id = spec.get("subcategory_id")

        t_result = await db.execute(select(Target).where(Target.id == target_id))
        target = t_result.scalar_one_or_none()
        if target is None:
            errors.append({"target_id": target_id, "error": "Target not found"})
            continue
        if target.go_getter_id != wizard.go_getter_id:
            errors.append(
                {
                    "target_id": target_id,
                    "error": f"Target does not belong to go_getter {wizard.go_getter_id}",
                }
            )
            continue

        # Resolve constraints
        constraint = {}
        if wizard.constraints and subcategory_id is not None:
            constraint = (
                wizard.constraints.get(str(subcategory_id))
                or wizard.constraints.get(subcategory_id)
                or {}
            )
        daily_minutes = constraint.get("daily_minutes") if constraint else None
        preferred_days = constraint.get("preferred_days") if constraint else None

        try:
            new_plan = await plan_generator.generate_plan(
                db=db,
                target=target,
                pupil_name=go_getter.name,
                grade=go_getter.grade,
                start_date=wizard.start_date,
                end_date=wizard.end_date,
                daily_study_minutes=daily_minutes,
                preferred_days=preferred_days,
                initial_status=PlanStatus.draft,
                deactivate_existing=False,  # P0: preserve live plans until confirm
                reference_materials=research_results.get(str(target_id)),
                wizard_id=wizard.id,
            )
            draft_plan_ids.append(new_plan.id)
        except Exception as exc:
            logger.exception(
                "Wizard %d: plan generation failed for target %d: %s",
                wizard.id,
                target_id,
                exc,
            )
            errors.append({"target_id": target_id, "error": str(exc)})

    # Persist draft_plan_ids regardless of errors
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        draft_plan_ids=draft_plan_ids,
        generation_errors=errors if errors else None,
    )

    # Run feasibility (always, even if some plans failed)
    risks = await check_feasibility(db, wizard)
    risks = await enrich_with_llm(risks)
    passed = not any(r.is_blocker for r in risks)
    wizard = await crud_wizard.update_wizard(
        db,
        wizard,
        feasibility_risks=[r.to_dict() for r in risks],
        feasibility_passed=1 if passed else 0,
        status=WizardStatus.feasibility_check,
    )
