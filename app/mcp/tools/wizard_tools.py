"""Wizard MCP tools: guided GoalGroup creation (role: best_pal/admin).

Conversation flow driven by the LLM:

  start_goal_group_wizard  →  set_wizard_scope  →  set_wizard_targets
    →  set_wizard_constraints  (triggers LLM plan gen + feasibility, ~10-30 s)
    →  [get_wizard_status to read risks]
    →  confirm_goal_group          (if feasibility_passed)
    →  adjust_wizard + confirm     (if blockers need fixing)
    →  cancel_goal_group_wizard    (see cancel semantics below)

State transitions (scope / targets / constraints / confirm / adjust / cancel)
are driven through the LangGraph wizard graph.  Read-only tools (status,
sources, feasibility) still query the DB directly.

Cancel semantics
----------------
cancel_goal_group_wizard is cooperative, not preemptive:

• When the wizard is waiting for input (scope / targets / constraints /
  human_gate): the cancel is delivered immediately through the graph and all
  committed draft plans are cancelled synchronously.

• When generation is in progress (research / generate_plans / feasibility):
  the wizard DB record is cancelled immediately, and all draft plans already
  committed at that moment are cancelled.  Any LLM calls still in flight
  complete in the background and their plans are also cancelled once they
  commit (because plan.wizard_id links them back to this wizard).
  There is no preemptive signal to stop the in-flight LLM calls.
"""

import logging
from datetime import date
from typing import Optional

from sqlalchemy import select

from app.database import AsyncSessionLocal
from app.mcp.auth import Role, require_role, verify_best_pal_owns_go_getter
from app.mcp.server import mcp
from app.crud import wizards as crud_wizard
from app.models.goal_group_wizard import GoalGroupWizard
from app.models.target import Target
from app.services import wizard_service
from app.services.wizard_graph import (
    INTERRUPT_NODES,
    Command,
    WizardState,
    assert_graph_awaiting,
    get_wizard_graph,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_chat_id(chat_id: Optional[int]) -> int:
    if chat_id is None:
        raise ValueError("X-Telegram-Chat-Id header is required")
    return chat_id


def _wizard_to_dict(wizard: GoalGroupWizard) -> dict:
    """Serialize wizard to a compact dict for MCP responses."""
    risks = wizard.feasibility_risks or []
    return {
        "wizard_id": wizard.id,
        "go_getter_id": wizard.go_getter_id,
        "status": wizard.status.value,
        "group_title": wizard.group_title,
        "start_date": str(wizard.start_date) if wizard.start_date else None,
        "end_date": str(wizard.end_date) if wizard.end_date else None,
        "target_specs": wizard.target_specs,
        "feasibility_passed": wizard.feasibility_passed,
        "blockers": [r for r in risks if r.get("is_blocker")],
        "warnings": [r for r in risks if not r.get("is_blocker")],
        "goal_group_id": wizard.goal_group_id,
        "generation_errors": wizard.generation_errors,
        "expires_at": wizard.expires_at.isoformat(),
    }


async def _load_wizard(db, wizard_id: int, go_getter_id: int) -> GoalGroupWizard:
    """Fetch wizard and assert it belongs to the expected go_getter."""
    wizard = await crud_wizard.get(db, wizard_id)
    if wizard is None:
        raise ValueError(f"Wizard {wizard_id} not found")
    if wizard.go_getter_id != go_getter_id:
        raise PermissionError(f"Wizard {wizard_id} does not belong to go_getter {go_getter_id}")
    return wizard


def _subcategory_map_from_specs(target_specs: list[dict]) -> dict[int, int]:
    """Extract {target_id: subcategory_id} from normalized wizard target_specs."""
    return {s["target_id"]: s["subcategory_id"] for s in target_specs}


async def _lookup_subcategory_ids(db, target_ids: list[int]) -> dict[int, int]:
    """Query DB for authoritative {target_id: subcategory_id} for each target_id.

    Raises ValueError if any target_id is not found.
    """
    result = await db.execute(select(Target).where(Target.id.in_(target_ids)))
    targets = {t.id: t.subcategory_id for t in result.scalars().all()}
    missing = set(target_ids) - set(targets)
    if missing:
        raise ValueError(f"Targets not found: {sorted(missing)}")
    return targets


def _build_constraints_dict(
    target_ids: list[int],
    subcategory_map: dict[int, int],
    daily_minutes_list: list[int],
    preferred_days_list: Optional[list[list[int]]],
) -> dict:
    """Build the {subcategory_id: {daily_minutes, preferred_days}} dict expected by wizard_service."""
    default_days = [0, 1, 2, 3, 4, 5, 6]
    return {
        subcategory_map[tid]: {
            "daily_minutes": daily_minutes_list[i],
            "preferred_days": preferred_days_list[i] if preferred_days_list else default_days,
        }
        for i, tid in enumerate(target_ids)
    }


def _initial_graph_state(wizard_id: int, go_getter_id: int) -> WizardState:
    return {
        "wizard_id": wizard_id,
        "go_getter_id": go_getter_id,
        "status": "collecting_scope",
        "human_decision": "",
        "error": "",
        "adjust_patch": {},
        "confirm_result": {},
    }


def _graph_config(wizard_id: int) -> dict:
    return {"configurable": {"thread_id": str(wizard_id)}}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def start_goal_group_wizard(
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Start a guided GoalGroup creation wizard for a go_getter.

    THIS IS THE PRIMARY ENTRY POINT for creating a new study plan / GoalGroup.
    Always use this wizard flow when the user wants to set up goals or a study plan:
      1. start_goal_group_wizard  → get wizard_id
      2. set_wizard_scope         → set title + date range (ask user for these)
      3. set_wizard_targets       → set which targets to include (ask user)
      4. set_wizard_constraints   → set daily minutes + preferred days, triggers AI plan gen
      5. confirm_goal_group       → finalise (if feasibility_passed=True)

    Ask the user for each piece of information conversationally before calling each step.
    Returns wizard_id and status='collecting_scope'.
    Raises an error if the go_getter already has an active wizard — call
    get_wizard_status with the existing wizard_id to resume it instead.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await wizard_service.create_wizard(db, go_getter_id=go_getter_id)
        await db.commit()
        wizard_id = wizard.id
        wizard_dict = _wizard_to_dict(wizard)

    # Initialize graph thread — scope_node immediately calls interrupt() so
    # ainvoke returns quickly without doing any further DB work.
    graph = get_wizard_graph()
    await graph.ainvoke(
        _initial_graph_state(wizard_id, go_getter_id),
        config=_graph_config(wizard_id),
    )

    return wizard_dict


@mcp.tool()
async def get_wizard_status(
    wizard_id: int,
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Get the current status and feasibility risks of a wizard.

    Use this to resume a wizard after a Telegram restart, or to read the
    feasibility risks after set_wizard_constraints completes.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return _wizard_to_dict(wizard)


@mcp.tool()
async def get_wizard_sources(
    wizard_id: int,
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Get curriculum reference materials found during web research for this wizard.

    Call after set_wizard_constraints to review sources before confirming.
    Returns reference_materials: {target_id: [{title, source, url, key_points}]}.
    Empty dict if web research has not run yet or found nothing.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return {
            "wizard_id": wizard_id,
            "reference_materials": wizard.reference_materials or {},
            "search_errors": wizard.search_errors or {},
        }


@mcp.tool()
async def set_wizard_scope(
    wizard_id: int,
    go_getter_id: int,
    title: str,
    start_date: str,
    end_date: str,
    description: Optional[str] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Set the GoalGroup title and date range (wizard step 2).

    start_date / end_date: ISO 8601 strings, e.g. '2026-06-01'.
    end_date must be at least 7 days after start_date.
    On success transitions to status='collecting_targets'.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)

    graph = get_wizard_graph()
    await assert_graph_awaiting(graph, wizard_id, "scope")
    await graph.ainvoke(
        Command(
            resume={
                "title": title,
                "description": description,
                "start_date": date.fromisoformat(start_date),
                "end_date": date.fromisoformat(end_date),
            }
        ),
        config=_graph_config(wizard_id),
    )

    async with AsyncSessionLocal() as db:
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return _wizard_to_dict(wizard)


@mcp.tool()
async def set_wizard_targets(
    wizard_id: int,
    go_getter_id: int,
    target_ids: list[int],
    priorities: Optional[list[int]] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Set the targets included in this GoalGroup (wizard step 3).

    target_ids: IDs of existing active Targets for this go_getter.
                Use list_targets to discover them.
    priorities: optional parallel list of 1-5 priority values (default 3 each).
    On success transitions to status='collecting_constraints'.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    if priorities is not None and len(priorities) != len(target_ids):
        raise ValueError("priorities must have the same length as target_ids")
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)

    # Pass subcategory_id=0 — wizard_service.set_targets normalises it from DB
    target_specs = [
        {
            "target_id": tid,
            "subcategory_id": 0,
            "priority": priorities[i] if priorities else 3,
        }
        for i, tid in enumerate(target_ids)
    ]

    graph = get_wizard_graph()
    await assert_graph_awaiting(graph, wizard_id, "targets")
    await graph.ainvoke(
        Command(resume={"target_specs": target_specs}),
        config=_graph_config(wizard_id),
    )

    async with AsyncSessionLocal() as db:
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return _wizard_to_dict(wizard)


@mcp.tool()
async def set_wizard_constraints(
    wizard_id: int,
    go_getter_id: int,
    target_ids: list[int],
    daily_minutes_list: list[int],
    preferred_days_list: Optional[list[list[int]]] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Set per-target study constraints then trigger AI plan generation (wizard step 4).

    target_ids and daily_minutes_list must be parallel lists of the same length.
    preferred_days_list: optional parallel list; each inner list is weekday integers
        (0=Mon … 6=Sun). Defaults to all days [0,1,2,3,4,5,6] per target.

    This triggers LLM plan generation — expect 10–30 seconds per target.
    On success returns status='feasibility_check' with feasibility_passed and
    blockers/warnings lists.

    Next step:
      • feasibility_passed=True  → call confirm_goal_group
      • blockers non-empty       → call adjust_wizard to fix, then confirm

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    if len(target_ids) != len(daily_minutes_list):
        raise ValueError("target_ids and daily_minutes_list must have the same length")
    if preferred_days_list is not None and len(preferred_days_list) != len(target_ids):
        raise ValueError("preferred_days_list must have the same length as target_ids")

    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        # Build subcategory_id keyed constraints from the normalised target_specs
        specs = wizard.target_specs or []
        sub_map = _subcategory_map_from_specs(specs)
        missing = set(target_ids) - set(sub_map)
        if missing:
            raise ValueError(
                f"Target IDs {sorted(missing)} are not in this wizard's target list. "
                "Call set_wizard_targets first, or use the IDs returned by that call."
            )
        constraints = _build_constraints_dict(
            target_ids, sub_map, daily_minutes_list, preferred_days_list
        )

    graph = get_wizard_graph()
    await assert_graph_awaiting(graph, wizard_id, "save_constraints")
    await graph.ainvoke(
        Command(resume={"constraints": constraints}),
        config=_graph_config(wizard_id),
    )

    async with AsyncSessionLocal() as db:
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return _wizard_to_dict(wizard)


@mcp.tool()
async def adjust_wizard(
    wizard_id: int,
    go_getter_id: int,
    target_ids: Optional[list[int]] = None,
    priorities: Optional[list[int]] = None,
    daily_minutes_list: Optional[list[int]] = None,
    preferred_days_list: Optional[list[list[int]]] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Adjust targets or per-target constraints and re-run plan generation + feasibility.

    Use after get_wizard_status shows blocking feasibility issues.
    All parameters are optional — pass only what you want to change.

    To change the target list: pass target_ids (+ optional priorities).
    To change study constraints: pass target_ids + daily_minutes_list
        (uses the wizard's stored target list if target_ids is omitted).
    To change both at once: pass target_ids, daily_minutes_list, and optionally
        priorities and preferred_days_list.

    Triggers LLM re-generation (~10–30 s per target).

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await _load_wizard(db, wizard_id, go_getter_id)

        patch: dict = {}

        # ── Resolve authoritative subcategory_ids from DB for new target list ──
        db_sub_map: dict[int, int] = {}
        if target_ids is not None:
            if priorities is not None and len(priorities) != len(target_ids):
                raise ValueError("priorities must have the same length as target_ids")
            db_sub_map = await _lookup_subcategory_ids(db, target_ids)
            patch["target_specs"] = [
                {
                    "target_id": tid,
                    "subcategory_id": db_sub_map[tid],
                    "priority": priorities[i] if priorities else 3,
                }
                for i, tid in enumerate(target_ids)
            ]

        # ── Update constraints ────────────────────────────────────────────
        if daily_minutes_list is not None:
            ref_ids = (
                target_ids
                if target_ids is not None
                else [s["target_id"] for s in (wizard.target_specs or [])]
            )
            if len(daily_minutes_list) != len(ref_ids):
                raise ValueError(
                    "daily_minutes_list length must match target_ids "
                    "(or the wizard's stored target count when target_ids is omitted)"
                )
            if preferred_days_list is not None and len(preferred_days_list) != len(ref_ids):
                raise ValueError("preferred_days_list must have the same length as target_ids")
            stored_map = _subcategory_map_from_specs(wizard.target_specs or [])
            sub_map = {tid: db_sub_map.get(tid) or stored_map[tid] for tid in ref_ids}
            patch["constraints"] = _build_constraints_dict(
                ref_ids, sub_map, daily_minutes_list, preferred_days_list
            )

        if not patch:
            raise ValueError("Provide at least one of: target_ids, daily_minutes_list")

    graph = get_wizard_graph()
    await assert_graph_awaiting(graph, wizard_id, "human_gate")
    await graph.ainvoke(
        Command(resume={"decision": "adjust", "patch": patch}),
        config=_graph_config(wizard_id),
    )

    async with AsyncSessionLocal() as db:
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        return _wizard_to_dict(wizard)


@mcp.tool()
async def confirm_goal_group(
    wizard_id: int,
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Confirm the wizard: create the GoalGroup and activate all draft plans.

    Only succeeds when feasibility_passed=True and no generation_errors.
    Returns goal_group_id, title, start_date, end_date on success.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)

    graph = get_wizard_graph()
    await assert_graph_awaiting(graph, wizard_id, "human_gate")
    result = await graph.ainvoke(
        Command(resume={"decision": "confirm"}),
        config=_graph_config(wizard_id),
    )

    confirm_data = result.get("confirm_result", {})
    return {
        "goal_group_id": confirm_data.get("goal_group_id"),
        "title": confirm_data.get("title"),
        "go_getter_id": confirm_data.get("go_getter_id"),
        "start_date": confirm_data.get("start_date"),
        "end_date": confirm_data.get("end_date"),
        "wizard_id": wizard_id,
        "status": "confirmed",
        "superseded_plans": confirm_data.get("superseded_plans", []),
    }


@mcp.tool()
async def cancel_goal_group_wizard(
    wizard_id: int,
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Cancel the wizard and discard all committed draft plans.

    When waiting for user input (scope/targets/constraints/human_gate): cancels
    immediately and synchronously via the graph.

    When generation is in progress: marks the wizard cancelled immediately and
    cancels all draft plans committed so far.  In-flight LLM calls are not
    preempted — their plans are cancelled when they commit because each plan
    carries a wizard_id backlink.  Callers should treat this as best-effort
    during generation.

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        wizard = await _load_wizard(db, wizard_id, go_getter_id)
        _ = wizard  # ownership verified

    graph = get_wizard_graph()
    snapshot = await graph.aget_state(_graph_config(wizard_id))
    at_interrupt = bool(snapshot and snapshot.next and snapshot.next[0] in INTERRUPT_NODES)

    if at_interrupt:
        await graph.ainvoke(
            Command(resume={"action": "cancel"}),
            config=_graph_config(wizard_id),
        )
    else:
        # Graph is mid-generation or already finished — cancel the DB record directly.
        async with AsyncSessionLocal() as cancel_db:
            fresh = await crud_wizard.get(cancel_db, wizard_id)
            if fresh:
                await wizard_service.cancel_wizard(cancel_db, fresh)
                await cancel_db.commit()

    return {"wizard_id": wizard_id, "status": "cancelled"}
