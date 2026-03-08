"""Guided GoalGroup creation wizard endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import require_best_pal_or_admin, verify_best_pal_owns_go_getter
from app.crud.wizards import get as crud_get_wizard
from app.database import AsyncSessionLocal, get_db
from app.models.goal_group_wizard import GoalGroupWizard, TERMINAL_STATUSES, WizardStatus
from app.schemas.wizard import (
    AdjustRequest,
    ConstraintsRequest,
    FeasibilityRiskOut,
    ScopeRequest,
    TargetsRequest,
    WizardCreate,
    WizardResponse,
)
from app.services import wizard_service
from app.services.wizard_graph import (
    INTERRUPT_NODES,
    Command,
    WizardState,
    assert_graph_awaiting,
    get_wizard_graph,
)

router = APIRouter(prefix="/wizards", tags=["wizards"])


# ---------------------------------------------------------------------------
# Helper: build WizardResponse from ORM object
# ---------------------------------------------------------------------------


def _build_response(wizard: GoalGroupWizard) -> WizardResponse:
    risks: list[FeasibilityRiskOut] = []
    if wizard.feasibility_risks:
        for r in wizard.feasibility_risks:
            risks.append(
                FeasibilityRiskOut(
                    rule_code=r["rule_code"],
                    level=r["level"],
                    subcategory_id=r.get("subcategory_id"),
                    detail=r["detail"],
                    llm_explanation=r.get("llm_explanation", ""),
                    is_blocker=r.get("is_blocker", False),
                )
            )
    feasibility_passed: Optional[bool] = None
    if wizard.feasibility_passed is not None:
        feasibility_passed = bool(wizard.feasibility_passed)

    return WizardResponse(
        id=wizard.id,
        go_getter_id=wizard.go_getter_id,
        status=wizard.status,
        group_title=wizard.group_title,
        group_description=wizard.group_description,
        start_date=wizard.start_date,
        end_date=wizard.end_date,
        target_specs=wizard.target_specs,
        constraints=wizard.constraints,
        draft_plan_ids=wizard.draft_plan_ids,
        feasibility_passed=feasibility_passed,
        feasibility_risks=risks,
        goal_group_id=wizard.goal_group_id,
        generation_errors=wizard.generation_errors,
        expires_at=wizard.expires_at,
        created_at=wizard.created_at,
        updated_at=wizard.updated_at,
    )


async def _load_wizard_and_verify(
    wizard_id: int,
    chat_id: int,
    db: AsyncSession,
) -> GoalGroupWizard:
    """Load wizard, verify ownership, return the ORM object."""
    wizard = await crud_get_wizard(db, wizard_id)
    if wizard is None:
        raise HTTPException(404, "Wizard not found")
    await verify_best_pal_owns_go_getter(wizard.go_getter_id, chat_id, db)
    return wizard


def _graph_config(wizard_id: int) -> dict:
    return {"configurable": {"thread_id": str(wizard_id)}}


async def _reload_wizard(wizard_id: int) -> GoalGroupWizard:
    """Load wizard in a fresh session (to see changes committed by graph nodes)."""
    async with AsyncSessionLocal() as db:
        wizard = await crud_get_wizard(db, wizard_id)
        assert wizard is not None
        return wizard


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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("", response_model=WizardResponse, status_code=201)
async def create_wizard(
    body: WizardCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Start a new GoalGroup creation wizard.

    Returns 409 if the go_getter already has an active (non-terminal) wizard.
    """
    await verify_best_pal_owns_go_getter(body.go_getter_id, chat_id, db)
    try:
        wizard = await wizard_service.create_wizard(db, go_getter_id=body.go_getter_id)
    except ValueError as e:
        raise HTTPException(409, str(e)) from e

    wizard_id = wizard.id
    wizard_go_getter_id = wizard.go_getter_id

    # Flush so wizard.id is available; middleware commits at request end.
    # We capture the response before initialising the graph thread (which
    # hits interrupt() immediately and needs no DB access).
    response = _build_response(wizard)

    graph = get_wizard_graph()
    await graph.ainvoke(
        _initial_graph_state(wizard_id, wizard_go_getter_id),
        config=_graph_config(wizard_id),
    )

    return response


@router.get("/{wizard_id}", response_model=WizardResponse)
async def get_wizard(
    wizard_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Resume / status-check a wizard."""
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    return _build_response(wizard)


@router.post("/{wizard_id}/scope", response_model=WizardResponse)
async def set_scope(
    wizard_id: int,
    body: ScopeRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Set time scope (title, description, start_date, end_date).

    Transitions wizard to collecting_targets.
    Validates end_date > start_date + 7 days.
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    _assert_active(wizard)

    graph = get_wizard_graph()
    try:
        await assert_graph_awaiting(graph, wizard_id, "scope")
        await graph.ainvoke(
            Command(
                resume={
                    "title": body.title,
                    "description": body.description,
                    "start_date": body.start_date,
                    "end_date": body.end_date,
                }
            ),
            config=_graph_config(wizard_id),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return _build_response(await _reload_wizard(wizard_id))


@router.post("/{wizard_id}/targets", response_model=WizardResponse)
async def set_targets(
    wizard_id: int,
    body: TargetsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Set the target specs (which existing Targets to include).

    Transitions wizard to collecting_constraints.
    Validates each target belongs to the go_getter.
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    _assert_active(wizard)
    target_specs_raw: list[dict[str, Any]] = [s.model_dump() for s in body.target_specs]

    graph = get_wizard_graph()
    try:
        await assert_graph_awaiting(graph, wizard_id, "targets")
        await graph.ainvoke(
            Command(resume={"target_specs": target_specs_raw}),
            config=_graph_config(wizard_id),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return _build_response(await _reload_wizard(wizard_id))


@router.post("/{wizard_id}/constraints", response_model=WizardResponse)
async def set_constraints(
    wizard_id: int,
    body: ConstraintsRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Set per-target constraints (daily_minutes, preferred_days).

    Triggers async plan generation and feasibility check.
    Transitions wizard through generating_plans → feasibility_check.
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    _assert_active(wizard)
    constraints_raw = {k: v.model_dump() for k, v in body.constraints.items()}

    graph = get_wizard_graph()
    try:
        await assert_graph_awaiting(graph, wizard_id, "save_constraints")
        await graph.ainvoke(
            Command(resume={"constraints": constraints_raw}),
            config=_graph_config(wizard_id),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return _build_response(await _reload_wizard(wizard_id))


@router.get("/{wizard_id}/feasibility", response_model=WizardResponse)
async def get_feasibility(
    wizard_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Return the current feasibility risks and passed flag."""
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    return _build_response(wizard)


@router.post("/{wizard_id}/adjust", response_model=WizardResponse)
async def adjust(
    wizard_id: int,
    body: AdjustRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Apply adjustments to target_specs / constraints and re-generate plans.

    Transitions wizard through adjusting → generating_plans → feasibility_check.
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    _assert_active(wizard)
    patch: dict[str, Any] = {}
    if body.target_specs is not None:
        patch["target_specs"] = [s.model_dump() for s in body.target_specs]
    if body.constraints is not None:
        patch["constraints"] = {k: v.model_dump() for k, v in body.constraints.items()}

    graph = get_wizard_graph()
    try:
        await assert_graph_awaiting(graph, wizard_id, "human_gate")
        await graph.ainvoke(
            Command(resume={"decision": "adjust", "patch": patch}),
            config=_graph_config(wizard_id),
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    return _build_response(await _reload_wizard(wizard_id))


@router.post("/{wizard_id}/confirm", response_model=WizardResponse)
async def confirm(
    wizard_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Confirm the wizard: create the GoalGroup and activate all draft plans.

    Returns 409 if feasibility_passed is False (has blockers).
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    _assert_active(wizard)

    graph = get_wizard_graph()
    try:
        await assert_graph_awaiting(graph, wizard_id, "human_gate")
        await graph.ainvoke(
            Command(resume={"decision": "confirm"}),
            config=_graph_config(wizard_id),
        )
    except ValueError as e:
        raise HTTPException(409, str(e)) from e

    return _build_response(await _reload_wizard(wizard_id))


@router.delete("/{wizard_id}", response_model=WizardResponse)
async def cancel_wizard(
    wizard_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Cancel the wizard and discard all committed draft plans.

    When waiting for user input: cancels immediately via the graph (synchronous).
    When generation is in progress: marks the wizard cancelled immediately; plans
    already committed are cancelled, in-flight LLM calls are not preempted but
    their plans are cancelled on commit via the wizard_id backlink.
    """
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
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
            fresh = await crud_get_wizard(cancel_db, wizard_id)
            if fresh:
                await wizard_service.cancel_wizard(cancel_db, fresh)
                await cancel_db.commit()

    return _build_response(await _reload_wizard(wizard_id))


@router.get("/{wizard_id}/sources")
async def get_wizard_sources(
    wizard_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    """Return reference materials found during web research for this wizard."""
    wizard = await _load_wizard_and_verify(wizard_id, chat_id, db)
    return {
        "wizard_id": wizard.id,
        "reference_materials": wizard.reference_materials or {},
        "search_errors": wizard.search_errors or {},
    }


# ---------------------------------------------------------------------------
# Guard helper
# ---------------------------------------------------------------------------


def _assert_active(wizard: GoalGroupWizard) -> None:
    if wizard.status in TERMINAL_STATUSES:
        raise HTTPException(
            409,
            f"Wizard is in terminal state '{wizard.status.value}' and cannot be modified.",
        )
