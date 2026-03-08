"""LangGraph StateGraph for the guided GoalGroup creation wizard.

Architecture: Write-Through + Thin Nodes
-----------------------------------------
LangGraph state carries only control-flow fields.  The GoalGroupWizard DB
table remains the authoritative store for all business data (target_specs,
constraints, draft_plan_ids, reference_materials, …).  Each node:

  1. Calls interrupt() to suspend execution and receive human input.
  2. Opens its own AsyncSessionLocal session to load the wizard from DB.
  3. Delegates to the existing wizard_service.* function (unchanged).
  4. Commits and returns updated control-flow state.

Graph topology:
  START → scope → targets → save_constraints
                               → research
                                 → generate_plans
                                   → feasibility
                                     → human_gate
                                         ├─ confirm → END
                                         ├─ adjust  → research (loop)
                                         └─ cancel  → END

Any node can also be cancelled by passing {"action": "cancel"} in the
resume payload — all interrupt points check for this and route to cancel.
"""

from __future__ import annotations

import logging
from typing import Any
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt  # noqa: F401 — Command re-exported for callers

from app.crud import wizards as crud_wizard
from app.database import AsyncSessionLocal
from app.models.goal_group_wizard import TERMINAL_STATUSES as _TERMINAL_STATUSES
from app.services import wizard_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class WizardState(TypedDict):
    wizard_id: int
    go_getter_id: int
    status: str  # mirrors WizardStatus enum value
    human_decision: str  # "confirm" | "adjust" | "cancel" | ""
    error: str  # non-empty on node failure
    adjust_patch: dict  # patch dict passed from human_gate → adjust_node
    confirm_result: dict  # populated by confirm_node: goal_group metadata


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


async def scope_node(state: WizardState) -> dict[str, Any]:
    """Interrupt for scope data; call wizard_service.set_scope() on resume."""
    data = interrupt({"awaiting": "scope"})
    if data.get("action") == "cancel":
        return {**state, "human_decision": "cancel"}
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.set_scope(
            db,
            wizard,
            title=data["title"],
            description=data.get("description"),
            start_date=data["start_date"],
            end_date=data["end_date"],
        )
        await db.commit()
    return {**state, "status": "collecting_targets", "human_decision": "", "error": ""}


async def targets_node(state: WizardState) -> dict[str, Any]:
    """Interrupt for target specs; call wizard_service.set_targets() on resume."""
    data = interrupt({"awaiting": "targets"})
    if data.get("action") == "cancel":
        return {**state, "human_decision": "cancel"}
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.set_targets(
            db,
            wizard,
            target_specs=data["target_specs"],
        )
        await db.commit()
    return {**state, "status": "collecting_constraints", "human_decision": "", "error": ""}


async def save_constraints_node(state: WizardState) -> dict[str, Any]:
    """Interrupt for constraints; save to DB and transition to generating_plans.

    The heavy work (web research, plan gen, feasibility) is handled by the
    subsequent non-interrupt nodes so each step gets its own checkpoint.
    """
    data = interrupt({"awaiting": "constraints"})
    if data.get("action") == "cancel":
        return {**state, "human_decision": "cancel"}
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.save_constraints_to_db(
            db,
            wizard,
            constraints=data["constraints"],
        )
        await db.commit()
    return {**state, "status": "generating_plans", "human_decision": "", "error": ""}


async def research_node(state: WizardState) -> dict[str, Any]:
    """Run web research for all targets (no interrupt, no user input needed).

    Skips its work if the wizard has reached a terminal state (e.g. cancelled
    mid-generation via the direct-DB cancel path).
    """
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        if wizard is None or wizard.status in _TERMINAL_STATUSES:
            logger.info("research_node: wizard %d is terminal, skipping", state["wizard_id"])
            return {**state, "error": ""}
        await wizard_service.run_web_research_step(db, wizard)
        await db.commit()
    return {**state, "error": ""}


async def generate_plans_node(state: WizardState) -> dict[str, Any]:
    """Generate draft plans in parallel for all targets (no interrupt).

    Skips if the wizard is terminal.
    """
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        if wizard is None or wizard.status in _TERMINAL_STATUSES:
            logger.info("generate_plans_node: wizard %d is terminal, skipping", state["wizard_id"])
            return {**state, "error": ""}
        plan_ids, errors = await wizard_service.generate_plans_parallel(db, wizard)
        await wizard_service.save_plan_gen_results(db, wizard, plan_ids, errors)
        await db.commit()
    return {**state, "error": ""}


async def feasibility_node(state: WizardState) -> dict[str, Any]:
    """Run feasibility check + LLM enrichment (no interrupt).

    Skips if the wizard is terminal.
    """
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        if wizard is None or wizard.status in _TERMINAL_STATUSES:
            logger.info("feasibility_node: wizard %d is terminal, skipping", state["wizard_id"])
            return {**state, "error": ""}
        await wizard_service.run_feasibility_step(db, wizard)
        await db.commit()
    return {**state, "status": "feasibility_check", "error": ""}


async def human_gate_node(state: WizardState) -> dict[str, Any]:
    """Interrupt to present plan / feasibility check to BestPal.

    Checks terminal state BEFORE calling interrupt(): if the wizard was cancelled
    mid-generation (direct-DB path) the graph routes to cancel_node and reaches
    END without ever pausing at human_gate.

    Resume with:
      {"decision": "confirm"}
      {"decision": "adjust",  "patch": {…}}
      {"decision": "cancel"}
      {"action": "cancel"}   ← from REST/MCP cancel endpoint
    """
    # Load and close the session before interrupt() so no connection is held
    # across the suspension point.
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
    if wizard is None or wizard.status in _TERMINAL_STATUSES:
        logger.info("human_gate_node: wizard %d is terminal, routing to cancel", state["wizard_id"])
        return {**state, "human_decision": "cancel"}

    data = interrupt({"awaiting": "human_decision"})
    # Support cancellation from the REST cancel endpoint
    if data.get("action") == "cancel":
        return {**state, "human_decision": "cancel"}
    decision = data.get("decision", "")
    patch = data.get("patch", {})
    return {**state, "human_decision": decision, "adjust_patch": patch}


async def confirm_node(state: WizardState) -> dict[str, Any]:
    """Create GoalGroup and activate draft plans via wizard_service.confirm()."""
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        group, superseded_plans = await wizard_service.confirm(db, wizard)
        await db.commit()
    return {
        **state,
        "status": "confirmed",
        "human_decision": "",
        "confirm_result": {
            "goal_group_id": group.id,
            "title": group.title,
            "go_getter_id": group.go_getter_id,
            "start_date": str(group.start_date),
            "end_date": str(group.end_date),
            "superseded_plans": superseded_plans,
        },
    }


async def adjust_node(state: WizardState) -> dict[str, Any]:
    """Save adjust patch and cancel old drafts; routes to research_node for re-generation."""
    patch = state.get("adjust_patch") or {}
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.save_adjust_patch(db, wizard, patch=patch)
        await db.commit()
    return {**state, "status": "generating_plans", "human_decision": "", "adjust_patch": {}}


async def cancel_node(state: WizardState) -> dict[str, Any]:
    """Cancel wizard and all draft plans via wizard_service.cancel_wizard()."""
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.cancel_wizard(db, wizard)
        await db.commit()
    return {**state, "status": "cancelled", "human_decision": ""}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _cancel_or(state: WizardState, next_node: str) -> str:
    if state.get("human_decision") == "cancel":
        return "cancel"
    return next_node


def route_after_scope(state: WizardState) -> str:
    return _cancel_or(state, "targets")


def route_after_targets(state: WizardState) -> str:
    return _cancel_or(state, "save_constraints")


def route_after_save_constraints(state: WizardState) -> str:
    return _cancel_or(state, "research")


def route_human_decision(state: WizardState) -> str:
    decision = state.get("human_decision", "")
    if decision in ("confirm", "adjust", "cancel"):
        return decision
    # Unexpected value — re-interrupt at human_gate
    return "human_gate"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_wizard_graph(checkpointer):
    """Build and compile the wizard StateGraph with the given checkpointer.

    Args:
        checkpointer: a LangGraph checkpointer (MemorySaver for tests,
                      AsyncSqliteSaver for dev/prod).

    Returns:
        Compiled CompiledGraph ready for ainvoke / aget_state.
    """
    builder = StateGraph(WizardState)

    builder.add_node("scope", scope_node)
    builder.add_node("targets", targets_node)
    builder.add_node("save_constraints", save_constraints_node)
    builder.add_node("research", research_node)
    builder.add_node("generate_plans", generate_plans_node)
    builder.add_node("feasibility", feasibility_node)
    builder.add_node("human_gate", human_gate_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("adjust", adjust_node)
    builder.add_node("cancel", cancel_node)

    builder.add_edge(START, "scope")
    builder.add_conditional_edges(
        "scope", route_after_scope, {"targets": "targets", "cancel": "cancel"}
    )
    builder.add_conditional_edges(
        "targets",
        route_after_targets,
        {"save_constraints": "save_constraints", "cancel": "cancel"},
    )
    builder.add_conditional_edges(
        "save_constraints",
        route_after_save_constraints,
        {"research": "research", "cancel": "cancel"},
    )
    builder.add_edge("research", "generate_plans")
    builder.add_edge("generate_plans", "feasibility")
    builder.add_edge("feasibility", "human_gate")
    builder.add_conditional_edges(
        "human_gate",
        route_human_decision,
        {
            "confirm": "confirm",
            "adjust": "adjust",
            "cancel": "cancel",
            "human_gate": "human_gate",
        },
    )
    builder.add_edge("adjust", "research")
    builder.add_edge("confirm", END)
    builder.add_edge("cancel", END)

    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Module-level singleton (MemorySaver default; override for production)
# ---------------------------------------------------------------------------

_default_graph = None


def get_wizard_graph():
    """Return the module-level wizard graph singleton.

    Defaults to MemorySaver (in-memory, process-scoped).  For persistent
    checkpointing across restarts, call set_wizard_graph() at startup with
    a graph built from AsyncSqliteSaver.
    """
    global _default_graph
    if _default_graph is None:
        from langgraph.checkpoint.memory import MemorySaver

        _default_graph = build_wizard_graph(MemorySaver())
    return _default_graph


def set_wizard_graph(graph) -> None:
    """Override the default graph singleton (e.g., to use AsyncSqliteSaver in prod)."""
    global _default_graph
    _default_graph = graph


# ---------------------------------------------------------------------------
# Step-ordering guard
# ---------------------------------------------------------------------------

# Nodes that pause via interrupt() and can receive a resume payload.
# Non-interrupt nodes (research, generate_plans, feasibility) are NOT here.
INTERRUPT_NODES: frozenset[str] = frozenset({"scope", "targets", "save_constraints", "human_gate"})

_FRIENDLY_NODE_NAMES: dict[str, str] = {
    "scope": "POST /scope",
    "targets": "POST /targets",
    "save_constraints": "POST /constraints",
    "human_gate": "POST /confirm or POST /adjust",
}


async def assert_graph_awaiting(graph, wizard_id: int, expected_node: str) -> None:
    """Assert the graph is paused at *expected_node*.

    Raises ValueError if:
    - The graph thread is not active (completed or never started).
    - The graph is currently running a non-interrupt processing node.
    - The graph is paused at a *different* interrupt node (out-of-order call).

    Callers (REST endpoints, MCP tools) convert the ValueError to 4xx.
    """
    config = {"configurable": {"thread_id": str(wizard_id)}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.next:
        raise ValueError(
            f"Wizard {wizard_id}: graph thread is not active "
            "(it may have completed or not been initialised)."
        )
    actual = snapshot.next[0]
    if actual not in INTERRUPT_NODES:
        raise ValueError(
            f"Wizard {wizard_id} is currently processing (step: '{actual}'). "
            "Wait for generation to complete before retrying."
        )
    if actual != expected_node:
        raise ValueError(
            f"Out-of-order call: wizard is waiting at "
            f"'{_FRIENDLY_NODE_NAMES.get(actual, actual)}', "
            f"but you called '{_FRIENDLY_NODE_NAMES.get(expected_node, expected_node)}'. "
            "Call steps in the correct order."
        )
