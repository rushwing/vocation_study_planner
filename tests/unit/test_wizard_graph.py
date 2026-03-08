"""Unit tests for the LangGraph wizard graph orchestration layer.

These tests exercise the graph routing, interrupt behaviour, and error
propagation using MemorySaver (in-memory, no file I/O) and mocked
wizard_service / crud calls — the business logic is tested separately in
test_wizard_service.py and test_feasibility_service.py.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.services.wizard_graph import (
    INTERRUPT_NODES,
    WizardState,
    assert_graph_awaiting,
    build_wizard_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpointer():
    return MemorySaver()


@pytest.fixture
def graph(checkpointer):
    return build_wizard_graph(checkpointer)


def _initial_state(wizard_id: int = 1, go_getter_id: int = 42) -> WizardState:
    return {
        "wizard_id": wizard_id,
        "go_getter_id": go_getter_id,
        "status": "collecting_scope",
        "human_decision": "",
        "error": "",
        "adjust_patch": {},
        "confirm_result": {},
    }


def _config(wizard_id: int = 1) -> dict[str, Any]:
    return {"configurable": {"thread_id": str(wizard_id)}}


# ---------------------------------------------------------------------------
# Shared mock factory
# ---------------------------------------------------------------------------


def _make_mocks(wizard_id: int = 1, go_getter_id: int = 42):
    """Return (session_factory, mock_db, mock_crud, mock_svc, mock_wizard, mock_group)."""
    mock_wizard = MagicMock()
    mock_wizard.id = wizard_id
    mock_wizard.go_getter_id = go_getter_id

    mock_group = MagicMock()
    mock_group.id = 100
    mock_group.title = "Test Group"
    mock_group.go_getter_id = go_getter_id
    mock_group.start_date = date.today()
    mock_group.end_date = date.today() + timedelta(days=30)

    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_db)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=session_cm)

    mock_crud = MagicMock()
    mock_crud.get = AsyncMock(return_value=mock_wizard)

    mock_svc = MagicMock()
    mock_svc.set_scope = AsyncMock(return_value=mock_wizard)
    mock_svc.set_targets = AsyncMock(return_value=mock_wizard)
    mock_svc.set_constraints = AsyncMock(return_value=mock_wizard)
    mock_svc.confirm = AsyncMock(return_value=(mock_group, []))
    mock_svc.adjust = AsyncMock(return_value=mock_wizard)
    mock_svc.cancel_wizard = AsyncMock(return_value=None)
    # New graph-node service functions
    mock_svc.save_constraints_to_db = AsyncMock(return_value=mock_wizard)
    mock_svc.run_web_research_step = AsyncMock(return_value=mock_wizard)
    mock_svc.generate_plans_parallel = AsyncMock(return_value=([], []))
    mock_svc.save_plan_gen_results = AsyncMock(return_value=mock_wizard)
    mock_svc.run_feasibility_step = AsyncMock(return_value=mock_wizard)
    mock_svc.save_adjust_patch = AsyncMock(return_value=mock_wizard)

    return session_factory, mock_db, mock_crud, mock_svc, mock_wizard, mock_group


# ---------------------------------------------------------------------------
# Helper: step through the graph to the human_gate interrupt
# ---------------------------------------------------------------------------


async def _advance_to_human_gate(graph, wizard_id: int = 1) -> None:
    """Run scope → targets → save_constraints → research → generate_plans → feasibility
    then stop at human_gate interrupt."""
    config = _config(wizard_id)
    start_date = date.today()
    end_date = start_date + timedelta(days=30)

    await graph.ainvoke(
        Command(
            resume={
                "title": "Study Plan",
                "start_date": start_date,
                "end_date": end_date,
            }
        ),
        config=config,
    )
    await graph.ainvoke(
        Command(resume={"target_specs": [{"target_id": 1, "subcategory_id": 1, "priority": 3}]}),
        config=config,
    )
    await graph.ainvoke(
        Command(resume={"constraints": {"1": {"daily_minutes": 30, "preferred_days": [0, 1, 2]}}}),
        config=config,
    )


# ---------------------------------------------------------------------------
# Tests: interrupt behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initial_ainvoke_interrupts_at_scope(graph):
    """Graph should interrupt immediately at scope_node after initialisation."""
    config = _config()
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(), config=config)

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("scope",)
        # scope_node has not yet called set_scope (waiting for human input)
        mock_svc.set_scope.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_scope_to_confirm(graph):
    """Full happy path: scope → targets → save_constraints → research → generate_plans
    → feasibility → human_gate → confirm → END."""
    wizard_id = 1
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, mock_group = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        # Initialise — scope interrupt
        await graph.ainvoke(_initial_state(wizard_id), config=config)

        # Set scope
        start_date = date.today()
        end_date = start_date + timedelta(days=30)
        await graph.ainvoke(
            Command(resume={"title": "Study Plan", "start_date": start_date, "end_date": end_date}),
            config=config,
        )
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("targets",)

        # Set targets
        await graph.ainvoke(
            Command(
                resume={"target_specs": [{"target_id": 1, "subcategory_id": 1, "priority": 3}]}
            ),
            config=config,
        )
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("save_constraints",)

        # Set constraints — graph auto-advances through research/generate/feasibility
        await graph.ainvoke(
            Command(resume={"constraints": {"1": {"daily_minutes": 30, "preferred_days": [0]}}}),
            config=config,
        )
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("human_gate",)

        # Confirm
        result = await graph.ainvoke(
            Command(resume={"decision": "confirm"}),
            config=config,
        )

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()  # graph complete

        assert result["status"] == "confirmed"
        assert result["confirm_result"]["goal_group_id"] == mock_group.id
        assert result["confirm_result"]["superseded_plans"] == []

        # All service calls were made once
        mock_svc.set_scope.assert_called_once()
        mock_svc.set_targets.assert_called_once()
        mock_svc.save_constraints_to_db.assert_called_once()
        mock_svc.run_web_research_step.assert_called_once()
        mock_svc.generate_plans_parallel.assert_called_once()
        mock_svc.save_plan_gen_results.assert_called_once()
        mock_svc.run_feasibility_step.assert_called_once()
        mock_svc.confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: adjust loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjust_loop_then_confirm(graph):
    """human_gate → adjust → research → generate_plans → feasibility → human_gate → confirm."""
    wizard_id = 2
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, mock_group = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("human_gate",)

        # Adjust — routes through research/generate/feasibility back to human_gate
        patch_data = {"constraints": {"1": {"daily_minutes": 45, "preferred_days": [1, 3]}}}
        await graph.ainvoke(
            Command(resume={"decision": "adjust", "patch": patch_data}),
            config=config,
        )
        # After adjust loop, should be back at human_gate
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("human_gate",)
        mock_svc.save_adjust_patch.assert_called_once()

        # Confirm this time
        result = await graph.ainvoke(
            Command(resume={"decision": "confirm"}),
            config=config,
        )
        assert result["status"] == "confirmed"
        mock_svc.confirm.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: cancel paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_from_human_gate(graph):
    """Cancel decision at human_gate routes to cancel_node → END."""
    wizard_id = 3
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        result = await graph.ainvoke(
            Command(resume={"decision": "cancel"}),
            config=config,
        )

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()  # graph complete

        assert result["status"] == "cancelled"
        mock_svc.cancel_wizard.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_from_scope_via_action(graph):
    """Passing {'action': 'cancel'} at scope interrupt routes to cancel_node."""
    wizard_id = 4
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        # Cancel while at scope node
        result = await graph.ainvoke(
            Command(resume={"action": "cancel"}),
            config=config,
        )

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()

        assert result["status"] == "cancelled"
        mock_svc.cancel_wizard.assert_called_once()
        # scope was never called (cancelled before set_scope)
        mock_svc.set_scope.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_from_targets_via_action(graph):
    """Passing {'action': 'cancel'} at targets interrupt routes to cancel_node."""
    wizard_id = 5
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        # Advance past scope
        start_date = date.today()
        await graph.ainvoke(
            Command(
                resume={
                    "title": "Plan",
                    "start_date": start_date,
                    "end_date": start_date + timedelta(days=30),
                }
            ),
            config=config,
        )
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("targets",)

        # Cancel at targets
        result = await graph.ainvoke(
            Command(resume={"action": "cancel"}),
            config=config,
        )

        assert result["status"] == "cancelled"
        mock_svc.set_scope.assert_called_once()  # scope completed
        mock_svc.set_targets.assert_not_called()  # targets never ran
        mock_svc.cancel_wizard.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_from_human_gate_via_action(graph):
    """{'action': 'cancel'} at human_gate also cancels correctly."""
    wizard_id = 6
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        result = await graph.ainvoke(
            Command(resume={"action": "cancel"}),
            config=config,
        )

        assert result["status"] == "cancelled"
        mock_svc.cancel_wizard.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_in_set_scope_propagates(graph):
    """ValueError from wizard_service.set_scope() propagates from ainvoke."""
    wizard_id = 7
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)
    mock_svc.set_scope = AsyncMock(side_effect=ValueError("date range too short"))

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        with pytest.raises(ValueError, match="date range too short"):
            await graph.ainvoke(
                Command(
                    resume={
                        "title": "Bad Plan",
                        "start_date": date.today(),
                        "end_date": date.today() + timedelta(days=1),
                    }
                ),
                config=config,
            )


@pytest.mark.asyncio
async def test_error_in_confirm_propagates(graph):
    """ValueError from wizard_service.confirm() propagates from ainvoke."""
    wizard_id = 8
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)
    mock_svc.confirm = AsyncMock(side_effect=ValueError("feasibility blockers present"))

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        with pytest.raises(ValueError, match="feasibility blockers"):
            await graph.ainvoke(
                Command(resume={"decision": "confirm"}),
                config=config,
            )


# ---------------------------------------------------------------------------
# Tests: multiple independent wizard threads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_independent_threads_do_not_interfere(checkpointer):
    """Two wizards with different wizard_ids run as independent threads."""
    graph = build_wizard_graph(checkpointer)

    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        # Initialise two separate threads
        await graph.ainvoke(_initial_state(wizard_id=10), config=_config(10))
        await graph.ainvoke(_initial_state(wizard_id=11), config=_config(11))

        # Both should be at scope interrupt
        snap_10 = await graph.aget_state(_config(10))
        snap_11 = await graph.aget_state(_config(11))
        assert snap_10.next == ("scope",)
        assert snap_11.next == ("scope",)

        # Advance thread 10 past scope — thread 11 should be unaffected
        await graph.ainvoke(
            Command(
                resume={
                    "title": "Plan A",
                    "start_date": date.today(),
                    "end_date": date.today() + timedelta(days=30),
                }
            ),
            config=_config(10),
        )
        snap_10 = await graph.aget_state(_config(10))
        snap_11 = await graph.aget_state(_config(11))
        assert snap_10.next == ("targets",)
        assert snap_11.next == ("scope",)  # unchanged


# ---------------------------------------------------------------------------
# Tests: parallel plan generation node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_plan_gen_node_calls_service(graph):
    """generate_plans_node calls generate_plans_parallel and save_plan_gen_results."""
    wizard_id = 12
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)
    mock_svc.generate_plans_parallel = AsyncMock(return_value=([101, 102], []))

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        # By the time we reach human_gate, generate_plans_parallel was called once
        mock_svc.generate_plans_parallel.assert_called_once()
        mock_svc.save_plan_gen_results.assert_called_once()

        # Verify save_plan_gen_results received the plan_ids from generate_plans_parallel
        call_args = mock_svc.save_plan_gen_results.call_args
        _db, _wizard, plan_ids, errors = call_args.args
        assert plan_ids == [101, 102]
        assert errors == []


# ---------------------------------------------------------------------------
# Tests: assert_graph_awaiting step-ordering guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assert_graph_awaiting_passes_at_correct_node(graph):
    """assert_graph_awaiting does not raise when graph is at the expected node."""
    wizard_id = 13
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("scope",)

        # Should not raise — graph IS at scope
        await assert_graph_awaiting(graph, wizard_id, "scope")


@pytest.mark.asyncio
async def test_assert_graph_awaiting_raises_on_wrong_node(graph):
    """assert_graph_awaiting raises ValueError for out-of-order calls."""
    wizard_id = 14
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        # Graph is at scope but we check for targets — should raise
        with pytest.raises(ValueError, match="Out-of-order"):
            await assert_graph_awaiting(graph, wizard_id, "targets")


@pytest.mark.asyncio
async def test_assert_graph_awaiting_raises_when_at_human_gate_but_scope_expected(graph):
    """Calling /scope while wizard is at human_gate raises ValueError."""
    wizard_id = 15
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        snapshot = await graph.aget_state(config)
        assert snapshot.next == ("human_gate",)

        with pytest.raises(ValueError, match="Out-of-order"):
            await assert_graph_awaiting(graph, wizard_id, "scope")


@pytest.mark.asyncio
async def test_assert_graph_awaiting_raises_when_completed(graph):
    """assert_graph_awaiting raises when the graph thread has finished."""
    wizard_id = 16
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)
        await graph.ainvoke(Command(resume={"decision": "confirm"}), config=config)

        # Graph is finished (next == ()), so any check raises
        with pytest.raises(ValueError, match="not active"):
            await assert_graph_awaiting(graph, wizard_id, "human_gate")


# ---------------------------------------------------------------------------
# Tests: cancel routing (interrupt vs. non-interrupt state)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_nodes_constant_contains_expected_values():
    """INTERRUPT_NODES contains exactly the four interrupt points."""
    assert {"scope", "targets", "save_constraints", "human_gate"} == INTERRUPT_NODES


@pytest.mark.asyncio
async def test_cancel_at_interrupt_node_is_detectable_via_snapshot(graph):
    """When graph is at scope, snapshot.next[0] is in INTERRUPT_NODES (graph cancel path)."""
    wizard_id = 17
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        snapshot = await graph.aget_state(config)

        assert snapshot.next[0] in INTERRUPT_NODES  # cancel should route via graph

        # Confirm the graph cancel path works end-to-end from this state
        result = await graph.ainvoke(Command(resume={"action": "cancel"}), config=config)
        assert result["status"] == "cancelled"
        mock_svc.cancel_wizard.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_after_completion_is_not_via_interrupt(graph):
    """When graph has finished, snapshot.next is empty — cancel should use DB path."""
    wizard_id = 18
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, _, _ = _make_mocks(wizard_id)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)
        await graph.ainvoke(Command(resume={"decision": "confirm"}), config=config)

        snapshot = await graph.aget_state(config)
        assert not snapshot.next  # graph completed

        # The REST/MCP cancel layer should detect this and use DB path (no graph resume)
        at_interrupt = bool(snapshot and snapshot.next and snapshot.next[0] in INTERRUPT_NODES)
        assert not at_interrupt  # confirms DB-cancel branch would be taken


# ---------------------------------------------------------------------------
# Tests: terminal-state guards in non-interrupt nodes
# ---------------------------------------------------------------------------


def _terminal_wizard_mocks(wizard_id: int = 1, go_getter_id: int = 42):
    """Return (session_factory, mock_db, mock_crud, mock_svc) with a CANCELLED wizard."""
    from app.models.goal_group_wizard import WizardStatus

    cancelled_wizard = MagicMock()
    cancelled_wizard.id = wizard_id
    cancelled_wizard.go_getter_id = go_getter_id
    cancelled_wizard.status = WizardStatus.cancelled

    mock_db = AsyncMock()
    mock_db.commit = AsyncMock()

    session_cm = AsyncMock()
    session_cm.__aenter__ = AsyncMock(return_value=mock_db)
    session_cm.__aexit__ = AsyncMock(return_value=False)
    session_factory = MagicMock(return_value=session_cm)

    mock_crud = MagicMock()
    mock_crud.get = AsyncMock(return_value=cancelled_wizard)

    mock_svc = MagicMock()
    mock_svc.run_web_research_step = AsyncMock()
    mock_svc.generate_plans_parallel = AsyncMock(return_value=([], []))
    mock_svc.save_plan_gen_results = AsyncMock()
    mock_svc.run_feasibility_step = AsyncMock()

    return session_factory, mock_db, mock_crud, mock_svc


@pytest.mark.asyncio
async def test_research_node_skips_when_wizard_is_terminal():
    """research_node exits early without calling run_web_research_step if wizard is terminal."""
    from app.services.wizard_graph import research_node

    session_factory, _, mock_crud, mock_svc = _terminal_wizard_mocks()
    state = _initial_state()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        result = await research_node(state)

    mock_svc.run_web_research_step.assert_not_called()
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_generate_plans_node_skips_when_wizard_is_terminal():
    """generate_plans_node exits early without generating or saving plans if terminal."""
    from app.services.wizard_graph import generate_plans_node

    session_factory, _, mock_crud, mock_svc = _terminal_wizard_mocks()
    state = _initial_state()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        result = await generate_plans_node(state)

    mock_svc.generate_plans_parallel.assert_not_called()
    mock_svc.save_plan_gen_results.assert_not_called()
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_feasibility_node_skips_when_wizard_is_terminal():
    """feasibility_node exits early without calling run_feasibility_step if terminal."""
    from app.services.wizard_graph import feasibility_node

    session_factory, _, mock_crud, mock_svc = _terminal_wizard_mocks()
    state = _initial_state()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        result = await feasibility_node(state)

    mock_svc.run_feasibility_step.assert_not_called()
    assert result["error"] == ""


@pytest.mark.asyncio
async def test_human_gate_skips_interrupt_when_wizard_is_terminal():
    """human_gate_node returns human_decision='cancel' without calling interrupt() if terminal.

    Outside a graph execution context interrupt() would raise GraphInterrupt, so if
    this test passes cleanly it proves interrupt() was never reached.
    """
    from app.services.wizard_graph import human_gate_node

    session_factory, _, mock_crud, _ = _terminal_wizard_mocks()
    state = _initial_state()

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
    ):
        result = await human_gate_node(state)

    assert result["human_decision"] == "cancel"


@pytest.mark.asyncio
async def test_mid_gen_cancel_graph_reaches_end(graph):
    """Full integration: if wizard is terminal when generation nodes run, graph reaches END.

    Simulates a mid-generation cancel by making crud_wizard.get return a normal
    wizard for the first 3 calls (scope/targets/save_constraints) then return a
    cancelled wizard for all subsequent calls (research/gen/feasibility/human_gate).
    """
    from app.models.goal_group_wizard import WizardStatus

    wizard_id = 19
    config = _config(wizard_id)
    session_factory, _, mock_crud, mock_svc, mock_wizard, _ = _make_mocks(wizard_id)

    cancelled_wizard = MagicMock()
    cancelled_wizard.id = wizard_id
    cancelled_wizard.go_getter_id = 42
    cancelled_wizard.status = WizardStatus.cancelled

    call_count = 0

    async def get_side_effect(db, wid):
        nonlocal call_count
        call_count += 1
        # First 3 calls: scope_node, targets_node, save_constraints_node (each calls get once)
        return mock_wizard if call_count <= 3 else cancelled_wizard

    mock_crud.get = AsyncMock(side_effect=get_side_effect)

    with (
        patch("app.services.wizard_graph.AsyncSessionLocal", session_factory),
        patch("app.services.wizard_graph.crud_wizard", mock_crud),
        patch("app.services.wizard_graph.wizard_service", mock_svc),
    ):
        await graph.ainvoke(_initial_state(wizard_id), config=config)
        await _advance_to_human_gate(graph, wizard_id)

        # Processing nodes should have been skipped
        mock_svc.run_web_research_step.assert_not_called()
        mock_svc.generate_plans_parallel.assert_not_called()
        mock_svc.run_feasibility_step.assert_not_called()

        # human_gate saw terminal wizard → routed to cancel → cancel_wizard called
        mock_svc.cancel_wizard.assert_called_once()

        # Graph should have reached END (no next node)
        snapshot = await graph.aget_state(config)
        assert snapshot.next == ()
