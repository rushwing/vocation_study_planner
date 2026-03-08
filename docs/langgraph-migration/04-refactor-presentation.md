# LangGraph Refactor: Migrating a Multi-Step Wizard to Explicit State Graphs

> **Audience**: Engineering peer review / conference talk notes
> **Project**: goal-agent — AI study planning assistant for K-12 students
> **Date**: 2026-03-09

---

## 1. Business Context: What We Were Building

A multi-step, human-in-the-loop wizard for creating structured study plans.

```
Parent (BestPal)  ──Telegram──►  OpenClaw LLM  ──MCP──►  goal-agent API
                                                               │
                                                               └──► Kimi LLM (plan gen)
                                                               └──► Jina Search (research)
```

**The wizard flow** guides a parent through five decision points:

```
Set time window  →  Choose targets  →  Set constraints
        →  [LLM: web research + plan generation + feasibility check]
                →  Parent reviews & confirms  →  GoalGroup activated
```

**Constraints that make this hard**:

- **Long-running LLM steps**: web research + plan generation can take 30–90s per wizard run
- **Human gate**: parent must explicitly review and confirm before anything is activated
- **Cross-session**: Telegram chat can drop; wizard must resume exactly where it left off
- **Cooperative cancel**: parent can cancel at any point, including during LLM generation
- **Multiple users**: each parent has their own independent wizard thread

---

## 2. The Original System and Its Pain Points

### What we had

A hand-rolled state machine in `wizard_service.py`:

```python
# wizard_service.py — before
async def set_constraints(db, wizard, *, constraints):
    _assert_not_terminal(wizard)
    wizard = await crud_wizard.update_wizard(db, wizard,
        status=WizardStatus.generating_plans, constraints=constraints)
    await db.commit()
    await _generate_and_check(db, wizard)   # ← everything in one shot
    ...

async def _generate_and_check(db, wizard):
    await _run_web_research(db, wizard)
    for spec in wizard.target_specs:        # ← serial loop, N × 30s
        new_plan = await plan_generator.generate_plan(...)
    await check_feasibility(db, wizard)
    wizard = await crud_wizard.update_wizard(db, wizard,
        status=WizardStatus.feasibility_check)
```

### Pain points (measured against baseline)

| Pain Point | Evidence |
|------------|----------|
| **P1 — State flow invisible** | 10 manual `status=WizardStatus.*` assignments across 3 functions; no diagram, no single routing function |
| **P2 — Human gate by convention only** | `confirm()` guard in one function + docstring + OpenClaw prompt — three separate places enforcing the same rule |
| **P3 — No crash recovery granularity** | `_generate_and_check()` is atomic from the checkpoint perspective; crash mid-way → restart from `collecting_constraints` (web research + all plans must re-run) |
| **P4 — Serial plan generation** | `for spec in target_specs` — 3 targets × ~30s = ~90s wall time |
| **P5 — Cancel semantics undefined** | Cancel endpoint wrote DB but running LLM calls were unaware; subsequent node writes could overwrite `cancelled` status |
| **P6 — Out-of-order calls return 500** | No validation that graph is at the right step; calling `/scope` twice caused internal `KeyError` |

---

## 3. Why LangGraph — and What We Were Actually Solving

We didn't pick LangGraph because it's popular. We picked it because our problem maps directly to its primitives:

| Our problem | LangGraph primitive |
|-------------|---------------------|
| Explicit state flow | `StateGraph` with named nodes + edges |
| Human gate (protocol-level pause) | `interrupt()` — suspends execution, can only resume via `Command(resume=...)` |
| Crash recovery at node granularity | Automatic checkpoint after each node (Checkpointer) |
| Long-running LLM steps as units | Each node is a recoverable checkpoint boundary |
| Per-user isolation | `thread_id` in config → separate checkpoint thread per wizard |
| Resume across process restarts | `AsyncSqliteSaver` / `AsyncPostgresSaver` persist checkpoint state |

---

## 4. Design Decisions: What We Did and Didn't Do

### Decision 1: Write-Through + Thin Nodes (keep DB as source of truth)

**Option A**: Let LangGraph state hold all business data (target_specs, constraints, reference_materials…); Checkpointer manages everything.

**Option B**: LangGraph state holds only control-flow fields; `GoalGroupWizard` DB table stays authoritative.

**We chose B.** Each node opens its own `AsyncSessionLocal`, reads the wizard from DB, calls a service function, commits, and returns only a small state delta.

```python
# Thin node — only control flow in state
async def research_node(state: WizardState) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        if wizard.status in _TERMINAL_STATUSES:  # terminal guard
            return {**state, "error": ""}
        await wizard_service.run_web_research_step(db, wizard)
        await db.commit()
    return {**state, "error": ""}
```

**Why B wins here**: existing REST read paths, TTL management, and monitoring queries all read `GoalGroupWizard`. Option A would require migrating all readers. Option B adds LangGraph's control-flow checkpoint value with minimal migration surface.

**Trade-off**: each node incurs one extra DB read. Acceptable for a wizard flow (seconds between steps).

---

### Decision 2: asyncio.gather for parallelism, not LangGraph Send

**Original plan**: use LangGraph `Send` API for fan-out — one sub-node per target, running in parallel.

**Why we didn't**: `Send` requires each branch to be an independent sub-graph node, and fan-in (collecting all plan IDs back to the wizard) needs a reducer or aggregation node. Combined with our Write-Through model (each target must atomically append its plan ID to the wizard row), the fan-in complexity outweighed the benefit.

**What we did instead**: single `generate_plans_node` using `asyncio.gather` with independent sessions per target:

```python
async def generate_plans_parallel(db, wizard):
    async def _generate_one(spec):
        async with AsyncSessionLocal() as gen_db:  # independent session
            new_plan = await plan_generator.generate_plan(..., wizard_id=wizard.id)
            # Atomic append — same transaction as plan creation
            w = await gen_db.execute(
                select(GoalGroupWizard).where(...).with_for_update()
            )
            w.draft_plan_ids = list(w.draft_plan_ids or []) + [new_plan.id]
            await gen_db.commit()
            return new_plan.id, None

    results = await asyncio.gather(
        *[_generate_one(spec) for spec in wizard.target_specs],
        return_exceptions=True,
    )
```

**Result**: plan generation time goes from O(N) serial to O(1) parallel (bounded by single-plan LLM latency). Simpler code, same concurrency, no fan-in complexity.

> **Lesson**: Use LangGraph `Send` when each branch genuinely has multi-step sub-graph logic. For "parallel LLM calls with a single aggregation write", `asyncio.gather` is simpler and more reliable.

---

### Decision 3: AsyncSqliteSaver in FastAPI lifespan (not lazy singleton)

`AsyncSqliteSaver.from_conn_string()` is an async context manager. It must be opened before the first graph call and held open until the process exits.

```python
# app/main.py — correct pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(_DEV_CHECKPOINTER_PATH) as checkpointer:
        set_wizard_graph(build_wizard_graph(checkpointer))
        yield  # entire app lifetime inside this context
```

Any other initialization pattern (lazy import, global variable, module-level instantiation) either fails to open the connection properly or leaks it on shutdown.

---

## 5. Key Implementation Details

### 5a. interrupt/resume for the human gate

```python
async def human_gate_node(state: WizardState) -> dict[str, Any]:
    # 1. Load wizard and close session BEFORE interrupt()
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
    # 2. Check terminal state — if cancelled mid-gen, skip interrupt entirely
    if wizard is None or wizard.status in _TERMINAL_STATUSES:
        return {**state, "human_decision": "cancel"}
    # 3. Suspend — no DB connection held during suspension
    data = interrupt({"awaiting": "human_decision"})
    decision = data.get("decision", "")
    return {**state, "human_decision": decision, "adjust_patch": data.get("patch", {})}
```

**Why close the session before `interrupt()`**: `interrupt()` suspends the entire coroutine. An open `AsyncSession` would hold a DB connection for the full suspension period (potentially minutes), exhausting the connection pool.

**Why check terminal before `interrupt()`**: if the wizard was cancelled via the direct DB path while generation was running, we must not leave a dangling interrupt checkpoint that requires an external resume to clear.

---

### 5b. assert_graph_awaiting — preventing out-of-order calls

Every step endpoint validates graph state before invoking:

```python
async def assert_graph_awaiting(graph, wizard_id: int, expected_node: str) -> None:
    snapshot = await graph.aget_state({"configurable": {"thread_id": str(wizard_id)}})
    if not snapshot or not snapshot.next:
        raise ValueError(f"Wizard {wizard_id}: graph thread is not active.")
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
```

Before this guard, calling `/scope` when the graph was mid-generation caused a `KeyError` → HTTP 500. After: HTTP 422 with a human-readable message.

---

### 5c. wizard_id backlink + orphan cleanup

**The problem**: in `asyncio.gather`, `_generate_one` could crash after writing the plan to DB but before appending the plan ID to `wizard.draft_plan_ids`. That plan becomes an orphan: exists in DB, invisible to the wizard's cleanup logic.

**Two-layer fix**:

1. **Atomic append**: plan creation and `draft_plan_ids` append happen in the same transaction via `SELECT ... FOR UPDATE` on the wizard row.
2. **wizard_id FK backlink**: migration 008 adds `plans.wizard_id`. `_cancel_draft_plans` queries both the JSON list and the FK backlink:

```python
async def _cancel_draft_plans(db, wizard):
    ids_to_cancel = set(wizard.draft_plan_ids or [])
    orphans = await db.execute(
        select(Plan).where(Plan.wizard_id == wizard.id, Plan.status == PlanStatus.draft)
    )
    for orphan in orphans.scalars():
        ids_to_cancel.add(orphan.id)
    # cancel all
```

---

### 5d. Terminal guards — three-layer defense against cancel being overwritten

**The problem**: cancel writes `status=cancelled` to DB. But an in-flight `generate_plans_node → feasibility_node` sequence continues running. `run_feasibility_step` then unconditionally writes `status=feasibility_check`, overwriting the cancel.

**Three layers**:

```
Layer 1: node-level terminal check (research, generate_plans, feasibility)
         → if wizard.status in TERMINAL: skip work, return early

Layer 2: service-level db.refresh before writing status
         → run_feasibility_step: await db.refresh(wizard)
                                 if wizard.status in TERMINAL: return without writing

Layer 3: human_gate pre-check before interrupt()
         → if terminal: return human_decision="cancel" (routes to cancel_node → END)
            avoids leaving a dangling interrupt checkpoint
```

This is cooperative cancellation, not preemptive. In-flight LLM calls complete; their plans are cleaned up via the `wizard_id` backlink on cancel.

---

## 6. Results

### What improved (measured)

| Metric | Before | After |
|--------|--------|-------|
| Tests | 94 | **117** (all green) |
| Graph flow visibility | Implicit (read 3 functions) | Explicit (`build_wizard_graph()`, 43 lines) |
| Out-of-order call response | HTTP 500 (KeyError) | HTTP 422 (clear message) |
| Orphan draft plan cleanup | Partial (JSON list only) | Dual-path (list + FK backlink) |
| Cancel-during-generation | Unreliable (overwrite risk) | Three-layer terminal guard |
| Plan generation | Serial O(N) | Parallel O(1) per asyncio.gather |
| Crash recovery granularity | `collecting_constraints` boundary | Per-node checkpoint boundary |

### What improved (structural, not benchmarked)

- **Human gate**: protocol-enforced via `interrupt()`, not convention
- **Step ordering**: enforced by `assert_graph_awaiting`, not LLM prompt
- **Checkpointer swap**: change one constructor call to switch from SQLite to Postgres
- **Graph tests**: MemorySaver + mock service layer; no DB needed

### What didn't change (honest)

- **Observability**: LangSmith not integrated. Debug workflow is the same as before (cross-module logs). Node-level log prefix helps marginally.
- **LOC**: total wizard-related LOC increased from 2168 to 3138 (+45%). The increase is additive (new orchestration layer + guards + tests), not a simplification.
- **Extension cost**: still 4+ files for a new wizard step (graph + service + API + MCP). The improvement is that routing logic is now explicit and co-located, not that file count dropped dramatically.

---

## 7. Lessons (the parts with the most signal)

### Don't use LangGraph features just because they exist

`Send` API would have been the "LangGraph way" to do parallel plan generation. We used `asyncio.gather` instead because:
- Our Write-Through model (single wizard row with JSON list) makes fan-in via Send awkward
- `asyncio.gather` achieves the same concurrency with a fraction of the complexity

The right question is: **does the LangGraph primitive solve the actual problem better than the simpler alternative?** For fan-out with a shared aggregation target, the answer here was no.

### interrupt() and DB session lifetime must be coordinated explicitly

`interrupt()` is not a Python `await` — it suspends the entire coroutine via LangGraph's resumption mechanism. Any resource held in a `with` block at the call site (DB connections, file handles, HTTP sessions) stays open for the entire suspension period. This is a connection pool killer.

**Rule**: close all connections before calling `interrupt()`. The easiest pattern: do DB work in a separate `async with` block that closes before the `interrupt()` line.

### The migration target is control flow, not persistence

Every LangGraph tutorial shows you replacing your persistence layer with a Checkpointer. That's the right default for greenfield. In a system with an existing ORM model that's read by multiple paths (REST, monitoring, TTL crons), the migration surface is too large.

Write-Through (thin nodes, DB as source of truth) is the correct bridging pattern. It captures LangGraph's core value — explicit topology, interrupt protocol, node-level checkpoints — without requiring you to migrate your persistence model.

### Three-layer terminal guard is the right pattern for cooperative cancel

Cooperative cancel in a long-running async pipeline requires defense in depth. A single DB write isn't enough because:
1. Non-interrupt nodes don't receive cancel signals; they must poll
2. `await db.refresh(wizard)` is needed to see concurrent writes (SQLAlchemy caches the row)
3. The final interrupt node (human_gate) needs to check terminal state before calling `interrupt()` to avoid leaving dangling checkpoints

Each layer catches a different failure mode. All three are necessary.

---

## 8. Final Assessment

### What worked

Migrating the control flow (explicit topology, interrupt/resume, node-level checkpoints) was the right call for this problem. The wizard now has:
- A single authoritative picture of its state machine (graph topology in 43 lines)
- Protocol-enforced human gate that can't be bypassed by a misbehaving LLM caller
- Clear step ordering enforced at the API/MCP layer
- Recoverable checkpoints at each major step boundary

### What we'd do differently

- Start with `assert_graph_awaiting` from day one — out-of-order call safety is not optional
- Include the `wizard_id` FK backlink in the original plan schema — orphan cleanup becomes obvious
- Design terminal guards as part of node boilerplate from the start, not as a patch

### What's still open

- **LangSmith integration**: the structural hooks are there; needs `LANGCHAIN_API_KEY` and trace configuration
- **Checkpointer backend**: AsyncSqliteSaver works for single-process; a multi-process / multi-host deployment needs `AsyncPostgresSaver` or similar
- **TTL cleanup**: `expire_stale()` still runs as a cron; aligning it with Checkpointer thread lifecycle is a future cleanup

### The honest summary

This migration is complete in the sense that matters: the control flow is explicit, the human gate is protocol-enforced, crashes recover at node granularity, and 117 tests pass. It is not complete in the sense of "LangGraph's full value realized" — observability (LangSmith), production Checkpointer backend, and TTL alignment are all follow-up work.

That's the right scope for a Strangler Fig migration. Ship the control flow first; instrument and optimize in subsequent iterations.
