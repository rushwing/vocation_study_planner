# goal-agent × LangGraph 迁移专题 — 03 踩坑与经验总结

> **状态**：Post-migration（重构完成）
> **完成日期**：2026-03-09

---

## 概述

本次迁移历经 4 轮主要改动：

1. **Phase 1**：StateGraph 基础骨架（scope / targets / constraints / human_gate / confirm / adjust / cancel）
2. **Phase 2**：节点拆分（constraints → 4 节点）+ 并行计划生成 + AsyncSqliteSaver 持久化
3. **Phase 3**：API/MCP 守卫（assert_graph_awaiting）+ Draft plan wizard_id backlink
4. **Phase 4**：终止状态守卫（terminal guard in non-interrupt nodes）+ human_gate pre-check

每轮均有具体发现（P1/P2 级别），以下按问题维度整理。

---

## L1 — AsyncSqliteSaver 必须作为上下文管理器在 lifespan 内打开

### 问题

`AsyncSqliteSaver.from_conn_string()` 返回的是 async context manager，不是直接可用对象。
如果在 `get_wizard_graph()` 里懒初始化（模块级 singleton），
sqlite 连接会在第一次 graph invoke 前无法打开，或进程退出前不关闭。

### 解法

在 FastAPI `lifespan()` 里用 `async with` 打开，持续到应用关闭：

```python
# app/main.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(_DEV_CHECKPOINTER_PATH) as checkpointer:
        set_wizard_graph(build_wizard_graph(checkpointer))
        logger.info("Wizard graph: AsyncSqliteSaver at %s", _DEV_CHECKPOINTER_PATH)
        # ... bot_task, scheduler ...
        yield
        # cleanup inside with block
```

`build_wizard_graph(checkpointer)` + `set_wizard_graph()` 在 lifespan 启动时注入。

### 规律

> **任何需要持续持有资源的 LangGraph checkpointer（Sqlite / Postgres / Redis）都必须在应用生命周期管理器里打开，不能懒初始化。**

---

## L2 — Write-Through + Thin Nodes：保留 DB 为 source of truth

### 问题

LangGraph State 自带 checkpoint，诱惑是把所有业务数据（target_specs、constraints、reference_materials…）也存入 graph state，让 Checkpointer 管理全部。

但 `GoalGroupWizard` 表已有 REST/MCP 读取路径、TTL 管理、监控查询。
全部迁入 Checkpointer 需要改所有读取路径。

### 解法

**方案 B（Write-Through + Thin Nodes）**：

- Graph state 只存控制流字段（`wizard_id`、`status`、`human_decision`、`adjust_patch`、`confirm_result`、`error`）
- 每个 node 打开自己的 `AsyncSessionLocal` session，从 DB 读取 wizard，调用 service 函数，commit，返回 state delta
- `GoalGroupWizard` 表保持 source of truth，REST GET 路径不变

```python
async def research_node(state: WizardState) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
        await wizard_service.run_web_research_step(db, wizard)
        await db.commit()
    return {**state, "error": ""}
```

### 代价

- 每个节点多一次 DB 读（不用 state 传业务数据）
- Checkpointer 崩溃恢复：控制流恢复到最后 checkpoint node；DB 状态由各 node 自己写，独立于 Checkpointer

### 规律

> **在已有 ORM 模型的项目里引入 LangGraph，优先用 Write-Through 模式而不是全量迁入 Checkpointer。
> 新增 LangGraph 的价值在于控制流 checkpoint 和 interrupt 协议，不在于替代持久化层。**

---

## L3 — interrupt() 前必须关闭 DB 连接

### 问题

`interrupt()` 把整个 async 协程挂起。如果 interrupt 前持有 `AsyncSession`，连接池连接被占用直到 resume（可能是分钟级），导致连接池耗尽。

### 解法

在调用 `interrupt()` 之前关闭所有 session：

```python
async def human_gate_node(state: WizardState) -> dict[str, Any]:
    # Load wizard BEFORE interrupt — session closed when 'async with' exits
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
    # Now check terminal state with closed session
    if wizard is None or wizard.status in _TERMINAL_STATUSES:
        return {**state, "human_decision": "cancel"}
    # Safe to interrupt: no DB connection held
    data = interrupt({"awaiting": "human_decision"})
    ...
```

对于 scope / targets / save_constraints 这三个节点，由于 interrupt 发生在 DB 操作之前（resume 后才读 DB），不存在此问题。

### 规律

> **在 interrupt() 调用前，确保所有 async context manager（DB session、HTTP client 等）已退出作用域。**

---

## L4 — 并行计划生成：用 asyncio.gather + 独立 session，而非 LangGraph Send API

### 问题

原计划用 LangGraph `Send` API 实现 fan-out（每个 target 一个子图节点）。
实际问题：

1. `Send` 要求每个 target 有独立的子图节点，fan-in 合并逻辑（收集所有 plan_id 写回 wizard）需要额外的聚合节点
2. 子图节点的 state 是独立的，合并回主 state 需要 reducer 函数
3. Write-Through 架构下，每个 target 的计划写入 DB 后还要 append 到 `wizard.draft_plan_ids`，Send 并行时出现写竞争

### 解法

在 `generate_plans_node`（单一节点）内用 `asyncio.gather` 并行：

```python
async def generate_plans_parallel(db, wizard):
    async def _generate_one(spec):
        async with AsyncSessionLocal() as gen_db:
            new_plan = await plan_generator.generate_plan(..., wizard_id=wizard.id)
            # Atomic append to wizard.draft_plan_ids
            w = await gen_db.execute(
                select(GoalGroupWizard).where(...).with_for_update()
            )
            w.draft_plan_ids = list(w.draft_plan_ids or []) + [new_plan.id]
            gen_db.add(w)
            await gen_db.commit()
            return new_plan.id, None

    results = await asyncio.gather(
        *[_generate_one(spec) for spec in wizard.target_specs], return_exceptions=True
    )
    ...
```

**关键**：每个 `_generate_one` 使用独立的 `AsyncSessionLocal`（不共享父 session），`SELECT ... FOR UPDATE` 保证 draft_plan_ids append 的原子性。

### 规律

> **LangGraph `Send` API 适合真正的子图 fan-out（每个分支有独立的多步骤逻辑）。
> 对于"并行调 LLM、结果汇总写一个 DB 行"这类场景，asyncio.gather + 独立 session 更简单。
> 不要为了用 LangGraph 特性而用。**

---

## L5 — 乱序调用守卫：assert_graph_awaiting 防 500

### 问题

REST/MCP 层直接 `graph.ainvoke(Command(resume=...))` 时，如果 graph 当前不在对应 interrupt 节点（例如 scope 被调了两次，或 targets 在 generate_plans 运行时被调），LangGraph 会抛出内部错误，HTTP 层返回 500。

### 解法

在每个 step endpoint 和 MCP tool 里，`graph.ainvoke` 之前先调 `assert_graph_awaiting`：

```python
async def assert_graph_awaiting(graph, wizard_id: int, expected_node: str) -> None:
    config = {"configurable": {"thread_id": str(wizard_id)}}
    snapshot = await graph.aget_state(config)
    if not snapshot or not snapshot.next:
        raise ValueError("Wizard graph thread is not active...")
    actual = snapshot.next[0]
    if actual not in INTERRUPT_NODES:
        raise ValueError(f"Wizard is currently processing (step: '{actual}'). Wait...")
    if actual != expected_node:
        raise ValueError(f"Out-of-order call: waiting at '{actual}', but called '{expected_node}'")
```

调用方捕获 `ValueError` → 返回 422：

```python
try:
    await assert_graph_awaiting(graph, wizard_id, "scope")
    await graph.ainvoke(Command(resume={...}), config=...)
except ValueError as e:
    raise HTTPException(422, str(e)) from e
```

**友好消息示例**：
- `"Out-of-order call: wizard is waiting at 'POST /targets', but you called 'POST /scope'. Call steps in the correct order."`
- `"Wizard 42 is currently processing (step: 'research'). Wait for generation to complete before retrying."`

### 规律

> **在 interrupt-based graph 里，REST/MCP 层必须在每个 resume 调用前验证 `snapshot.next`。
> 将这个验证封装成工具函数，统一错误消息格式，避免每个 endpoint 重复实现。**

---

## L6 — Draft Plan 孤儿问题：双保险追踪（wizard_id FK + draft_plan_ids JSON）

### 问题

`generate_plans_parallel` 中，`_generate_one` 可能在 plan 写入 DB 之后、append 到 `wizard.draft_plan_ids` 之前崩溃。该 plan 成为孤儿：DB 里存在，但 wizard 不知道它。

cancel 时只遍历 `wizard.draft_plan_ids`，无法清理孤儿。

### 解法

两层保险：

1. **atomic append**：`_generate_one` 在同一 session 里用 `SELECT ... FOR UPDATE` 锁住 wizard，plan 写入和 draft_plan_ids append 在同一事务里 commit。事务要么全成功要么全回滚，消灭孤儿窗口。

2. **wizard_id FK backlink**：新增 `plans.wizard_id` 列（migration 008），`plan_generator.generate_plan` 传 `wizard_id=wizard.id`。`_cancel_draft_plans` 双路径：

```python
async def _cancel_draft_plans(db, wizard):
    ids_to_cancel = set(wizard.draft_plan_ids or [])
    # Catch any orphans via wizard_id backlink
    orphan_result = await db.execute(
        select(Plan).where(Plan.wizard_id == wizard.id, Plan.status == PlanStatus.draft)
    )
    for orphan in orphan_result.scalars().all():
        ids_to_cancel.add(orphan.id)
    # cancel all
    ...
```

### 规律

> **并行写 DB 后聚合结果时，不要依赖"最终一致"。把聚合（append 到列表）和生成（写 plan）放在同一事务里。
> 同时建立 backlink 索引作为第二条追踪路径，防御单点失效。**

---

## L7 — 中途取消不能被覆盖：terminal guard in non-interrupt nodes

### 问题

`generate_plans_node` → `feasibility_node` 是 non-interrupt 节点，进入时 wizard 可能已被取消。
`run_feasibility_step()` 无条件写 `status=feasibility_check`，覆盖 DB 里的 `cancelled` 状态。

取消请求写了 DB（cooperative cancel），但图仍在运行并在 feasibility 节点覆盖了取消状态。

### 解法

三层防御：

1. **Node 级 terminal guard**：每个 non-interrupt 节点开头加：
   ```python
   if wizard is None or wizard.status in _TERMINAL_STATUSES:
       logger.info("research_node: wizard %d is terminal, skipping", ...)
       return {**state, "error": ""}
   ```

2. **Service 级 db.refresh**：`run_feasibility_step` 在写状态前 refresh，重新读取 DB 状态：
   ```python
   await db.refresh(wizard)
   if wizard.status in _TERMINAL:
       return wizard  # do not overwrite
   wizard = await crud_wizard.update_wizard(db, wizard, status=WizardStatus.feasibility_check, ...)
   ```

3. **human_gate pre-check**：`human_gate_node` 在调 `interrupt()` 前读 wizard 状态；如果 terminal，直接返回 `human_decision="cancel"`，跳过 interrupt，graph 路由到 `cancel_node → END`。

```python
async def human_gate_node(state: WizardState) -> dict[str, Any]:
    async with AsyncSessionLocal() as db:
        wizard = await crud_wizard.get(db, state["wizard_id"])
    # Session closed before interrupt()
    if wizard is None or wizard.status in _TERMINAL_STATUSES:
        return {**state, "human_decision": "cancel"}  # route to cancel_node, skip interrupt
    data = interrupt({"awaiting": "human_decision"})
    ...
```

如果没有 human_gate pre-check，图会在 interrupt 点挂起（snapshot.next = ["human_gate"]），需要外部 resume 才能继续到 cancel_node。对于已经 DB 取消的 wizard，这是一个悬挂的 interrupt checkpoint，需要额外的清理逻辑。

### 规律

> **Cooperative cancel（写 DB 标记 → 期待节点感知并停止）在 LangGraph 里需要每个节点主动检查。
> 特别是 non-interrupt 节点没有"暂停等待"，必须在节点开头主动守卫。
> human_gate 要在 interrupt() 之前检查 terminal state，避免悬挂 checkpoint。**

---

## L8 — 测试模式：MemorySaver + mock service layer

### 推荐模式

```python
@pytest.fixture
def _make_mocks(monkeypatch):
    mocks = {
        "set_scope": AsyncMock(),
        "set_targets": AsyncMock(),
        "save_constraints_to_db": AsyncMock(),
        "run_web_research_step": AsyncMock(),
        "generate_plans_parallel": AsyncMock(return_value=([], [])),
        "save_plan_gen_results": AsyncMock(),
        "run_feasibility_step": AsyncMock(),
        "save_adjust_patch": AsyncMock(),
        "confirm": AsyncMock(return_value=(mock_group, [])),
        "cancel_wizard": AsyncMock(),
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(f"app.services.wizard_service.{name}", mock)
    return mocks

@pytest.fixture
def graph():
    return build_wizard_graph(MemorySaver())
```

### Terminal guard 测试

用 `side_effect` 计数器控制 `crud_wizard.get` 返回值：

```python
call_count = 0
def _get_wizard_side_effect(*args, **kwargs):
    nonlocal call_count
    call_count += 1
    if call_count <= 3:          # scope / targets / save_constraints
        return normal_wizard
    return cancelled_wizard      # 所有后续 node 读到 terminal

monkeypatch.setattr("app.services.wizard_graph.crud_wizard.get",
                    AsyncMock(side_effect=_get_wizard_side_effect))
```

### 规律

> **将 graph 测试和 service 测试彻底分离：graph 测试 mock 整个 service layer（验证调用顺序 + 路由），service 测试 mock LLM（验证业务逻辑）。
> 用 MemorySaver 替代 AsyncSqliteSaver，无需真实 DB 即可测试完整图流程。**

---

## 总结

| 类别 | 踩坑 | 解法关键词 |
|------|------|-----------|
| Checkpointer 生命周期 | 懒初始化导致连接未打开 | lifespan async with |
| DB session 与 interrupt | interrupt 持有连接耗尽连接池 | interrupt 前关闭 session |
| Write-Through vs 全量迁入 | 全量迁入改动面太大 | 保留 ORM source of truth，graph 只存控制流 |
| 并行计划生成 | Send API fan-in 复杂 + 写竞争 | asyncio.gather + 独立 session + SELECT FOR UPDATE |
| 乱序调用 | ainvoke 抛 500 | assert_graph_awaiting + 422 |
| 孤儿 draft plan | crash between plan write and id append | atomic append in same tx + wizard_id FK backlink |
| Cooperative cancel 覆盖 | non-interrupt 节点不感知 cancel | terminal guard 三层（node / service / human_gate pre-check） |
| Graph 测试 | 依赖 DB fixture | MemorySaver + mock service + side_effect 计数器 |
