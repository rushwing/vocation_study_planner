# goal-agent × LangGraph 迁移专题 — 02 Before vs After 对比

> **状态**：Post-migration（控制流迁移完成；LangSmith / Tracing 等 observability 增强为后续项）
> **完成日期**：2026-03-09
> **对比基线**：`main` @ 7d87f88（PR #55，重构前）
> **对比快照**：本次迁移 PRs（wizard_graph + node splitting + parallel plan gen + cancel semantics）
>
> **数据说明**：本文指标分两类——
> - 🔬 **实测**：通过 `wc -l`、`pytest` 计数、代码静态分析等直接测量得到
> - 📐 **估算/工程判断**：基于实现原理和结构分析给出的合理推断，未做严格 benchmark

---

## 1. 量化指标总览

### 1.1 代码量（LOC）

🔬 数据来源：`wc -l`（2026-03-09 快照）

| 文件 | Before | After | 说明 |
|------|--------|-------|------|
| `app/services/wizard_service.py` | 517 | 773 | +6 个 graph-node service 函数（旧函数全部保留） |
| `app/api/v1/wizards.py` | 288 | 388 | 增加 `assert_graph_awaiting` 守卫 + cancel 路由 |
| `app/mcp/tools/wizard_tools.py` | 444 | 554 | 同上（MCP 层同步更新） |
| `app/services/wizard_graph.py` | — | 413 | **新文件**：StateGraph + 10 节点 + 路由函数 + guards |
| `app/services/wizard_checkpointer.py` | — | 25 | **新文件**：AsyncSqliteSaver 路径常量 |
| `app/services/plan_generator.py` | 228 | 230 | 增加可选参数 `wizard_id` |
| `app/models/plan.py` | ~63 | 64 | 增加 `wizard_id` FK 列 |
| 其余 7 个文件 | 891 | 891 | 未修改 |
| **合计** | **2168** | **3138** | +970 LOC（约 +45%）；增量来自新增图层 |

> **说明**：LOC 增加主要来自两个方向：(a) `wizard_graph.py` 是全新的编排层；
> (b) `wizard_service.py` 追加了 6 个专供 graph node 调用的细粒度函数（不删旧函数，Strangler Fig 模式）。
> 旧的 HTTP/MCP 层因为增加了 `assert_graph_awaiting` 守卫和取消路由逻辑而略有增长。
> LOC 增加不等于复杂度增加——原有控制流逻辑仍在，新增部分是之前缺失的守卫和显式拓扑。

### 1.2 测试

| 指标 | Before | After |
|------|--------|-------|
| 总测试数 | 94 | **117** |
| Wizard 测试 LOC | 922 (519 + 403) | **1752** (519 + 403 + 830) |
| 新增测试文件 | — | `tests/unit/test_wizard_graph.py` (830 LOC) |
| 测试通过率 | 100% | **100%** |

### 1.3 复杂度

| 指标 | Before | After |
|------|--------|-------|
| 手写状态跳转次数（`status=WizardStatus.*`） | 10 | 13（wizard_service） + **0**（wizard_graph，状态由 graph 控制） |
| wizard_service.py 内 if/elif 分支数 | 43 | 62（含新增函数） |
| **wizard_graph.py** 内状态路由分支数 | — | **7**（4 个 `_cancel_or` 调用 + 3 个 route 函数）|
| graph 节点数 | — | 10（scope/targets/save_constraints/research/generate_plans/feasibility/human_gate/confirm/adjust/cancel） |
| interrupt 节点数 | — | 4（scope/targets/save_constraints/human_gate） |

---

## 2. 七维对比（D1–D7）

### D1 — 代码可读性（State Flow Clarity）

| 维度 | Before | After |
|------|--------|-------|
| 状态路由方式 | 手写 if/elif，散落在 `_generate_and_check()` / `adjust()` / 多个 guard clause | `StateGraph` 节点 + 条件边，拓扑一图可读 |
| 状态图可视化 | 无（只能读代码） | LangGraph Studio 自动渲染；graph 拓扑在 `wizard_graph.py` 末尾 `build_wizard_graph()` 集中定义 |
| 新增步骤需改文件数 | 5–7（model + migration + service + api + mcp + schema ± plugin.json） | **4**（graph node + service fn + api route + mcp tool；拓扑变更集中在 wizard_graph.py） |
| 控制流可见性 | 隐式（散落在 Python 控制流） | 显式（`add_node` / `add_edge` / `add_conditional_edges`） |

🔬 **实测**：`wizard_graph.py` 的 `build_wizard_graph()` 函数 43 行，包含完整图拓扑。
🔬 **实测**：重构前，等效控制流分散在 `set_constraints()` / `_generate_and_check()` / `adjust()` 共约 150 行、3 个函数。
📐 **估算**：新增步骤改文件数（4 vs 5–7）是工程判断，未严格验证；实际取决于步骤是否需要新 schema 字段或 migration。

---

### D2 — 健壮性（Crash Recovery）

| 维度 | Before | After |
|------|--------|-------|
| Checkpoint 粒度 | WizardStatus（DB 行级别，仅 collecting_scope / collecting_targets / collecting_constraints / generating_plans / feasibility_check）| 每个 graph node 结束后自动 checkpoint（AsyncSqliteSaver） |
| 崩溃恢复点 | `set_constraints` 触发的 `_generate_and_check()` 是整体，中途崩溃回到 `collecting_constraints`（web research + 所有 plan gen 重来） | 崩溃后从最后 checkpoint 恢复；`save_constraints_node` 已过 → `research_node` 不重跑；`research_node` 已过 → `generate_plans_node` 不重跑 |
| Draft plan 孤儿问题 | 并行 gen 中途崩溃 → 已写 DB 的 plan 未进 `draft_plan_ids`，成为孤儿 | `_generate_one` 内 atomic append（`SELECT ... FOR UPDATE`）；`plans.wizard_id` FK backlink 双保险，cancel 时两条路径均清查 |
| 中途取消覆盖问题 | `run_feasibility_step` 无守卫，取消后仍写 `status=feasibility_check` | 每个 non-interrupt node 先查 `wizard.status in TERMINAL_STATUSES`；`run_feasibility_step` 做 `await db.refresh(wizard)` 再查 terminal |
| TTL 管理 | 手工 `expires_at` + cron `expire_stale()` | 保留（GoalGroupWizard 仍为 source of truth）；AsyncSqliteSaver thread 与 wizard 同生命周期 |

**新增迁移**：`alembic/versions/008_plan_wizard_id.py` — `plans.wizard_id` nullable FK，带 `ON DELETE SET NULL` 和索引 `ix_plans_wizard_id`。

---

### D3 — 并发性能（Throughput）

| 维度 | Before | After | 数据来源 |
|------|--------|-------|---------|
| Web Research | `asyncio.gather`（并发） | `asyncio.gather`（保持不变） | 🔬 代码分析 |
| 计划生成 | **串行 for loop**（N targets × ~30s/plan） | **`asyncio.gather` 并行**（瓶颈 ≈ 单个 plan 耗时） | 🔬 代码分析 |
| 理论耗时（3 targets） | ≈ 3× 单 plan 耗时 | ≈ 1× 单 plan 耗时 | 📐 理论推算（未做 mock benchmark）；实际受 LLM API 限流影响 |
| 实现机制 | — | `generate_plans_parallel()`：每个 target 独立 `AsyncSessionLocal` + `await gen_db.commit()`；`asyncio.gather` 并发 | 🔬 代码实现 |

> **注意**：原计划使用 LangGraph `Send` API 做 fan-out，实际采用 `asyncio.gather` + 独立 session 方案。
> 原因：`Send` API 要求每个 target 独立的子图节点，且 fan-in 合并逻辑复杂；
> `asyncio.gather` 方案在现有 Write-Through 架构下更简单且同等并发。
> 详见 [03-lessons-learned.md § L4](./03-lessons-learned.md)。

---

### D4 — Human-in-the-loop 可靠性

| 维度 | Before | After |
|------|--------|-------|
| 暂停机制 | 约定（文档 + LLM prompt），无协议保证 | `interrupt()` 协议级暂停（scope / targets / save_constraints / human_gate） |
| 绕过风险 | OpenClaw 跳过 confirm 调其他 tool，无保护 | graph 在 interrupt 状态，任何 `ainvoke` 都返回 interrupt state 而不前进 |
| 顺序保证 | 无（endpoint 独立，调用顺序只靠 LLM prompt 约定） | `assert_graph_awaiting(graph, wizard_id, expected_node)` 在每个 step 前验证；乱序调用返回 422 |
| 取消时机 | 任意时刻调 cancel endpoint，行为未定义 | interrupt 节点：图路由到 `cancel_node`；非 interrupt 节点（生成中）：直接 DB cancel，节点内 terminal 守卫防止状态覆盖 |
| 恢复方式 | 调 `/confirm` endpoint | `graph.ainvoke(Command(resume={"decision": "confirm"}), ...)` |

🔬 **定性评估**（结构性，非 benchmark）：
- Before：可被跳过（OpenClaw 可直接调 confirm 而跳过 feasibility 展示）
- After：无法跳过 interrupt 节点（graph 必须经过 feasibility → human_gate，interrupt 后等待 resume）

---

### D5 — 可观测性（Observability）

| 维度 | Before | After | 数据来源 |
|------|--------|-------|---------|
| 状态快照 | 查 `GoalGroupWizard.status` DB 列 | `graph.aget_state(config).next` 精确到 node 名；DB 状态同步保留 | 🔬 代码实现 |
| Trace 粒度 | Python logging，无 run-level trace ID | 结构上支持 LangSmith（`LANGCHAIN_TRACING_V2=true`）；**本项目暂未接入，无实测数据** | ⚠️ 未实现 |
| Debug "计划生成失败" | 需跨 3 个模块查日志 | **同前**（LangSmith 未接入）；每个 node 是独立函数，日志带节点名，但查多模块的问题未根本改善 | 🔬 现状如实 |
| 乱序调用诊断 | 500 KeyError（无法定位）| 422，含明确消息（`assert_graph_awaiting` 返回）| 🔬 实测（test_wizard_graph.py 覆盖） |

> **D5 小结**：本次迁移在 Observability 上有结构性改善（node 级快照、乱序调用诊断），但 LangSmith trace 接入是后续项。
> 在 LangSmith 未接入前，"一个 trace 看全流程"的目标尚未实现。

---

### D6 — 测试性（Testability）

| 维度 | Before | After |
|------|--------|-------|
| 测试总数 | 94 | **117**（+23 条） |
| Graph 逻辑测试 | 无（状态机隐含在 service 函数里） | `test_wizard_graph.py`：11 类场景，830 LOC |
| State 测试方式 | 依赖 SQLite fixture + ORM 对象 | Graph state 是 Python dict；mock wizard_service → 纯内存图测试 |
| Mock 方式 | `patch("app.services.wizard_service.*")` | 同前；`_make_mocks()` 统一 patch 所有 service 函数，MemorySaver 做 checkpointer |
| 新增测试场景 | — | terminal 守卫测试、`assert_graph_awaiting` 测试、cancel 路由测试、mid-gen cancel 集成测试 |
| 能否不依赖 SQLite 测图逻辑 | 不能 | **能**（MemorySaver + mock service，无 DB 依赖） |

---

### D7 — 扩展成本（Extension Cost）

| 操作 | Before | After | 数据来源 |
|------|--------|-------|---------|
| 新增 wizard 步骤 | 5–7 个文件 | 📐 估算 **≥ 4 个文件**（graph.py + service.py + api.py + mcp.py）；若需新 DB 字段则仍需 model + migration | 📐 工程判断，未验证 |
| 修改状态路由逻辑 | 读懂分散在 3 个函数的隐式控制流 | 改 graph 边；`add_conditional_edges` 参数直接表达路由意图 | 🔬 代码结构 |
| 替换 LLM 模型 | 修改 `llm_service.py` | 同前（LLM 封装层未改） | 🔬 代码结构 |
| 接入新 Checkpointer 后端 | 不支持（手写 DB flush） | 换 checkpointer 实现（如 `AsyncPostgresSaver`），`build_wizard_graph(checkpointer)` 传参 | 📐 LangGraph API 设计保证 |
| 增加取消守卫到新节点 | 手写 if 判断 + 手动 status 更新 | 在节点开头加 `if wizard.status in _TERMINAL_STATUSES: return {**state}` 两行 | 🔬 现有节点模式 |

---

## 3. 架构对比图

### Before（重构前控制流）

```
wizard_service.py
├── set_scope()
│     └── crud_wizard.update_wizard(status=collecting_targets)
├── set_targets()
│     └── crud_wizard.update_wizard(status=collecting_constraints)
├── set_constraints()
│     └── _generate_and_check()   ← 串行：research → for-loop plan gen → feasibility
│           └── crud_wizard.update_wizard(status=feasibility_check)
└── adjust()
      └── _generate_and_check()   ← 复用，但状态路由手工再来一遍
```

### After（重构后图拓扑）

```
wizard_graph.py — build_wizard_graph()

START
  → scope_node          [interrupt: awaiting scope]
  → targets_node        [interrupt: awaiting targets]
  → save_constraints_node [interrupt: awaiting constraints]
  → research_node       [terminal guard → skip]
  → generate_plans_node [terminal guard → skip; asyncio.gather 并行]
  → feasibility_node    [terminal guard + db.refresh guard → skip]
  → human_gate_node     [terminal pre-check → skip interrupt if cancelled]
       ├─ confirm  → confirm_node  → END
       ├─ adjust   → adjust_node  → research_node (loop)
       └─ cancel   → cancel_node  → END

任意 interrupt 节点收到 {"action": "cancel"} → cancel_node
```

---

## 4. 风险项兑现情况

| 风险 | 预期（01-migration-plan.md §4） | 实际 |
|------|------|------|
| R1 AsyncSession 生命周期 | 需与 graph 对齐 | **已解决**：Write-Through 模式；每个 node 开自己的 `AsyncSessionLocal` session，interrupt 前关闭 |
| R2 Checkpointer vs DB 模型 | 方案 A vs B | **选方案 B**：GoalGroupWizard 保留为 source of truth；Checkpointer 仅做控制流 checkpoint |
| R3 MCP Tool 并发安全 | thread_id 多用户隔离 | **已解决**：`thread_id = str(wizard_id)`，每个 wizard 独立 graph thread |
| R4 OpenClaw 协议对齐 | interrupt 需 OpenClaw 理解 | **已解决**：REST/MCP tool 层封装了 `Command(resume=...)` 调用；OpenClaw tool interface 不变 |
| R5 测试框架对齐 | MemorySaver + 新 fixture | **已解决**：`_make_mocks()` 统一 mock service 层；117 条测试全绿 |
