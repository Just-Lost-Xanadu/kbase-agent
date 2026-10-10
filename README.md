# kbase-agent 仓库

本仓库是一个**LangGraph 编排层**工程：企业知识库 + 工具调用 Agent（RAG + 真 MCP stdio 接入 + 两层评测 Harness + SQLite 会话持久化 + 节点级 trace/成本可观测）。

> **项目实际代码在子目录 [`kbase-agent/`](./kbase-agent/)**，完整的项目说明、架构、API、评测口径与已知取舍都在
> **[`kbase-agent/README.md`](./kbase-agent/README.md)**。本文件只作为仓库入口。

## 这个项目做什么

制度类问题（员工手册 / 产品 FAQ）走知识库检索，个人状态类问题（年假剩余 / 报销进度）再查业务数据，
两类工具结合回答并带【来源：文件名】引用。

- **Agent 编排**：手写 LangGraph `StateGraph`（ReAct 式工具循环）——两条边 + 条件路由（按末条消息的
  `tool_calls` 决定 `agent → tools`），不是预制 Agent 函数。
- **MCP 真接入**：工具由 FastMCP 声明，经 `langchain-mcp-adapters` 以 **stdio 子进程**接入，不是装饰性的 function calling；
  **一个 runtime 常驻一个 stdio 会话**（不是每次调用新起子进程）——实测单次 `retrieve_knowledge` **3782ms → 12ms**，
  会话被 OOM/杀掉时按传输层异常自愈重启，工具业务报错不触发重建。
- **RAG**：解析（md/txt/docx/xlsx/pdf）→ 分块 → 向量 + BM25 走 **RRF** 融合 → 可选 rerank → 引用溯源。
  语料 **39 篇 / 40 分块**，检索层有真实区分度：仅向量 0.875、仅 BM25 0.964、**RRF 0.982**。
  （规模边界一并写明：全库约 1.6 万字、最长一块 795 字符 < `chunk_size=800`，
  所以区分度来自刻意构造的近义干扰文档、**不来自规模**；据此不宣称"做了分块调优"。详见主 README。）
- **工程化**：FastAPI 同步 + **SSE 节点级流式**；会话 checkpoint 落 SQLite（WAL），重启可续聊；
  `/api/runs` 提供节点级 trace（含每步实际调用的工具与参数）与 token/成本估算；护栏含单步超时、输出截断、单次运行内重复调用判重。
- **评测闭环**：自建 **60 条金标两层 Harness**——检索层（离线零成本）+ 端到端回归层
  （回答率 / 检索口径引用覆盖 / **答案引用口径** / 答案关键词覆盖率 / **应拒答通过率** / 成本与 p50·p95），
  金标含**单源 / 多跳 / 冲突 / 应拒答**四类，支持 `--tag` 落报告、`--compare` 基线 diff、
  `--reanalyze` 离线重算指标。实测报告在 [`kbase-agent/docs/eval-reports/`](./kbase-agent/docs/eval-reports/)。
- **参数消融可一条命令复现**：检索路（`--route`）、`top_k`（`--sweep-k`）、切分粒度
  （`--sweep-chunk-size`，会重建索引并在跑完后自动恢复默认值）都有开关。切分粒度那张表给出了一个
  具体结论：**块越大，向量路单调变差（0.9643→0.8214），而 BM25 几乎不动**——两条路的失效模式不同，
  这正是值得做混合检索的又一条证据；RRF 在 400~800 有平台，到 1600 才跟着掉。
- **可归因的性能改造（两步，各自形状不同）**：第一步把同一批的多个工具调用从串行改成
  `asyncio.gather` 并发，实测两个工具 8127ms → 4663ms，端到端 `p95` 11049ms → **7256ms（−34%）**
  而 `p50` 不变——收益只落在"一次调多个工具"的尾部用例上，这个形状本身就是改动打对了地方的证据。
  第二步把 MCP 从"每次调用新起 stdio 子进程"改成**常驻会话**（`session()` + `load_mcp_tools`），
  实测单次工具调用 **3782ms → 12ms**，端到端 `p50` 由约 6.6s 降到约 2s；此时延迟大头已变成
  LLM 往返而不是工具调用。两步都**没有**放弃"真 MCP"这个属性。
- **不宣称测不出来的东西**：README 记录了一次**真实的扩容实验**——9 篇语料时三条检索路都是 1.0
  （评测没有区分度，"用了混合检索"没有数据支撑）；扩到 39 篇并把金标加到 60 条后差异才出现。
  也写明了**规模边界**：全库 1.5 万字，`fixed` vs `recursive` 的差别只作用在 3/39 篇文档上，
  统计效力不足以支撑"做了分块调优"，所以不宣称。另外 `p50/p95`、关键词覆盖率、应拒答通过率
  都明确标注了自己的口径边界与已知盲区。

## 快速开始

```bash
cd kbase-agent
python -m venv .venv
.venv\Scripts\activate                     # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -r requirements.lock           # 复现已验证的依赖组合（Python 3.12）
pip install -e ".[dev]"
copy .env.example .env                     # 填入 DEEPSEEK_API_KEY
uvicorn app.main:app --reload              # http://127.0.0.1:8000
```

Windows 上也可直接双击 `kbase-agent/start.bat`（一键建 venv → 装依赖 → 建索引 → 起服务）。

不需要 API key 的离线路径：`python scripts/index_docs.py` → `python scripts/eval.py` →
`python scripts/eval_e2e.py --reanalyze --tag corpus-v2`。
「检索路消融」的表格现在可以一条命令复现：`python scripts/eval.py --route vector|bm25`、
`python scripts/eval.py --sweep-k`、`python scripts/eval.py --sweep-chunk-size`。

## 与姊妹项目 mcp-tools 的关系

两仓库是**一个系统的两层**，不是两个重复 demo：本仓库是**编排层**（Agent 怎么决策、怎么检索、
怎么管状态、怎么评测），[`mcp-tools`](https://github.com/Just-Lost-Xanadu/mcp-tools) 是**协议层**
（把工具按 MCP 标准做成可被任何客户端消费的 Server，安全边界在其内部）。

## 许可证

[MIT](./LICENSE) © 2026 Just-Lost-Xanadu
