# kbase-agent

**企业知识库 + 工具调用 Agent（单 Agent）**：基于 LangGraph 的 ReAct 式工具循环，RAG 检索与业务工具经 **MCP 协议真接入**（stdio）；配**两层评测 Harness**（40 条金标：检索层 + 端到端回归）、SQLite 会话持久化、trace/成本可观测与护栏。求职学习项目的旗舰作品。

## 目录结构

```
app/
  config.py            # 环境配置（DeepSeek 主，OpenAI 兼容层模型无关）
  main.py              # FastAPI 入口（懒加载资源）
  services.py          # 检索管道 + Agent runtime 懒加载与进程内缓存
  store.py             # 会话/消息记录 + run_traces 观测表（读历史/观测接口数据源）
  observability.py     # 轻量 trace：节点耗时/token/成本（contextvars，不侵入 LangGraph state）
  api/
    chat.py            # 请求/响应模型(内联) + POST /chat（同步）+ POST /chat/stream（SSE 节点级流式）
  retrieval/
    loader.py          # 文档解析 md/txt/docx/xlsx/pdf（文本层抽取，_ 前缀忽略）
    chunker.py         # fixed vs recursive 切分
    embedder.py        # 默认 FastEmbed(ONNX 免 torch)，可切 bge-m3
    vector_store.py    # Chroma（开发；生产可换 Milvus/ES）
    keyword.py         # BM25 关键词索引（纯 Python）
    hybrid.py          # 向量 + BM25 走 RRF 合并 + 可选 rerank
    pipeline.py        # 装配入口：index / retrieve / 懒加载
  agent/
    graph.py           # AgentState(内联) + LangGraph StateGraph + LLM 工厂(make_llm) + SqliteSaver(checkpoint) + 护栏 + MCP 接线
  mcp/
    servers.py         # FastMCP：retrieve_knowledge / query_business_db（HR 个人数据）
  guardrails.py        # 轮次/超时/工具输出截断/重复调用检测
  data/
    docs/                # 知识库语料：6 篇 md + docx/xlsx/pdf 样例，统一分块入库（_ 前缀忽略）
    business/            # 业务系统个人数据（首次调用自动生成示例）
    checkpoints.sqlite   # 会话 checkpoint + 消息记录（自动生成；.gitignore 已忽略 *.db）
eval/                  # 40 条评测集（检索层 eval.py / 端到端 eval_e2e.py）
scripts/               # index_docs.py / eval.py / eval_e2e.py / demo_agent.py
static/                # 单文件演示前端（index.html，无构建，打开即聊）
tests/                 # smoke + 纯逻辑单测
```

## 快速开始

**一键启动（推荐）**：进入 `kbase-agent` 目录，双击 `start.bat`（Windows），或运行 `./start.ps1`。
脚本会自动完成：检查/创建 `.venv` → 缺依赖时安装 → 生成 `.env`（仍占位则提示你填 key）→ 缺索引时自动建库 → 启动服务并打开 http://127.0.0.1:8000。`Ctrl+C` 停止。

手动步骤（等价，想看清每步时用）：

```bash
cd kbase-agent
python -m venv .venv
.venv\Scripts\activate          # Windows；macOS/Linux 用 source .venv/bin/activate
pip install -e ".[dev]"         # 默认零 torch；重排/bge-m3 才需要 .[embed]
copy .env.example .env          # 填入 DEEPSEEK_API_KEY
uvicorn app.main:app --reload   # http://127.0.0.1:8000/docs
```

## 使用

```bash
python scripts/index_docs.py                     # 建索引（首次会下载 ~几十MB ONNX embedding）
python scripts/eval.py                           # 跑 40 条评测（命中率/引用准确率）
python scripts/demo_agent.py "张三还剩几天年假？"  # 命令行跑一遍完整 Agent
python scripts/eval_e2e.py --limit 5     # 端到端回归（真实调 API，判回答/引用/成本）
```

网页对话：启动服务后浏览器打开 **http://127.0.0.1:8000** 即聊（`static/index.html` 单文件页面，无构建、无依赖）。页面走 **SSE 节点级流式（`/api/chat/stream`）**，提问后可见 Agent 逐步工具调用（检索→查库）与最终回答。左侧会话栏可**新建 / 回看 / 切换历史会话**：消息与 checkpoint 落 `data/checkpoints.sqlite`，刷新页面甚至重启服务后仍能恢复并继续对话。

API：
- `POST /api/chat`：同步返回 `{answer, sources}`。
- `POST /api/chat/stream`：SSE 逐步下发节点增量，`done` 事件带最终答案与来源。
- `GET /api/sessions` / `GET /api/sessions/{session_id}/messages`：读会话列表与历史消息（给前端"历史会话"用）。
- `GET /api/runs` / `GET /api/runs/{id}`：运行 trace（耗时/token/成本/错误 + 节点级 steps），可观测用。
- 请求体：`{messages:[{role,content}], session_id?}`。**同一 session 请只追加最新一条消息**（LangGraph 按 thread 自动拼历史，避免重复）。
- **会话持久化**：Agent checkpoint（AsyncSqliteSaver）与消息记录共用 `data/checkpoints.sqlite`（WAL），**服务重启可续聊、历史可读**。

示例对话：问"我今年还剩几天年假？按手册能结转吗？"——Agent 会先 `retrieve_knowledge` 拿《员工手册》结转规则，再 `query_business_db` 拿张三个人剩余天数，两条来源结合作答，末尾列引用。

## 演示截图

网页对话（多轮、含引用来源）：

![网页对话示例](docs/screenshots/web-chat.png)

## 效果度量（简历口径）

`eval/questions.jsonl` 每条含 `expected_source`：
- `scripts/eval.py`（检索层，离线零成本）：`topk_hit_rate` 是否命中正确来源。同一脚本里的 `citation_accuracy` 与它是同一个判定（都看 top-k 来源里有没有真值文件），两个数字不会不同——别当成两个指标讲。
- `scripts/eval_e2e.py`（端到端，真实调 Agent API）：统计回答率（**只判非空**）、引用覆盖（真值来源是否出现在**本轮工具返回的 sources** 里，不判答案文本里有没有引用）、耗时与成本——用于 prompt/模型/工具改动后的回归把关。

40 条是回归冒烟集，不是统计评测；简历别写百分比，写"离线回归集 + 可视化坏例调参"。

## Agent 决策流程

```mermaid
flowchart TD
    A[用户输入] --> B[LangGraph agent 节点]
    B --> C{模型判断: 需要工具?}
    C -- 是 --> D[调用 MCP 工具<br/>retrieve_knowledge 查规则 / query_business_db 查个人数据]
    D --> E{有答案且不再需工具?}
    E -- 否 --> B
    E -- 是 --> F[生成回答 + 引用来源]
    C -- 否 --> F
    F --> G[SSE 节点级流式返回前端]
    B -. 护栏 .-> H[recursion_limit / 单步超时 /<br/>输出截断 / 重复调用检测]
    H -. 中断并基于现有信息收尾 .-> F
```

## 面试可讲点

- **框架**：LangGraph 有状态图、SQLite 断点续聊（AsyncSqliteSaver + WAL，重启不丢会话；高并发生产可换 Postgres）；AutoGen 并入 Microsoft Agent Framework 后我以 LangGraph 为主线。
- **MCP**：工具经 `langchain-mcp-adapters` 以真 MCP（stdio 子进程）接入，不是手写 function calling 的装饰——工具与编排解耦，天然可跨语言复用。
- **RAG**：混合检索（向量 + BM25 做 RRF）去抖 + 可选重排 + 引用溯源 + 评测集验证，坏例能说清怎么调好的。
- **工程化**：SSE 节点级流式、护栏（死循环/超时/上下文截断/重复调用）、懒加载与多轮状态管理。

## 对标 JD 主要能力（面试话术）

| JD 共性要求 | 本项目对应 | 说明 |
|---|---|---|
| 标准化评估闭环 | 40 条金标集 + 两层 Harness | `eval.py` 检索层（离线零成本）+ `eval_e2e.py` Agent 层（`--tag` 落报告、`--compare` 回归 diff，指标含回答率/引用覆盖/成本） |
| Agent 架构与工具调用 | **单 Agent** ReAct 式工具循环 + MCP 工具层 | LangGraph `agent→tools` 条件边做工具路由；决策在模型、编排在框架；多 Agent/Skills 为下一阶段方向（**未实现，不宣称**） |
| 工程化与生产落地 | 上下文管理 / 工具失效处理 / 持久化 / 可观测 | 工具输出截断、重复工具调用判重（窗口=单次运行内 25 步）防死循环、超时异常结构化回填、checkpoint 断点续聊（SQLite）、trace/成本/历史会话 |
| 权限与重试降级 | 见"已知取舍"（生产化方向） | 按用户鉴权、工具失败重试/降级列为生产化改进，**未实现不宣称** |

## 已知取舍 / 改进方向

- 开发用 Chroma，生产切 Milvus / Elasticsearch（换 collection 层即可）。
- 默认 FastEmbed（ONNX，零 torch）；要更准可 `EMBED_BACKEND=flagembedding` 上 bge-m3，或 `RERANK_ENABLED=true` 加重排——均需 `pip install -e ".[embed]"`。**换 embedding 后删除 `data/chroma/` 重建索引**。
- 切分实现 fixed vs recursive 对比；语义 / 父子分块列为改进方向。
- DeepSeek 默认，`.env` 两行即可切 GLM / Qwen。
- `MAX_RECURSION` + prompt 第 5 条 `max_steps` + 重复调用检测 = 三道防死循环。
- 解析层支持 `.md/.txt/.docx/.xlsx/.pdf`（统一抽成纯文本/表格文本）；扫描件/图片类 PDF 无文本层，需 OCR，列为扩展。**替换已有同名文档后请删 `data/chroma/` 重建索引**（增量新增文件可直接 `python scripts/index_docs.py`）。
- 会话 checkpoint 落 SQLite（`data/checkpoints.sqlite`，WAL）：Agent 状态与消息记录分离（checkpoint 表 vs conversations/messages + run_traces）；重启不丢、同一 session 续聊。多进程/高并发生产换 Postgres 并加按用户鉴权与消息分库。成本估算为估算值（单价见 `.env` 的 `LLM_PRICE_*`）。
- 多轮对话历史目前全量进上下文（checkpoint 保存全量消息）；超长会话建议后续加历史压缩/裁剪（如 summarize 节点或 max_turns 截断），列为方向。
- 架构边界明确：当前是**单 Agent + MCP 工具层**；多 Agent 编排 / Skills / 工具失败重试与降级 / prompt 注入防护 是后续方向——面试可主动讲思路，但不写进"已完成"能力。

## 常见坑

- **Anaconda 下 onnxruntime 报 `DLL load failed`**：是 Anaconda 自带旧版 VC 运行库（vcruntime140/msvcp140≈14.29）盖过了系统新版。执行 `conda update -n base -c conda-forge -y vs2015_runtime` 一次即可（torch/bge 同理会遇到）。
- 首次跑 `scripts/index_docs.py` 会从 HuggingFace 下载 embedding 模型（几十 MB）；换模型后务必删 `data/chroma/`。
- 每个 MCP 工具调用都会新起一个 stdio 子进程（真 MCP 的代价）；演示规模无所谓，要提速可把 `app/mcp/servers.py` 改成进程内直连。

