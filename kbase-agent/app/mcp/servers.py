"""MCP 工具服务：检索（知识库） + 业务库（HR 个人数据）。

运行方式（被 LangGraph agent 以 stdio 拉起）：python -m app.mcp.servers
单个工具通过 FastMCP 声明，天然支持跨语言/跨进程复用；进程内也可直接 import 调用。
"""

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from app.retrieval.pipeline import RetrievalPipeline

ROOT = Path(__file__).resolve().parents[2]
RECORDS_FILE = ROOT / "data" / "business" / "records.jsonl"

mcp = FastMCP("kbase-tools")
_pipeline: RetrievalPipeline | None = None

# 知识库只有"制度规则"；个人状态（年假剩余/报销进度等）在业务系统里。
# 这样 Agent 才需要"先查知识库拿规则 -> 再查业务库拿个人数据"两次工具调用。
DEFAULT_EMPLOYEE_RECORDS = [
    {
        "name": "张三",
        "annual_leave_total": 10,
        "annual_leave_remaining": 6,
        "overtime_balance_hours": 12,
        "latest_expense": {"date": "2026-08-12", "amount": 980, "status": "审批中"},
    },
    {
        "name": "李四",
        "annual_leave_total": 10,
        "annual_leave_remaining": 1,
        "overtime_balance_hours": 0,
        "latest_expense": {"date": "2026-08-20", "amount": 420, "status": "已打款"},
    },
    {
        "name": "王五",
        "annual_leave_total": 10,
        "annual_leave_remaining": 9,
        "overtime_balance_hours": 6,
        "latest_expense": None,
    },
]


def set_pipeline(pipeline: RetrievalPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def _ensure_records_file() -> None:
    if RECORDS_FILE.exists():
        return
    RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RECORDS_FILE.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in DEFAULT_EMPLOYEE_RECORDS
        )
        + "\n",
        encoding="utf-8",
    )


@mcp.tool()
def retrieve_knowledge(question: str) -> str:
    """从企业知识库（员工手册 / 产品 FAQ）检索制度规则，返回带【来源：文件名】的片段。"""
    if _pipeline is None:
        return "错误：检索管道未初始化。"
    try:
        hits = _pipeline.retrieve(question, top_k=3)
    except Exception as exc:
        return f"错误：{exc}"
    if not hits:
        return "知识库中未找到与问题相关的内容，请如实告知用户资料中暂无此信息。"
    parts = []
    for hit in hits:
        source = hit.get("source", "未知")
        parts.append(f"【来源：{source}】\n{hit['content']}")
    return "\n\n---\n\n".join(parts)


@mcp.tool()
def query_business_db(question: str) -> str:
    """查内部业务系统拿员工的个人状态数据：年假剩余天数、加班调休余额、最新报销单状态。
    只回答"个人数据"，不含制度规则（制度请调 retrieve_knowledge）。
    必须在问题里明确员工姓名；姓名缺失或有歧义时本工具会拒绝并说明原因，不会猜测默认员工。"""
    _ensure_records_file()
    records = [
        json.loads(line)
        for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        return "业务系统中暂无员工记录。"

    # 身份解析：只认"问题里出现的员工姓名"，且要求唯一。
    # 早期实现用 records[0] 兜底"没写姓名"的情况，会把张三的年假/报销金额当成提问人的数据返回
    # （静默错人）；多姓名时也只取列表里第一个命中，同样是静默错人。两者都改为显式拒绝。
    # 另注（演示边界）：本工具**没有鉴权**——它是把"问题文本"当身份来源，而非绑定会话用户，
    # 因此无法阻止用户询问他人数据。生产化应改为由会话层注入 user_id 并做行级过滤，
    # 该限制已在 README「已知取舍」如实写明，不对外宣称具备权限控制。
    matched_names = [r["name"] for r in records if r.get("name") and r["name"] in question]
    if not matched_names:
        return (
            "无法确定员工身份：请在问题中明确员工姓名（例如“张三还剩几天年假”）。"
            "为避免返回他人数据，本工具不会猜测默认员工。"
        )
    if len(matched_names) > 1:
        return (
            f"问题中出现了多个员工姓名（{'、'.join(matched_names)}），身份不唯一，"
            "请一次只询问一位员工。"
        )
    # 与上面那行同一口径：这里也必须用 .get("name")——records.jsonl 是面向用户可编辑的数据文件，
    # 只要混进一条没有 name 键、且排在命中记录之前的行，直接下标就会 KeyError（工具直接报错）。
    matched = next(r for r in records if r.get("name") == matched_names[0])
    return json.dumps(
        {
            "employee": matched["name"],
            "annual_leave_total": matched["annual_leave_total"],
            "annual_leave_remaining": matched["annual_leave_remaining"],
            "overtime_balance_hours": matched["overtime_balance_hours"],
            "latest_expense": matched["latest_expense"],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    try:
        pipeline = RetrievalPipeline()
        pipeline.ensure_ready()
        set_pipeline(pipeline)
    except Exception as exc:
        print(f"[mcp] 检索索引暂不可用，retrieve_knowledge 将返回错误：{exc}", file=sys.stderr)
    mcp.run()  # 默认 stdio 传输：被 `python -m app.mcp.servers` 拉起
