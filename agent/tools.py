"""Agent 工具集：RAG 检索 / 订单查询 / 计算器。供手写版与 LangGraph 版 Agent 共用。"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from rag.qa import search

MOCK_ORDERS = {"A1001": {"status": "已支付", "amount": 299.0, "course": "继续教育-专技90学时"},
               "A1002": {"status": "退款中", "amount": 199.0, "course": "考证培训班"}}


def search_knowledge_base(query: str, k: int = 2) -> str:
    """在平台业务知识库中检索规则、政策、流程说明。"""
    hits = search(query, k=k)
    return "\n".join(f"[{s}] {c}" for c, s, _ in hits) or "知识库中没有相关内容"


def get_order_status(order_id: str) -> str:
    """查询订单状态、金额与课程。"""
    return json.dumps(MOCK_ORDERS.get(order_id, {"status": "未找到该订单"}), ensure_ascii=False)


def calculator(expression: str) -> str:
    """计算四则运算表达式（演示用途，生产应替换为安全解析器）。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"


TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge_base",
        "description": "在平台业务知识库中检索规则、政策、流程说明",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_order_status",
        "description": "查询订单状态、金额与课程",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "calculator",
        "description": "计算四则运算表达式",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
]


def run_tool(name, args):
    if name == "search_knowledge_base":
        return search_knowledge_base(args.get("query", ""))
    if name == "get_order_status":
        return get_order_status(args.get("order_id", ""))
    if name == "calculator":
        return calculator(args.get("expression", ""))
    return "未知工具"