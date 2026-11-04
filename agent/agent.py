"""最小可用 Agent：function calling 循环。
工具：search_knowledge_base（RAG 检索）、get_order_status（模拟订单接口）、calculator。
演示"模型自己决定调什么工具 → 拿到结果 → 继续推理 → 给出最终回答"的工作流。
用法：python agent/agent.py 帮我查订单 A1001 状态，顺便讲讲退款规则"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parent.parent
from config import BASE_URL, API_KEY, CHAT_MODEL
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
sys.path.insert(0, str(ROOT))
from rag.qa import search  # 复用 RAG 检索

MOCK_ORDERS = {"A1001": {"status": "已支付", "amount": 299.0, "course": "继续教育-专技90学时"},
               "A1002": {"status": "退款中", "amount": 199.0, "course": "考证培训班"}}

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
        hits = search(args["query"], k=2)
        return "\n".join(f"[{s}] {c}" for c, s, _ in hits) or "知识库中没有相关内容"
    if name == "get_order_status":
        return json.dumps(MOCK_ORDERS.get(args["order_id"], {"status": "未找到该订单"}), ensure_ascii=False)
    if name == "calculator":
        return str(eval(args["expression"]))  # 演示用途
    return "未知工具"

def main():
    question = " ".join(sys.argv[1:]) or "帮我查一下订单 A1001 的状态，并说明退款和优惠券能否同时使用"
    messages = [{"role": "system", "content": "你是平台智能客服 Agent，可调用工具查证后再回答，回答给出结论与依据。"},
                {"role": "user", "content": question}]
    for step in range(6):  # 最大 6 轮工具调用，防死循环
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print(f"\n【最终回答】\n{msg.content}")
            return
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = run_tool(tc.function.name, args)
            print(f"  [第{step+1}轮] 调用 {tc.function.name}({args}) → {result[:80]}…")
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": result})

if __name__ == "__main__":
    main()
