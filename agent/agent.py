"""最小可用 Agent（手写 function calling 循环版）。
与 agent/agent_langgraph.py（LangGraph 版）对照，展示「手写状态机」与「框架图编排」的差异。

CLI 用法：python agent/agent.py 帮我查订单 A1001 状态，顺便讲讲退款规则
"""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL
from agent.tools import TOOLS, run_tool

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def run_agent(question: str, max_steps: int = 6):
    """手写 Agent 循环：模型自主决定调工具 → 拿结果 → 继续推理 → 最终回答。
    返回 {"answer": str, "steps": [{"round", "tool", "args", "result"}]}。
    """
    messages = [{"role": "system",
                 "content": "你是平台智能客服 Agent，可调用工具查证后再回答，回答给出结论与依据。"},
                {"role": "user", "content": question}]
    steps = []
    for step in range(max_steps):  # 最大 6 轮工具调用，防死循环
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return {"answer": msg.content or "", "steps": steps}
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = run_tool(tc.function.name, args)
            steps.append({"round": step + 1, "tool": tc.function.name,
                          "args": args, "result": result[:200]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return {"answer": f"（达到 {max_steps} 轮调用上限，已终止）", "steps": steps}


def main():
    question = " ".join(sys.argv[1:]) or "帮我查一下订单 A1001 的状态，并说明退款和优惠券能否同时使用"
    out = run_agent(question)
    for s in out["steps"]:
        print(f"  [第{s['round']}轮] 调用 {s['tool']}({s['args']}) → {s['result'][:80]}…")
    print(f"\n【最终回答】\n{out['answer']}")


if __name__ == "__main__":
    main()