"""LangGraph 版 Agent：用 create_react_agent + @tool 重写同一个客服工作流。
与 agent/agent.py（手写循环）对照，展示框架帮我们管理「状态、多轮循环、工具编排」这件事。

CLI 用法：python agent/agent_langgraph.py 帮我查订单 A1001 状态
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from config import BASE_URL, API_KEY, CHAT_MODEL
from agent.tools import search_knowledge_base, get_order_status, calculator


@tool
def search_kb(query: str) -> str:
    """在平台业务知识库中检索规则、政策、流程说明。"""
    return search_knowledge_base(query)


@tool
def get_order(order_id: str) -> str:
    """查询订单状态、金额与课程。"""
    return get_order_status(order_id)


@tool
def calc(expression: str) -> str:
    """计算四则运算表达式。"""
    return calculator(expression)


def build_agent():
    llm = ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=CHAT_MODEL)
    return create_react_agent(llm, [search_kb, get_order, calc])


def run(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


def main():
    question = " ".join(sys.argv[1:]) or "帮我查一下订单 A1001 的状态，并说明退款和优惠券能否同时使用"
    print(run(question))


if __name__ == "__main__":
    main()