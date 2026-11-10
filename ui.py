"""Gradio 客户演示页：对话式调用 /qa 与 /agent，给客户看效果。

启动：
  python ui.py                      # 默认连 http://localhost:8000，监听 7860
  API_URL=http://api:8000 python ui.py   # Docker Compose 内连 api 服务
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import gradio as gr
import requests

API = os.getenv("API_URL", "http://localhost:8000")


def chat(message, history, mode):
    endpoint = "qa" if mode == "RAG 问答" else "agent"
    try:
        r = requests.post(f"{API}/{endpoint}", json={"question": message}, timeout=120)
        r.raise_for_status()
        return r.json().get("answer", "（无回答）")
    except Exception as e:
        return f"请求失败：{e}"


with gr.Blocks(title="企业知识库 AI 客服演示") as demo:
    gr.Markdown(
        "# 企业知识库 AI 客服演示\n"
        "基于 **FastAPI + RAG + Function Calling Agent**，可切换「RAG 带引用问答」与 "
        "「Agent 多轮工具调用」两种工作流。知识库为真实业务规则（折扣计价 / 退款优惠券）。"
    )
    mode = gr.Radio(["RAG 问答", "Agent 工作流"], value="Agent 工作流", label="模式")
    gr.ChatInterface(fn=chat, additional_inputs=[mode])


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_PORT", "7860")))