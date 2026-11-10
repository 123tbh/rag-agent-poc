"""FastAPI 服务：把 RAG 问答与 Agent 工作流包装成 REST API，供前端 / Gradio / 客户系统调用。

启动：
  uvicorn server:app --reload          # http://127.0.0.1:8000/docs 可交互测试
接口：
  GET  /health                       健康检查
  POST /qa        {question, store?, k?}        RAG 带引用问答
  POST /agent     {question, backend?}          Agent 工作流（native 手写 / langgraph）
"""
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.qa import answer, search
from agent.agent import run_agent

app = FastAPI(title="RAG + Agent 演示服务", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QAReq(BaseModel):
    question: str
    store: Optional[str] = None   # "numpy" | "chroma"，缺省读 STORE_BACKEND
    k: int = 3


class AgentReq(BaseModel):
    question: str
    backend: Optional[str] = None  # "native" | "langgraph"，缺省 native


@app.get("/health")
def health():
    return {"status": "ok", "services": ["qa", "agent"]}


@app.post("/qa")
def qa(req: QAReq):
    ans, hits = answer(req.question, k=req.k, store=req.store)
    return {"answer": ans,
            "hits": [{"content": c, "source": s, "score": round(sc, 4)} for c, s, sc in hits]}


@app.post("/agent")
def agent(req: AgentReq):
    backend = req.backend or os.getenv("AGENT_BACKEND", "native")
    if backend == "langgraph":
        from agent import agent_langgraph
        return {"answer": agent_langgraph.run(req.question), "steps": [], "backend": "langgraph"}
    out = run_agent(req.question)
    return {"answer": out["answer"], "steps": out["steps"], "backend": "native"}