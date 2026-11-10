# rag-agent-poc · 企业知识库 RAG + Agent 客服（FDE 求职演示项目）

面向 **AI 前线部署工程师（FDE）** 岗位的端到端演示项目：一套可部署的「企业知识库 AI 客服」，
覆盖 RAG（检索增强问答）与 Agent（模型自主调用工具）两条主流 AI 应用工作流，可直接对接任意
OpenAI 兼容网关（LiteLLM 协议）。知识库使用真实业务规则（折扣计价 / 退款优惠券），非演示数据。

## 技术栈（横向覆盖 FDE 全链路）

| 层 | 选型 | 说明 |
|---|---|---|
| 服务 | FastAPI | `/qa` `/agent` REST API，`server.py` |
| RAG 检索 | numpy 手写余弦 **+** ChromaDB **+** PostgreSQL/pgvector | 三套后端可切换（`--store` / `STORE_BACKEND`） |
| Agent | 手写 function calling **+** LangGraph | `agent/agent.py` 与 `agent/agent_langgraph.py` 对照 |
| 客户 Demo | Gradio | `ui.py` 对话式演示页 |
| 部署 | Docker / Docker Compose | `docker-compose up` 一键拉起 api + ui |

> 设计意图：先手写实现底层原理（150 行内），再接入框架/向量库做对照，面试时既能讲清
> 「框架帮我们管了什么」，又能讲清「不依赖框架时我踩过的坑」。

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # 填入 LLM_BASE_URL / LLM_API_KEY / CHAT_MODEL / EMBED_MODEL

# 1) RAG 检索增强问答
.venv/bin/python rag/ingest.py                                # 构建 numpy 向量库
.venv/bin/python rag/qa.py "退款和优惠券能同时用吗？"

# 2) function calling Agent
.venv/bin/python agent/agent.py "查订单 A1001 状态，并说明退款规则"
.venv/bin/python agent/agent_langgraph.py "查订单 A1001 状态"   # LangGraph 版

# 3) FastAPI 服务（http://127.0.0.1:8000/docs 可交互）
.venv/bin/uvicorn server:app --reload

# 4) Gradio 客户演示页（http://127.0.0.1:7860）
.venv/bin/python ui.py

# 5) Docker 一键部署（api + ui 两个服务）
docker compose up --build
```

## 检索后端切换（numpy ↔ ChromaDB ↔ pgvector）

```bash
# ChromaDB
.venv/bin/python rag/ingest.py --store chroma          # 额外写入 ChromaDB
STORE_BACKEND=chroma .venv/bin/python rag/qa.py "退款规则"

# PostgreSQL + pgvector（先起本地库）
docker compose up -d postgres
.venv/bin/python rag/ingest.py --store pgvector
STORE_BACKEND=pgvector .venv/bin/python rag/qa.py "退款规则"
curl -s localhost:8000/qa -H 'Content-Type: application/json' \
     -d '{"question":"退款和优惠券能同时用吗？","store":"pgvector"}'
```

## 面试可讲的调优点

- **切块**：按段落语义切分 + 超长按句截断（max_len=400），避免规则句被拦腰切断
- **检索**：余弦相似度 top-3；检索不中时模型按 system 约束回答「不知道」而不是幻觉
- **Agent 兜底**：最大 6 轮工具调用上限防死循环、工具结果截断防上下文爆炸
- **LangGraph 对照**：同一套工具分别用手写循环与 `create_react_agent` 实现，讲清
  「状态管理 / 多轮循环 / 工具编排」框架替你做了什么
- **向量库选型**：numpy 手写（可讲余弦/HNSW 原理）→ ChromaDB 持久化（cosine 空间），
  可平滑扩展 FAISS / pgvector / Milvus，重型排（rerank）、查询改写