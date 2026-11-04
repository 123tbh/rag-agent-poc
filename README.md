# rag-agent-poc · RAG 检索增强 + function calling Agent 最小实战

面向 FDE / 全栈岗位的求职演示项目：不依赖任何向量库与重框架，用 ~150 行 Python 实现
RAG（检索增强问答）与 Agent（工具自主调用）两条主流 AI 应用工作流，可直接对接任意
OpenAI 兼容网关。

基于 OpenAI 兼容网关（LiteLLM 协议）构建的最小 RAG 检索增强问答与 function calling Agent，
知识库使用真实业务规则（折扣计价 / 退款优惠券），可演示"检索增强回答"与"模型自主调用工具"两种工作流。

## 运行
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt python-dotenv
# 把 docs/ 换成任意知识库文档
.venv/bin/python rag/ingest.py
.venv/bin/python rag/qa.py "退款和优惠券能同时用吗？"
.venv/bin/python agent/agent.py "查订单 A1001 状态，并说明退款规则"
```

## 面试可讲的调优点
- **切块**：按段落语义切分 + 超长按句截断（max_len=400），避免规则句被拦腰切断
- **检索**：余弦相似度 top-3；检索不中时模型按 system 约束回答"不知道"而不是幻觉
- **Agent 兜底**：最大 6 轮工具调用上限防死循环；工具结果截断防上下文爆炸
- **可扩展**：向量库可换 FAISS/pgvector，可加重排（rerank）、查询改写、流式输出
