"""RAG 问答：问题 → 向量检索 top-k → 拼上下文 → 大模型生成带引用的回答。
检索后端可选 numpy（手写余弦检索，默认）、chroma（ChromaDB）或 pgvector（PostgreSQL）。

CLI 用法：
  python rag/qa.py 你的问题                       # numpy 后端
  STORE_BACKEND=chroma python rag/qa.py 你的问题  # ChromaDB 后端
  STORE_BACKEND=pgvector python rag/qa.py 你的问题  # pgvector 后端
"""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL, EMBED_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

_np_store = None
_chroma_col = None


def _embed(text: str) -> np.ndarray:
    qv = np.array(client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding,
                  dtype=np.float32)
    return qv / np.linalg.norm(qv)


def _numpy_search(qv, k):
    global _np_store
    if _np_store is None:
        _np_store = np.load(ROOT / "rag" / "vectors.npz", allow_pickle=False)
    scores = _np_store["vectors"] @ qv
    top = scores.argsort()[::-1][:k]
    return [(str(_np_store["chunks"][i]), str(_np_store["sources"][i]), float(scores[i]))
            for i in top]


def _chroma_search(qv, k):
    global _chroma_col
    if _chroma_col is None:
        import chromadb
        cli = chromadb.PersistentClient(path=str(ROOT / "rag" / "chroma"))
        _chroma_col = cli.get_or_create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})
    res = _chroma_col.query(query_embeddings=[qv.tolist()], n_results=k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    # cosine 空间下 distance = 1 - cos，转回相似度分数
    return [(d, m.get("source", ""), float(1.0 - dist)) for d, m, dist in zip(docs, metas, dists)]


def search(query: str, k: int = 3, store=None):
    """检索：返回 top-k（块, 来源, 分数）。store None 时读环境变量 STORE_BACKEND。
    支持 numpy（手写余弦）/ chroma（ChromaDB）/ pgvector（PostgreSQL）三种后端。
    """
    store = store or os.getenv("STORE_BACKEND", "numpy")
    qv = _embed(query)
    if store == "chroma":
        return _chroma_search(qv, k)
    if store == "pgvector":
        from rag.pg_store import search as pg_search
        return pg_search(qv.tolist(), k)
    return _numpy_search(qv, k)


def answer(question: str, k: int = 3, store=None):
    """RAG 生成：检索 → 拼上下文 → 生成带引用回答。返回 (回答, hits)。"""
    hits = search(question, k=k, store=store)
    context = "\n---\n".join(f"[{s} 分数{sc:.2f}] {c}" for c, s, sc in hits)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是平台客服助手。只依据下面的知识库片段回答，"
                                          "回答末尾用 [来源文件名] 标注引用；知识库没有的内容如实说不知道。"},
            {"role": "user", "content": f"知识库片段：\n{context}\n\n问题：{question}"},
        ],
    )
    return resp.choices[0].message.content, hits


def main():
    question = " ".join(sys.argv[1:]) or "退款和优惠券能同时用吗？"
    ans, _ = answer(question)
    print(ans)


if __name__ == "__main__":
    main()