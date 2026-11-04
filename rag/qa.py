"""RAG 问答 CLI：问题 → 向量检索 top-k → 拼上下文 → 大模型生成带引用的回答。
用法：python rag/qa.py 你的问题"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from openai import OpenAI
from config import BASE_URL, API_KEY, CHAT_MODEL, EMBED_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
store = np.load(ROOT / "rag" / "vectors.npz", allow_pickle=False)

def search(query: str, k: int = 3):
    """检索：问题向量化后与知识块做余弦相似度，返回 top-k（块, 来源, 分数）。"""
    qv = np.array(client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding,
                  dtype=np.float32)
    qv = qv / np.linalg.norm(qv)
    scores = store["vectors"] @ qv
    top = scores.argsort()[::-1][:k]
    return [(str(store["chunks"][i]), str(store["sources"][i]), float(scores[i])) for i in top]

def main():
    question = " ".join(sys.argv[1:]) or "退款和优惠券能同时用吗？"
    hits = search(question)
    context = "\n---\n".join(f"[{s} 分数{sc:.2f}] {c}" for c, s, sc in hits)
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": "你是平台客服助手。只依据下面的知识库片段回答，"
                                          "回答末尾用 [来源文件名] 标注引用；知识库没有的内容如实说不知道。"},
            {"role": "user", "content": f"知识库片段：\n{context}\n\n问题：{question}"},
        ],
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
