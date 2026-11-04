"""知识库入库：读 docs/*.md → 按段落切块 → 逐块向量化 → 存 rag/vectors.npz"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from openai import OpenAI
from config import BASE_URL, API_KEY, EMBED_MODEL

ROOT = pathlib.Path(__file__).resolve().parent.parent
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def chunk_text(text: str, max_len: int = 400):
    """按空行分段，超长段落再按句切，块间保留来源便于引用。"""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for p in paras:
        while len(p) > max_len:
            cut = p.rfind("。", 0, max_len)
            cut = cut + 1 if cut > 0 else max_len
            chunks.append(p[:cut]); p = p[cut:]
        if p: chunks.append(p)
    return chunks

def embed(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def main():
    docs_dir = ROOT / "docs"
    chunks, sources = [], []
    for f in sorted(docs_dir.glob("*.md")):
        for c in chunk_text(f.read_text(encoding="utf-8")):
            chunks.append(c); sources.append(f.name)
    print(f"共 {len(chunks)} 个知识块，开始向量化…")
    vecs = embed(chunks)
    np.savez(ROOT / "rag" / "vectors.npz",
             vectors=vecs / np.linalg.norm(vecs, axis=1, keepdims=True),
             chunks=np.array(chunks), sources=np.array(sources))
    print("已写入 rag/vectors.npz")

if __name__ == "__main__":
    main()
