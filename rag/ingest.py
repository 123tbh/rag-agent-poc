"""知识库入库：读 docs/*.md → 按段落切块 → 逐块向量化 → 写 numpy(npz) + 可选 ChromaDB。

CLI 用法：
  python rag/ingest.py                 # 仅 numpy（手写余弦检索基线）
  python rag/ingest.py --store chroma  # 同时写 ChromaDB 向量库
"""
import sys, pathlib, argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=["numpy", "chroma", "pgvector"], default="numpy",
                        help="额外写入的向量库后端")
    args = parser.parse_args()

    docs_dir = ROOT / "docs"
    chunks, sources = [], []
    for f in sorted(docs_dir.glob("*.md")):
        for c in chunk_text(f.read_text(encoding="utf-8")):
            chunks.append(c); sources.append(f.name)
    print(f"共 {len(chunks)} 个知识块，开始向量化…")
    vecs = embed(chunks)
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    np.savez(ROOT / "rag" / "vectors.npz",
             vectors=norm, chunks=np.array(chunks), sources=np.array(sources))
    print("已写入 rag/vectors.npz")

    if args.store == "chroma":
        import chromadb
        cli = chromadb.PersistentClient(path=str(ROOT / "rag" / "chroma"))
        try:
            cli.delete_collection("knowledge_base")
        except Exception:
            pass
        col = cli.create_collection("knowledge_base", metadata={"hnsw:space": "cosine"})
        col.add(ids=[f"c{i}" for i in range(len(chunks))],
                embeddings=norm.tolist(),
                documents=chunks,
                metadatas=[{"source": s} for s in sources])
        print("已写入 rag/chroma（ChromaDB，cosine 空间）")

    if args.store == "pgvector":
        from rag.pg_store import reset_and_load
        reset_and_load(chunks, sources, norm.tolist())
        print("已写入 PostgreSQL + pgvector")


if __name__ == "__main__":
    main()