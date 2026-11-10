"""pgvector 检索后端：PostgreSQL + pgvector 扩展。

本地起库：docker compose up -d postgres（pgvector/pgvector:pg16 镜像）
连接配置走环境变量：PG_HOST / PG_PORT / PG_USER / PG_PASSWORD / PG_DATABASE
"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# 维度占位符 {dim} 在建表时按实际 embedding 维度填充
DDL_TABLE = (
    "CREATE TABLE kb_chunks ("
    "id BIGSERIAL PRIMARY KEY, "
    "content TEXT NOT NULL, "
    "source TEXT NOT NULL, "
    "embedding VECTOR({dim}))"
)
SQL_INSERT = (
    "INSERT INTO kb_chunks (content, source, embedding) VALUES (%s, %s, %s::vector);"
)
SQL_SEARCH = (
    "SELECT content, source, 1 - (embedding <=> %s::vector) AS score "
    "FROM kb_chunks "
    "ORDER BY embedding <=> %s::vector "
    "LIMIT %s;"
)


def _conn():
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        user=os.getenv("PG_USER", "postgres"),
        password=os.getenv("PG_PASSWORD", "postgres"),
        dbname=os.getenv("PG_DATABASE", "ragdemo"),
    )


def _to_vec(vec):
    """list 转 pgvector 字面量字符串，形如 '[0.1,0.2,...]'。"""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def reset_and_load(chunks, sources, vectors):
    """重建 kb_chunks 表并写入全部知识块，向量维度按实际 embedding 自动适配。"""
    dim = len(vectors[0])
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("DROP TABLE IF EXISTS kb_chunks;")
            cur.execute(DDL_TABLE.format(dim=dim))
            rows = [(c, s, _to_vec(v)) for c, s, v in zip(chunks, sources, vectors)]
            cur.executemany(SQL_INSERT, rows)
            cur.execute(
                "CREATE INDEX kb_chunks_cosine ON kb_chunks "
                "USING hnsw (embedding vector_cosine_ops);"
            )
        conn.commit()
    finally:
        conn.close()


def search(qv, k):
    """余弦检索 top-k：pgvector 的 <=> 是 cosine distance，相似度 = 1 - distance。"""
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(SQL_SEARCH, (_to_vec(qv), _to_vec(qv), k))
            return [(c, s, float(score)) for c, s, score in cur.fetchall()]
    finally:
        conn.close()