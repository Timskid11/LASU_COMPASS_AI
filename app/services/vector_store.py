"""
Minimal pure-Python vector store. No compiled dependencies (no numpy,
no chromadb) — nothing to build, works on any machine with just httpx.
Fine at hackathon scale (hundreds of chunks): cosine similarity over a
JSON file is plenty fast.

Persists to a single JSON file so re-ingesting isn't required every
server restart.
"""
import json
import math
import os
from app.config import settings
from app.services.embeddings import embed_text

_STORE_PATH = settings.vector_store_path
_store: list[dict] = []  # [{"id":..., "text":..., "metadata":..., "embedding":[...]}]


def _load():
    global _store
    if os.path.exists(_STORE_PATH):
        with open(_STORE_PATH, "r", encoding="utf-8") as f:
            _store = json.load(f)
    else:
        _store = []


def _save():
    os.makedirs(os.path.dirname(_STORE_PATH) or ".", exist_ok=True)
    with open(_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(_store, f)


_load()


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def add_documents(chunks: list[str], ids: list[str], metadatas: list[dict] | None = None) -> int:
    """Embed and store chunks. metadatas e.g. [{"source": "handbook.pdf", "section": "SIWES"}]"""
    metadatas = metadatas or [{} for _ in chunks]
    for chunk, doc_id, meta in zip(chunks, ids, metadatas):
        embedding = await embed_text(chunk)
        _store.append({"id": doc_id, "text": chunk, "metadata": meta, "embedding": embedding})
    _save()
    return len(chunks)


def clear():
    """Wipe the store — call before a fresh /ingest/ if you want to avoid duplicates."""
    global _store
    _store = []
    _save()


async def retrieve(query: str, top_k: int = 4) -> list[dict]:
    if not _store:
        return []
    query_embedding = await embed_text(query)
    scored = [
        (_cosine(query_embedding, item["embedding"]), item)
        for item in _store
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": item["text"], "metadata": item["metadata"]} for _, item in scored[:top_k]]
