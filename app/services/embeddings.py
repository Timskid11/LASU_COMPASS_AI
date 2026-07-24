"""
Text embeddings via Google AI Studio (same API key as your Gemma calls).
Used by the pure-Python vector store instead of a local embedding model —
keeps the whole stack free of compiled dependencies.
"""
import httpx
from app.config import settings

BASE = "https://generativelanguage.googleapis.com/v1beta"


async def embed_text(text: str) -> list[float]:
    url = f"{BASE}/models/{settings.embedding_model}:embedContent?key={settings.aistudio_api_key}"
    payload = {"content": {"parts": [{"text": text}]}}

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["embedding"]["values"]
