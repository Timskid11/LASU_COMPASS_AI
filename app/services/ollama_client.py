"""
Thin wrapper around the local Ollama server running Gemma.

Setup (do this before the day, per the workshop):
    ollama pull gemma4          # or gemma4:e2b on 8GB laptops
    ollama run gemma4           # confirms it works

Ollama exposes an OpenAI-compatible-ish REST API on port 11434.
Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""
import httpx
from app.config import settings


class OllamaClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        """
        Single-shot generation. Good for structured prompting (letters,
        classification, RAG answer synthesis).
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "")

    async def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """
        Multi-turn chat. messages = [{"role": "user"/"assistant"/"system", "content": "..."}]
        Use this for the Campus Assistant conversational endpoint.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")


ollama_client = OllamaClient()
