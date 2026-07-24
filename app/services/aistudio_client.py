"""
Client for Gemma via Google AI Studio's Generative Language API — an
alternative to local Ollama for laptops without the RAM to run Gemma
locally. Same method names/shapes as OllamaClient so the rest of the
app doesn't care which backend is active.

Setup:
    1. Get a free API key: https://aistudio.google.com/apikey
    2. Put it in .env as AISTUDIO_API_KEY
    3. Set GEMMA_BACKEND=aistudio in .env

TODO (build day): confirm the exact current Gemma model id in AI Studio
(model names/versions do change) and update AISTUDIO_MODEL in .env if
needed — check https://aistudio.google.com for the list.

Needs internet — this is the tradeoff vs. Ollama's offline story, so
decide as a team which backend the live demo actually runs on.
"""
import httpx
from app.config import settings

GENERATIVE_LANGUAGE_BASE = "https://generativelanguage.googleapis.com/v1beta"


class AIStudioClient:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.aistudio_api_key
        self.model = model or settings.aistudio_model

    async def generate(self, prompt: str, system: str | None = None, temperature: float = 0.3) -> str:
        return await self._call(
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            system=system,
            temperature=temperature,
        )

    async def chat(self, messages: list[dict], temperature: float = 0.3) -> str:
        """messages = [{"role": "user"/"assistant"/"system", "content": "..."}]"""
        system = None
        contents = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]  # AI Studio takes system separately
                continue
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        return await self._call(contents=contents, system=system, temperature=temperature)

    async def _call(self, contents: list[dict], system: str | None, temperature: float) -> str:
        if not self.api_key:
            raise RuntimeError(
                "AISTUDIO_API_KEY not set. Get one at https://aistudio.google.com/apikey"
            )

        url = f"{GENERATIVE_LANGUAGE_BASE}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                return ""


aistudio_client = AIStudioClient()
