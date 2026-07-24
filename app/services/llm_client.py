import os
import httpx

API_KEY = os.getenv("GEMMA_API_KEY", "")
raw_url = os.getenv("GEMMA_API_URL", "").strip()

# Fallback to OpenRouter if GEMMA_API_URL is empty or invalid
if not raw_url:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
elif not raw_url.startswith(("http://", "https://")):
    API_URL = f"https://{raw_url}"
else:
    API_URL = raw_url

MODEL_NAME = os.getenv("GEMMA_MODEL", "google/gemma-2-9b-it:free")


class LLMClient:
    async def chat(self, messages: list[dict]) -> str:
        """
        Sends a RAG payload to an OpenAI-compatible endpoint (like OpenRouter).
        """
        if not API_KEY:
            return "Error: GEMMA_API_KEY is not configured on the server."

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.1
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, headers=headers, json=payload, timeout=45.0)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


llm_client = LLMClient()