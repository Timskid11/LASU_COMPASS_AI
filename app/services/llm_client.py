import os
import httpx

API_KEY = os.getenv("GEMMA_API_KEY", "").strip()
raw_url = os.getenv("GEMMA_API_URL", "").strip()

# Sanitize URL with fallback to OpenRouter
if not raw_url:
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
elif not raw_url.startswith(("http://", "https://")):
    API_URL = f"https://{raw_url}"
else:
    API_URL = raw_url

MODEL_NAME = os.getenv("GEMMA_MODEL", "google/gemma-2-9b-it:free").strip()


class LLMClient:
    async def chat(self, messages: list[dict]) -> str:
        """
        Sends a RAG payload to an OpenAI-compatible endpoint (like OpenRouter).
        """
        if not API_KEY:
            print("ERROR: GEMMA_API_KEY is missing from environment variables.")
            return "Server configuration error: GEMMA_API_KEY is not set."

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://lasu-compass-ai.onrender.com",  # Required by OpenRouter for free tier
            "X-Title": "LASU Compass AI",
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "temperature": 0.1
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(API_URL, headers=headers, json=payload, timeout=45.0)
                
                # If OpenRouter returns an error code, log the exact body to Render logs
                if response.status_code != 200:
                    print(f"LLM API Error ({response.status_code}): {response.text}")
                    return f"The LLM provider returned an error ({response.status_code}). Please check server logs."

                data = response.json()
                return data["choices"][0]["message"]["content"]

        except Exception as e:
            print(f"LLM Client Exception: {e}")
            return "An internal connection error occurred while contacting the AI service."


llm_client = LLMClient()