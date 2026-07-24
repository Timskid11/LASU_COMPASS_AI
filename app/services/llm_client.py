import os
import httpx

# Pull the new OpenRouter/Groq credentials from Render/Environment
API_KEY = os.getenv("GEMMA_API_KEY", "")
API_URL = os.getenv("GEMMA_API_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.getenv("GEMMA_MODEL", "google/gemma-2-9b-it:free")

class LLMClient:
    async def chat(self, messages: list[dict]) -> str:
        """
        Sends a RAG payload to an OpenAI-compatible endpoint (like OpenRouter).
        Expects messages in format: [{"role": "system", "content": "..."}, ...]
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
            "temperature": 0.1  # Kept low so it doesn't hallucinate outside of LASU documents
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(API_URL, headers=headers, json=payload, timeout=45.0)
            
            # If the API provider throws an error, catch it
            response.raise_for_status()
            
            # Extract Gemma's response from the JSON body
            data = response.json()
            return data["choices"][0]["message"]["content"]

# Export the instance so rag_service.py can use it
llm_client = LLMClient()