"""
Single point of truth for "which Gemma backend is active". Everywhere
else in the app imports `llm_client` from here instead of importing
OllamaClient or AIStudioClient directly — so switching backends is a
one-line .env change (GEMMA_BACKEND=ollama or aistudio), not a code change.

Both clients expose the same interface:
    await client.generate(prompt, system=None, temperature=0.3) -> str
    await client.chat(messages, temperature=0.3) -> str
"""
from app.config import settings
from app.services.ollama_client import ollama_client
from app.services.aistudio_client import aistudio_client

if settings.gemma_backend == "aistudio":
    llm_client = aistudio_client
else:
    llm_client = ollama_client
