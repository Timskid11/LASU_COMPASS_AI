import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Which backend actually serves Gemma: "ollama" (local, offline) or
    # "aistudio" (cloud, needs internet + API key). Switch freely between
    # dev machines — same code, same request/response shapes either way.
    gemma_backend: str = os.getenv("GEMMA_BACKEND", "ollama")

    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4")

    # Get a free key at https://aistudio.google.com/apikey
    aistudio_api_key: str = os.getenv("AISTUDIO_API_KEY", "")
    aistudio_model: str = os.getenv("AISTUDIO_MODEL", "gemma-4-26b-a4b-it")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")

    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store.json")
    app_env: str = os.getenv("APP_ENV", "development")


settings = Settings()
