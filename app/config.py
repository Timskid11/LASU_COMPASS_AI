import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4")
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
    app_env: str = os.getenv("APP_ENV", "development")


settings = Settings()
