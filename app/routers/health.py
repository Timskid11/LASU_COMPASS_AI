from fastapi import APIRouter
import httpx
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = resp.status_code == 200
    except Exception:
        ollama_ok = False

    return {"status": "ok", "ollama_reachable": ollama_ok, "model": settings.ollama_model}
