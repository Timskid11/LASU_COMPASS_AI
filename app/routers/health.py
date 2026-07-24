from fastapi import APIRouter
import httpx
from app.config import settings

router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    backend = settings.gemma_backend
    reachable = False

    try:
        if backend == "aistudio":
            reachable = bool(settings.aistudio_api_key)  # can't cheaply ping without spending a call
        else:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{settings.ollama_base_url}/api/tags")
                reachable = resp.status_code == 200
    except Exception:
        reachable = False

    model = settings.aistudio_model if backend == "aistudio" else settings.ollama_model
    return {"status": "ok", "gemma_backend": backend, "backend_reachable": reachable, "model": model}
