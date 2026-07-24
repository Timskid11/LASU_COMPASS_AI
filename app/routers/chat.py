from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_service import answer_query

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    query: str
    top_k: int = 4
    history: list[ChatMessage] = []  # frontend resends prior turns; server is stateless


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@router.post("/", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main Campus Assistant endpoint. RAG over the LASU knowledge base
    (handbook, calendar, SIWES guidelines, clearance, etc).

    Conversational: pass prior turns in `history` to let follow-up
    questions ("what about SIWES?") resolve correctly. No server-side
    session state — the client owns and resends history each call.
    """
    history_dicts = [m.model_dump() for m in req.history]
    result = await answer_query(req.query, top_k=req.top_k, history=history_dicts)
    return result
