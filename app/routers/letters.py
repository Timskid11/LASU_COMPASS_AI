from fastapi import APIRouter
from pydantic import BaseModel
from app.services.letter_service import generate_letter, LETTER_TYPES

router = APIRouter(prefix="/letters", tags=["letters"])


class LetterRequest(BaseModel):
    letter_type: str  # one of LETTER_TYPES
    student_details: dict  # e.g. {"name": ..., "matric_no": ..., "department": ...}
    purpose: str  # free text context for the letter body


class LetterResponse(BaseModel):
    letter: str


@router.get("/types")
def get_letter_types():
    return {"letter_types": LETTER_TYPES}


@router.post("/", response_model=LetterResponse)
async def create_letter(req: LetterRequest):
    letter = await generate_letter(req.letter_type, req.student_details, req.purpose)
    return {"letter": letter}
