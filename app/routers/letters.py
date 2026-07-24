from fastapi import APIRouter
from pydantic import BaseModel
from app.services.letter_service import generate_letter, LETTER_TYPES
from app.services.llm_client import llm_client

router = APIRouter(prefix="/letters", tags=["letters"])

# --- Schemas ---
class LetterRequest(BaseModel):
    letter_type: str  # one of LETTER_TYPES
    student_details: dict  # e.g. {"name": ..., "matric_no": ..., "department": ...}
    purpose: str  # free text context for the letter body

class LetterResponse(BaseModel):
    letter: str

class PolishRequest(BaseModel):
    raw_purpose: str

class PolishResponse(BaseModel):
    polished_purpose: str

# --- Routes ---
@router.get("/types")
def get_letter_types():
    return {"letter_types": LETTER_TYPES}

@router.post("/", response_model=LetterResponse)
async def create_letter(req: LetterRequest):
    letter = await generate_letter(req.letter_type, req.student_details, req.purpose)
    return {"letter": letter}

@router.post("/polish-purpose", response_model=PolishResponse)
async def polish_purpose(request: PolishRequest):
    """
    Takes a rough draft of a student's letter purpose and uses Gemma 
    to rewrite it professionally for official LASU correspondence.
    """
    prompt = f"""You are an expert academic writing assistant for a Lagos State University (LASU) student. 
The student needs to state the 'purpose' of their official letter, but their current draft is too casual or messy.

Rewrite the following rough draft to be highly professional, formal, and clear.
Provide ONLY the polished text. Do not include quotes, introductory phrases like "Here is the polished version:", or conversational filler.

Rough Draft: {request.raw_purpose}
"""
    
    messages = [{"role": "user", "content": prompt}]
    
    # Call the LLM
    polished_text = await llm_client.chat(messages=messages)
    
    # Clean up any accidental whitespace or quotes the model might add
    clean_text = polished_text.strip(' "\'')
    
    return {"polished_purpose": clean_text}