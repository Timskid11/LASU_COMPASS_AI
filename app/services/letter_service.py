"""
Official Letter Generator — structured prompting over Gemma.

TODO (build day): write real templates/examples for the common letter
types your team decides on (e.g. leave of absence, transcript request,
industrial training introduction letter, hostel complaint, etc). Few-shot
examples in the system prompt will improve consistency a lot more than
a generic prompt.
"""
from app.services.llm_client import llm_client

LETTER_TYPES = [
    "transcript_request",
    "siwes_introduction",
    "clearance_request",
    "leave_of_absence",
    "general_request",
]

SYSTEM_PROMPT = """You draft formal, well-structured letters for LASU
students to submit to university offices. Use standard Nigerian
tertiary-institution letter conventions: sender address, date, recipient
address, subject line, formal salutation, clear body, respectful closing.

Respond with ONLY the finished letter text. Do not show your reasoning,
planning, or any commentary before or after the letter.

TODO: paste 1-2 real example letters here as few-shot examples once your
team picks the exact letter types to support — this matters more than
anything else for output quality.
"""


async def generate_letter(letter_type: str, student_details: dict, purpose: str) -> str:
    prompt = f"""Letter type: {letter_type}

Student details: {student_details}

Purpose / context: {purpose}

Write the full formal letter."""

    return await llm_client.generate(prompt=prompt, system=SYSTEM_PROMPT, temperature=0.4)
