"""
RAG pipeline over the LASU knowledge base
"""
import re
from app.services.llm_client import llm_client
from app.services import vector_store

# Updated prompt forcing detailed, bulleted responses
SYSTEM_PROMPT = """You are the LASU Campus Assistant.

RULES:
1. GREETINGS: If the user input is a simple greeting (e.g., "hi", "hello"), completely ignore the context. Respond warmly and ask how you can help today.
2. QUESTIONS: Answer the student's request using ONLY the provided Context. Provide highly detailed, comprehensive, and helpful answers. Break down long processes (like clearance or SIWES registration) into clear, step-by-step bullet points. Expand on the information provided in the context; do not give brief or one-sentence answers.
3. MISSING INFO: If the answer is not in the Context, do not guess. Say exactly: "I don't have specific details on that in the documents. Please check with the relevant LASU office or SIWES unit."
4. NO SCRATCHPAD: NEVER output internal reasoning, thought processes, or self-corrections. Provide ONLY the final conversational response.
"""

async def add_documents(chunks: list[str], ids: list[str], metadatas: list[dict] | None = None) -> int:
    """Embed and store chunks. metadatas e.g. [{"source": "handbook.pdf", "section": "SIWES"}]"""
    return await vector_store.add_documents(chunks, ids, metadatas)


async def retrieve(query: str, top_k: int = 4) -> list[dict]:
    return await vector_store.retrieve(query, top_k=top_k)


async def answer_query(query: str, top_k: int = 4, history: list[dict] | None = None) -> dict:
    """
    Full RAG: retrieve context, ask Gemma to synthesize an answer.
    """
    # 1. Catch simple greetings instantly (bypass RAG and LLM completely)
    clean_query = re.sub(r'[^\w\s]', '', query.strip().lower())
    if clean_query in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]:
        return {
            "answer": "Hello! I am your LASU Campus Assistant. How can I help you with your clearance, SIWES, or other academic questions today?",
            "sources": []
        }

    # 2. For everything else, do the normal RAG process
    hits = await retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join(h["text"] for h in hits) if hits else "(no matching context found)"

    turn_prompt = f"""Context from LASU documents:
{context}

Student input: {query}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": turn_prompt})

    answer = await llm_client.chat(messages=messages)
    
    # 3. Add direct PDF download links to source metadata
    formatted_sources = []
    for h in hits:
        meta = dict(h.get("metadata", {}))
        source_file = meta.get("source") or meta.get("file_name") or ""
        
        if source_file:
            # Build the full public URL so the frontend can make the chips clickable
            meta["url"] = f"https://lasu-compass-ai.onrender.com/documents/{source_file}"
            
        formatted_sources.append(meta)
    
    return {
        "answer": answer,
        "sources": formatted_sources,
    }