"""
RAG pipeline over the LASU knowledge base
(handbook, academic calendar, SIWES guidelines, clearance process, etc).

TODO (build day):
    1. Populate app/data/ with the actual source docs (txt/pdf).
    2. Run the ingest endpoint to chunk + embed them.
    3. Tune SYSTEM_PROMPT and chunk size for answer quality.

Uses a pure-Python vector store (app/services/vector_store.py) with
Google AI Studio embeddings — no compiled dependencies, nothing to build.
"""
from app.services.llm_client import llm_client
from app.services import vector_store

SYSTEM_PROMPT = """You are the LASU Campus Assistant.

RULES:
1. GREETINGS: If the user input is a simple greeting (e.g., "hi", "hello", "good morning"), completely ignore the context. Respond warmly and ask how you can help with LASU-related matters today.
2. QUESTIONS: For all other questions, answer ONLY using the provided Context.
3. MISSING INFO: If the answer is not in the Context, do not guess. Say exactly: "I don't have that information. Please check with the relevant LASU office or SIWES unit."
4. NO SCRATCHPAD: NEVER output your internal thoughts, reasoning steps, or bulleted self-evaluations. Provide ONLY the final conversational answer.
"""

async def add_documents(chunks: list[str], ids: list[str], metadatas: list[dict] | None = None) -> int:
    """Embed and store chunks. metadatas e.g. [{"source": "handbook.pdf", "section": "SIWES"}]"""
    return await vector_store.add_documents(chunks, ids, metadatas)

async def retrieve(query: str, top_k: int = 4) -> list[dict]:
    return await vector_store.retrieve(query, top_k=top_k)

async def answer_query(query: str, top_k: int = 4, history: list[dict] | None = None) -> dict:
    """
    Full RAG: retrieve context, ask Gemma to synthesize an answer.

    history: prior turns from the client, e.g.
        [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Server stays stateless — the frontend owns and resends conversation
    history on every call. Retrieval is still done fresh each turn using
    only the latest query (not the full history) to keep it simple.
    """
    hits = await retrieve(query, top_k=top_k)
    context = "\n\n---\n\n".join(h["text"] for h in hits) if hits else "(no matching context found)"

    turn_prompt = f"""Context from LASU documents:
{context}

Student input: {query}"""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": turn_prompt})

    answer = await llm_client.chat(messages=messages)
    
    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits],
    }