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

SYSTEM_PROMPT = """You are the LASU Campus Assistant. Answer ONLY using the
provided context from official LASU documents (handbook, academic calendar,
SIWES guidelines, clearance process, etc). If the answer isn't in the
context, say you don't have that information and suggest where the student
might check (e.g. the relevant office). Be concise and specific.

Respond with ONLY the final answer as plain, direct sentences. Do not show
your reasoning, planning, options considered, or step-by-step thinking —
students should see a clean answer, not a scratchpad.

TODO: refine this prompt during the build — add tone/format guidance,
citation style (e.g. "per the Student Handbook §3.2"), etc.
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

Student question: {query}

Answer the question using only the context above."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": turn_prompt})

    answer = await llm_client.chat(messages=messages)
    return {
        "answer": answer,
        "sources": [h["metadata"] for h in hits],
    }
