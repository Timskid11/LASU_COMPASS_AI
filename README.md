# LASU Campus Assistant — Backend Scaffold

GDGoC LASU "Build with Gemma" hackathon. This is **prep scaffolding only** —
per the workshop rules, no real project logic gets finished until the build
window opens on July 24 (9:50 AM–2:10 PM). What's here: structure,
endpoints, and Gemma wiring so the team can move fast on the day.

## Stack
- FastAPI (backend)
- Ollama running `gemma4` (or `gemma4:e2b` on 8GB laptops) — local, offline
- ChromaDB for the RAG vector store (also runs fully offline)

## Setup (do this before the day)
```bash
# 1. Install & pull Gemma
ollama pull gemma4        # or: ollama pull gemma4:e2b
ollama run gemma4         # sanity check, then Ctrl+D to exit

# 2. Python env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Config
cp .env.example .env

# 4. Run
uvicorn app.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive Swagger docs.

## Endpoints (for the team)

| Method | Path            | Purpose                                                        |
|--------|-----------------|------------------------------------------------------------------|
| GET    | `/health`       | Checks the API is up and Ollama is reachable                    |
| POST   | `/chat/`        | Main Campus Assistant — RAG Q&A over the LASU knowledge base    |
| GET    | `/letters/types`| Lists supported letter types                                    |
| POST   | `/letters/`     | Generates a formal letter from student details + purpose        |
| POST   | `/ingest/`      | (Re)loads documents from `app/data/` into the vector store       |

### `POST /chat/`
Stateless — no server-side sessions. The frontend keeps the running
conversation and resends it as `history` on every call.
```json
// request
{
  "query": "What about SIWES?",
  "top_k": 4,
  "history": [
    {"role": "user", "content": "What's needed for clearance?"},
    {"role": "assistant", "content": "You need to visit..."}
  ]
}

// response
{ "answer": "...", "sources": [{"source": "handbook.pdf", "section": "3.3 SIWES Guidelines"}] }
```
First message of a conversation: omit `history` or send `[]`.

### `POST /letters/`
```json
// request
{
  "letter_type": "transcript_request",
  "student_details": {"name": "...", "matric_no": "...", "department": "..."},
  "purpose": "Applying for a master's program abroad, need 2 official copies."
}

// response
{ "letter": "..." }
```

## Still to do on build day
- [ ] Drop real LASU documents into `app/data/` (handbook, calendar, SIWES
      guidelines, course registration guide, clearance process, office
      directory, admission guide, faculty info, contacts, student affairs)
- [ ] `POST /ingest/` once docs are in place
- [ ] Tune `SYSTEM_PROMPT` in `app/services/rag_service.py` for answer quality
- [ ] Add few-shot letter examples in `app/services/letter_service.py`
- [ ] Confirm `gemma4` vs `gemma4:e2b` per teammate laptop RAM
- [ ] Wire frontend to these endpoints
- [ ] Lock down CORS origins in `app/main.py` before demo

## Architecture note (for the Writeup)
Structured prompting + RAG: Gemma answers only from retrieved LASU
document chunks (`app/services/rag_service.py`), keeping it grounded and
avoiding hallucinated policy info. The letter generator uses a separate
structured-prompting path, not retrieval.
