"""
Turns raw docs (txt/pdf) in app/data/ into clean, well-scoped chunks in
the vector store.

Pipeline: read -> clean -> chunk (section-aware, falls back to fixed-size)
-> tag with metadata -> embed + store.

TODO (build day): drop real LASU documents into app/data/ — handbook,
academic calendar, SIWES guidelines, course registration guide, clearance
process, office directory, admission guide, faculty info, contacts,
student affairs info — then call ingest_directory().
"""
import os
import re
import uuid
from collections import Counter
from pypdf import PdfReader
from app.services.rag_service import add_documents

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Matches numbered section headings like "3.2 Clearance Process" or
# "Section 4: Course Registration". TODO: adjust if the real handbook
# uses a different heading style (e.g. ALL CAPS titles, Roman numerals).
HEADING_PATTERN = re.compile(
    r"^(?:(\d+(?:\.\d+)*)\s+|Section\s+\d+:?\s*)([A-Z][^\n]{3,80})$",
    re.MULTILINE,
)


# ---------- Reading ----------

def _read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_pdf_file(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ---------- Cleaning ----------

def clean_text(text: str) -> str:
    """Strip repeated headers/footers, fix line-wrap hyphens, normalize
    whitespace. PDF extraction is messy — this is the step that most
    affects answer quality, more than the chunking strategy."""
    lines = text.split("\n")

    # Drop lines that repeat on almost every "page" (rough proxy: any
    # short line appearing 3+ times, e.g. "LASU Student Handbook 2025").
    line_counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) < 60)
    boilerplate = {l for l, c in line_counts.items() if c >= 3}

    cleaned_lines = [l for l in lines if l.strip() not in boilerplate]
    text = "\n".join(cleaned_lines)

    # Drop standalone page-number lines, e.g. "12" or "Page 12"
    text = re.sub(r"(?m)^\s*(Page\s+)?\d{1,4}\s*$", "", text)

    # Rejoin hyphenated line-wraps: "regis-\ntration" -> "registration"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Collapse excess whitespace but keep paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ---------- Chunking ----------

def _split_oversized(text: str, max_chunk_size: int, overlap: int = 100) -> list[str]:
    """Split text that's too big for one chunk. Tries paragraph breaks
    first (keeps natural boundaries); if a paragraph is itself still too
    big (common with PDFs that don't preserve blank lines), falls back
    to fixed-size slicing so no chunk ever exceeds max_chunk_size."""
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []

    pieces = []
    buf = ""
    for p in paras:
        if len(p) > max_chunk_size:
            # flush whatever's buffered, then hard-slice this oversized paragraph
            if buf:
                pieces.append(buf.strip())
                buf = ""
            start = 0
            while start < len(p):
                end = start + max_chunk_size
                pieces.append(p[start:end].strip())
                start = end - overlap
        elif len(buf) + len(p) > max_chunk_size and buf:
            pieces.append(buf.strip())
            buf = p + "\n\n"
        else:
            buf += p + "\n\n"
    if buf.strip():
        pieces.append(buf.strip())

    return [p for p in pieces if p]


def chunk_by_headings(text: str, max_chunk_size: int = 1200) -> list[dict]:
    """Section-aware chunking: split on numbered/titled headings so each
    chunk stays topically whole (e.g. all of '3.2 Clearance Process'
    together, not sliced across two chunks). Long sections are further
    split, guaranteed to stay under max_chunk_size even if the PDF has
    no paragraph breaks. Returns [{"text":..., "section": ...}]."""
    matches = list(HEADING_PATTERN.finditer(text))
    if not matches:
        return []  # caller falls back to fixed-size chunking

    chunks = []
    for i, m in enumerate(matches):
        section_title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()

        if len(section_text) <= max_chunk_size:
            chunks.append({"text": section_text, "section": section_title})
        else:
            for piece in _split_oversized(section_text, max_chunk_size):
                chunks.append({"text": piece, "section": section_title})

    return chunks


def chunk_fixed_size(text: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Fallback for documents with no detectable headings (e.g. a flat
    contact list or FAQ)."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append({"text": piece, "section": None})
        start = end - overlap
    return chunks


def chunk_document(text: str) -> list[dict]:
    chunks = chunk_by_headings(text)
    return chunks if chunks else chunk_fixed_size(text)


# ---------- Ingestion ----------

async def ingest_directory(directory: str = DATA_DIR) -> dict:
    total_chunks = 0
    files_processed = []

    for fname in os.listdir(directory):
        fpath = os.path.join(directory, fname)
        if fname.lower().endswith(".pdf"):
            raw = _read_pdf_file(fpath)
        elif fname.lower().endswith(".txt"):
            raw = _read_text_file(fpath)
        else:
            continue

        cleaned = clean_text(raw)
        chunks = chunk_document(cleaned)

        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"source": fname, "section": c["section"] or "general"} for c in chunks]

        await add_documents(texts, ids, metadatas)

        total_chunks += len(chunks)
        files_processed.append({"file": fname, "chunks": len(chunks)})

    return {"files_processed": files_processed, "total_chunks": total_chunks}
