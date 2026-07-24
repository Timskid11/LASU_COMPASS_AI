from fastapi import APIRouter
from app.services.ingest_service import ingest_directory
from app.services import vector_store

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/")
async def ingest():
    """
    Re-ingests everything in app/data/ into the vector store.
    Call this once after dropping the real LASU documents into app/data/
    (and again any time you add/update a document — this clears the old
    store first so re-running doesn't create duplicates).
    """
    vector_store.clear()
    result = await ingest_directory()
    return result
