from fastapi import APIRouter
from app.services.ingest_service import ingest_directory

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("/")
def ingest():
    """
    Re-ingests everything in app/data/ into the vector store.
    Call this once after dropping the real LASU documents into app/data/
    (and again any time you add/update a document).
    """
    result = ingest_directory()
    return result
