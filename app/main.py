from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import chat, letters, ingest, health

app = FastAPI(
    title="LASU Campus Assistant API",
    description="Backend for GDGoC LASU 'Build with Gemma' hackathon",
    version="0.1.0",
)

# TODO (build day): lock this down to your actual frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(letters.router)
app.include_router(ingest.router)


@app.get("/")
def root():
    return {"message": "LASU Campus Assistant API", "docs": "/docs"}
