import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import your existing routers
from app.routers import chat, letters 

app = FastAPI(title="LASU Compass AI")

# Configure CORS so the frontend can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory to serve the LASU PDFs
if os.path.exists("app/data"):
    app.mount("/documents", StaticFiles(directory="app/data"), name="documents")

# Include your endpoint routers
app.include_router(chat.router)
app.include_router(letters.router)

@app.get("/")
def root():
    return {"status": "LASU Compass AI Backend is running"}