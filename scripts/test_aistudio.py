"""
Quick standalone check that your AISTUDIO_API_KEY works — no FastAPI
needed. Also lists available Gemma models, since exact model IDs on AI
Studio can shift (confirms/fixes the AISTUDIO_MODEL TODO).

Usage:
    pip install httpx python-dotenv
    python scripts/test_aistudio.py
"""
import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("AISTUDIO_API_KEY", "")
BASE = "https://generativelanguage.googleapis.com/v1beta"


async def list_gemma_models(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(f"{BASE}/models?key={API_KEY}")
    resp.raise_for_status()
    models = resp.json().get("models", [])
    return [m["name"].replace("models/", "") for m in models if "gemma" in m["name"].lower()]


async def test_generate(client: httpx.AsyncClient, model: str) -> str:
    url = f"{BASE}/models/{model}:generateContent?key={API_KEY}"
    payload = {"contents": [{"role": "user", "parts": [{"text": "Say hello in one short sentence."}]}]}
    resp = await client.post(url, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


async def main():
    if not API_KEY:
        print("ERROR: AISTUDIO_API_KEY not set in .env")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=30) as client:
        print("Checking key + listing available Gemma models...")
        try:
            gemma_models = await list_gemma_models(client)
        except httpx.HTTPStatusError as e:
            print(f"FAILED: key or request rejected — {e.response.status_code} {e.response.text}")
            sys.exit(1)

        if not gemma_models:
            print("Key works, but no Gemma models found in the list. Check your account access.")
            sys.exit(1)

        print(f"Available Gemma models:\n  " + "\n  ".join(gemma_models))
        test_model = gemma_models[0]

        print(f"\nTesting generation with '{test_model}'...")
        try:
            reply = await test_generate(client, test_model)
            print(f"SUCCESS — Gemma replied: {reply.strip()}")
            print(f"\n=> Set AISTUDIO_MODEL={test_model} in your .env")
        except httpx.HTTPStatusError as e:
            print(f"Generation FAILED: {e.response.status_code} {e.response.text}")


if __name__ == "__main__":
    asyncio.run(main())
