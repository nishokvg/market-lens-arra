"""Create the Pinecone index required by the MarketLens LlamaIndex adapter."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "market-lens-sources")
API_KEY = os.getenv("PINECONE_API_KEY")

if not API_KEY:
    raise SystemExit("PINECONE_API_KEY is required. Add it to .env before running this script.")

client = Pinecone(api_key=API_KEY)
if client.has_index(INDEX_NAME):
    print(f"Pinecone index '{INDEX_NAME}' already exists.")
else:
    client.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"Created Pinecone index '{INDEX_NAME}'. Wait until it is ready before starting the app.")
