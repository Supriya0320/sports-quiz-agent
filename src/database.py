"""
database.py
------------
Owns all interaction with ChromaDB (our local vector database).
Responsibilities:
  1. Create/open a persistent ChromaDB client (data survives between runs).
  2. Populate the "sports_history" collection from data/sports_facts.json.
  3. Answer similarity-search queries, optionally filtered by sport.

No other module should talk to ChromaDB directly -- this keeps the vector
store logic isolated and easy to swap out later (e.g., for Pinecone/Weaviate).
"""

import os
import json

# --- Windows/Linux sqlite3 compatibility shim -------------------------------
# ChromaDB needs a modern sqlite3. If the system's built-in sqlite3 is too
# old, `pip install pysqlite3-binary` and uncomment the two lines below.
#
# __import__("pysqlite3")
# import sys
# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# -----------------------------------------------------------------------------

import chromadb
from chromadb.utils import embedding_functions

from src.config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, SPORTS_FACTS_PATH

_embedding_fn = embedding_functions.DefaultEmbeddingFunction()


def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client saving to disk."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_collection(client):
    """Gets (or lazily creates) the shared sports_history collection."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=_embedding_fn,
    )


def setup_and_populate_db(json_file_path: str = SPORTS_FACTS_PATH):
    """
    Reads the offline JSON facts, creates a collection, and populates it.
    Safe to call on every app startup -- it skips re-inserting facts if the
    collection is already populated.
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    if collection.count() > 0:
        print(f"[database] Already populated with {collection.count()} facts.")
        return collection

    if not os.path.exists(json_file_path):
        print(f"[database] ERROR: fact data file not found at {json_file_path}")
        return collection

    with open(json_file_path, "r", encoding="utf-8") as f:
        facts_list = json.load(f)

    documents, metadata_list, ids = [], [], []
    for idx, item in enumerate(facts_list):
        documents.append(item["fact"])
        # Storing sport as metadata lets us filter queries by sport later.
        metadata_list.append({"sport": item["sport"]})
        ids.append(f"fact_{idx}")

    collection.add(documents=documents, metadatas=metadata_list, ids=ids)
    print(f"[database] Vectorized and stored {len(documents)} facts.")
    return collection


def query_historic_facts(sport: str, query_text: str, n_results: int = 3):
    """
    Queries ChromaDB for historic documents relating to a sport.
    Filters results to only the selected sport category via metadata.

    Returns a list of matching document strings (may be empty).
    """
    client = get_chroma_client()
    collection = _get_collection(client)

    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where={"sport": sport},
    )
    return results.get("documents", [[]])[0]


def get_collection_stats():
    """Returns simple stats used by the UI (e.g., total facts stored)."""
    client = get_chroma_client()
    collection = _get_collection(client)
    return {"total_facts": collection.count()}
