"""
STEP 1: BUILD THE VECTOR DATABASE
==================================
This script does three things (the "Indexing" phase of RAG):

1. LOAD  -> read our .txt documents from the documents/ folder
2. CHUNK -> split each document into small overlapping pieces of text
3. EMBED -> convert each chunk into a vector (list of numbers) that
            captures its MEANING, and store it in a vector database
            (ChromaDB) so we can search it later by meaning, not by
            exact keywords.

Run this once before running 2_query_rag.py
    python 1_build_vector_db.py
"""

import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# -----------------------------------------------------------------
# 1. LOAD DOCUMENTS
# -----------------------------------------------------------------
DOCUMENTS_FOLDER = "documents"

def load_documents(folder_path):
    """Reads every .txt file in the folder and returns a list of
    (filename, full_text) tuples."""
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append((filename, text))
    return documents


# -----------------------------------------------------------------
# 2. CHUNK DOCUMENTS
# -----------------------------------------------------------------
# WHY CHUNK? LLMs and embedding models work best on small, focused
# pieces of text. If we embed an entire document as ONE vector, the
# vector becomes a "blurry average" of many different ideas mixed
# together, and retrieval quality drops badly.
#
# Here we chunk by paragraph (our sample docs already have one idea
# per paragraph), which is the simplest possible chunking strategy.
# In real projects you would use a fixed word/token size with some
# overlap - we mention this to students but keep the code simple.

def chunk_text(text, filename):
    """Splits text into chunks (here: paragraphs) and returns a list
    of dicts with the chunk text + metadata about where it came from."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    for i, para in enumerate(paragraphs):
        chunks.append({
            "id": f"{filename}_chunk_{i}",
            "text": para,
            "source": filename
        })
    return chunks


# -----------------------------------------------------------------
# 3. EMBED + STORE IN VECTOR DATABASE
# -----------------------------------------------------------------
def build_vector_db():
    # ChromaDB will save its data to disk in this folder, so we don't
    # need to rebuild the database every time we run a query.
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    # This is the embedding model. It's a small, free, local model
    # (all-MiniLM-L6-v2, running via ONNX) that turns any sentence into
    # a 384-dimensional vector. Similar meaning -> similar vector
    # (close together in space). No internet/API key needed after the
    # first run (the model file is downloaded once and cached).
    embedding_function = embedding_functions.DefaultEmbeddingFunction()

    # Delete old collection if it exists, so re-running this script
    # always gives a fresh, correct database.
    try:
        client.delete_collection("space_facts")
    except Exception:
        pass

    collection = client.create_collection(
        name="space_facts",
        embedding_function=embedding_function
    )

    # Load and chunk all documents
    documents = load_documents(DOCUMENTS_FOLDER)
    all_chunks = []
    for filename, text in documents:
        all_chunks.extend(chunk_text(text, filename))

    print(f"Loaded {len(documents)} documents -> split into {len(all_chunks)} chunks")

    # Add every chunk to the vector database.
    # Chroma automatically calls the embedding_function on each chunk's
    # text and stores the resulting vector alongside the original text.
    collection.add(
        ids=[c["id"] for c in all_chunks],
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks]
    )

    print("Vector database built successfully at ./chroma_db")
    print(f"Total chunks stored: {collection.count()}")


if __name__ == "__main__":
    build_vector_db()