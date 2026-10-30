"""
BONUS: SQL KEYWORD SEARCH vs VECTOR SEARCH
============================================
This script proves - with actual code, not just theory - why RAG
systems use vector databases instead of traditional SQL/keyword
search.

We ask the SAME question two ways:
  1. Using a SQL-style keyword search (LIKE '%word%')
  2. Using our vector database (meaning-based search)

Try the question: "Which planet spins on its side?"
Notice: the documents/planets.txt describes Uranus's tilt but never
uses the word "spins" or "side" - it says "rotates on its side" is
close, but try: "Which world has crazy fast winds?" (Neptune's
document says "extremely strong winds", never the word "crazy" or
"fast" together like that).

Run with:
    python 3_sql_vs_vector_demo.py
"""

import sqlite3
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def keyword_search_demo(query, chunks):
    """Simulates what a SQL 'WHERE text LIKE %keyword%' search does:
    it can ONLY find chunks containing the exact words in the query."""
    query_words = query.lower().replace("?", "").split()
    matches = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(word in chunk_lower for word in query_words if len(word) > 3):
            matches.append(chunk)
    return matches


def get_all_chunks():
    chunks = []
    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            with open(os.path.join("documents", filename), "r", encoding="utf-8") as f:
                text = f.read()
            chunks.extend([p.strip() for p in text.split("\n\n") if p.strip()])
    return chunks


def vector_search_demo(query):
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection("space_facts", embedding_function=embedding_function)
    results = collection.query(query_texts=[query], n_results=2)
    return results["documents"][0]


def main():
    query = "Which world has unusually powerful winds?"
    print(f"QUESTION: {query}\n")

    chunks = get_all_chunks()

    print("=== SQL-STYLE KEYWORD SEARCH (WHERE text LIKE '%word%') ===")
    keyword_matches = keyword_search_demo(query, chunks)
    if keyword_matches:
        for m in keyword_matches:
            print("- " + m[:150] + "...")
    else:
        print("No matches found! The word 'unusually', 'powerful', and")
        print("'winds' don't all literally appear together in any chunk")
        print("the way SQL LIKE search needs them to.")

    print("\n=== VECTOR SEARCH (meaning-based, via ChromaDB) ===")
    try:
        vector_matches = vector_search_demo(query)
        for m in vector_matches:
            print("- " + m[:150] + "...")
        print("\nNotice: vector search found the Neptune chunk about")
        print("'extremely strong winds' even though the exact words")
        print("'unusually' and 'powerful' never appear in the text.")
        print("It matched based on MEANING, not exact keywords.")
    except Exception as e:
        print("Run 1_build_vector_db.py first! Error:", e)


if __name__ == "__main__":
    main()