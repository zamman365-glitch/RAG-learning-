"""
STEP 2: QUERY THE RAG SYSTEM
=============================
This script does the "Retrieval + Generation" phase of RAG:

1. EMBED the user's question into a vector (same model as before)
2. RETRIEVE the most similar chunks from the vector database
3. AUGMENT a prompt to the LLM with those chunks as context
4. GENERATE a final answer using an LLM (Claude)

This is where the "R-A-G" name comes from:
    R = Retrieval  (step 2)
    A = Augmented  (step 3 - we augment/add context to the prompt)
    G = Generation (step 4)

Run with:
    python 2_query_rag.py "your question here"

If no ANTHROPIC_API_KEY is set, the script will still show you the
retrieved chunks so you can see retrieval working even without an
LLM call - useful for teaching without API keys.
"""

import sys
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions


def retrieve_relevant_chunks(query, n_results=3):
    """Embeds the query and finds the n_results most similar chunks
    in the vector database using cosine/L2 similarity search."""
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )
    embedding_function = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_collection(
        name="space_facts",
        embedding_function=embedding_function
    )

    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    # results is a dict of lists (one list per query). We only sent
    # one query, so we take index [0] of each list.
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]  # lower = more similar

    return list(zip(chunks, sources, distances))


def build_augmented_prompt(query, retrieved_chunks):
    """This is the 'Augmentation' step: we build a prompt that gives
    the LLM the retrieved context PLUS clear instructions to only
    answer using that context (this reduces hallucination)."""
    context_text = "\n\n".join(
        [f"[Source: {source}]\n{chunk}" for chunk, source, _ in retrieved_chunks]
    )

    prompt = f"""You are a helpful assistant. Answer the user's question
using ONLY the context provided below. If the answer is not contained
in the context, say "I don't have enough information to answer that."

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""
    return prompt


def generate_answer(prompt):
    """Sends the augmented prompt to Claude and returns the answer.
    Requires the ANTHROPIC_API_KEY environment variable to be set."""
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Ask a question about space: ")

    print(f"\nQUESTION: {query}\n")

    # --- RETRIEVAL ---
    retrieved_chunks = retrieve_relevant_chunks(query, n_results=3)

    print("--- RETRIEVED CHUNKS (this is the 'R' in RAG) ---")
    for i, (chunk, source, distance) in enumerate(retrieved_chunks, 1):
        print(f"\n[{i}] (source: {source}, distance: {distance:.4f})")
        print(chunk)
    print("\n" + "-" * 60)

    # --- AUGMENTATION ---
    prompt = build_augmented_prompt(query, retrieved_chunks)

    # --- GENERATION ---
    answer = generate_answer(prompt)

    print("\n--- FINAL ANSWER (this is the 'G' in RAG) ---")
    if answer:
        print(answer)
    else:
        print("(No ANTHROPIC_API_KEY found, so skipping LLM generation.)")
        print("But notice: the retrieved chunks above already contain")
        print("the answer to the question - that's RAG's retrieval")
        print("step working correctly on its own.")


if __name__ == "__main__":
    main()