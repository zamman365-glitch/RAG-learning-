
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Folder containing text files
DOCUMENTS_FOLDER = "documents"


# Step 1: Load text documents
def load_documents(folder_path):
    documents = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(
            f"Folder '{folder_path}' not found."
        )

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append((filename, text))

    return documents


# Step 2: Split documents into chunks
def chunk_text(text, filename):
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []

    for i, paragraph in enumerate(paragraphs):
        chunks.append({
            "id": f"{filename}_chunk_{i}",
            "text": paragraph,
            "source": filename
        })

    return chunks


# Step 3: Build the ChromaDB vector database
def build_vector_db():
    client = chromadb.PersistentClient(
        path="./chroma_db",
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    # Create the embedding function
    embedding_function = (
        embedding_functions.DefaultEmbeddingFunction()
    )

    # Remove the old collection, if it exists
    try:
        client.delete_collection("space_facts")
    except Exception:
        pass

    # Create a new collection
    collection = client.create_collection(
        name="space_facts",
        embedding_function=embedding_function
    )

    # Load documents
    documents = load_documents(DOCUMENTS_FOLDER)

    # Create chunks from all documents
    all_chunks = []

    for filename, text in documents:
        chunks = chunk_text(text, filename)
        all_chunks.extend(chunks)
    print(f"Total chunks created: {len(all_chunks)}")

    # Store chunks in ChromaDB
    if all_chunks:
        collection.add(
            ids=[chunk["id"] for chunk in all_chunks],
            documents=[chunk["text"] for chunk in all_chunks],
            metadatas=[
                {"source": chunk["source"]}
                for chunk in all_chunks
            ]
        )

    print(f"Documents loaded: {len(documents)}")
    print(f"Total chunks stored: {len(all_chunks)}")
    print(f"Total chunks stored: {collection.count()}")
    print("Vector database created successfully!")

    return collection


# Step 4: Run the program
if __name__ == "__main__":
    collection = build_vector_db()
