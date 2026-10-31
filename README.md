# RAG From Scratch — A 1-Hour Teaching Project

A minimal, fully working Retrieval-Augmented Generation (RAG) system built to be
**explained and live-coded in a 1-hour class** with students who have never
heard of RAG before.

---

## 1. What is RAG? (Explain first, ~10 min)

LLMs (like GPT, Claude, Gemini) are trained on data up to a certain date and
have no knowledge of:
- Your company's private documents
- Content created after their training cutoff
- Anything not on the public internet

**RAG (Retrieval-Augmented Generation)** solves this WITHOUT retraining the
model. Instead of relying purely on what the LLM "remembers", we:

1. Store our own documents in a searchable database
2. When the user asks a question, **retrieve** the most relevant pieces of
   text from that database
3. **Augment** the prompt sent to the LLM by inserting those pieces of text
   as context
4. Let the LLM **generate** an answer using that context

That's it — RAG is just "search, then ask the LLM to answer using the search
results." No fine-tuning, no retraining.

### Analogy for students
Think of an **open-book exam**. A closed-book exam is like asking an LLM a
question with no RAG — it can only answer from memory. RAG turns it into an
open-book exam: right before answering, the student (LLM) is handed the exact
pages of the textbook (retrieved chunks) most relevant to the question.

### The RAG Pipeline (draw this on the board)

```
                     OFFLINE / ONE-TIME "INDEXING" PHASE
   ┌──────────┐    ┌───────────┐    ┌────────────┐    ┌─────────────────┐
   │  Raw     │ -> │  Chunking │ -> │  Embedding  │ -> │  Vector Database │
   │Documents │    │(split text)│   │(text->vector)│   │  (store vectors) │
   └──────────┘    └───────────┘    └────────────┘    └─────────────────┘

                     ONLINE / PER-QUESTION "QUERY" PHASE
   ┌──────────┐    ┌────────────┐    ┌───────────────┐    ┌──────────┐
   │   User   │ -> │  Embed the │ -> │ Retrieve top-K │ -> │  Build    │
   │ Question │    │  question  │    │ similar chunks │    │  Prompt   │
   └──────────┘    └────────────┘    └───────────────┘    └──────────┘
                                                                  │
                                                                  v
                                                          ┌───────────────┐
                                                          │  LLM generates │
                                                          │  final answer  │
                                                          └───────────────┘
```

**R**etrieval → find relevant chunks
**A**ugmented → stuff those chunks into the prompt
**G**eneration → LLM writes the answer using that prompt

---

## 2. Why embeddings? Why a Vector Database instead of SQL? (~10 min)

This is usually the most confusing part for students — spend real time here.

### The core idea: embeddings capture MEANING

An embedding model converts a sentence into a list of numbers (a vector),
e.g. 384 numbers for our model. Sentences with **similar meaning** end up as
vectors that are **close together** in that 384-dimensional space, even if
they don't share any of the same words.

Example:
- "Neptune has very strong winds" 
- "The winds on that ice giant are extremely fast"

These two sentences share almost no exact words, but an embedding model
places their vectors close together because the *meaning* is similar.

### Why not just use SQL / keyword search?

Traditional SQL search (`WHERE text LIKE '%word%'`, or full-text search
indexes) can only match **exact words or word stems**. It has no concept of
meaning.

| | SQL / Keyword Search | Vector Database |
|---|---|---|
| Matches on | Exact words / substrings | Semantic meaning |
| "car" finds "automobile"? | No | Yes |
| Handles typos/synonyms/paraphrasing | No | Yes |
| Data type | Structured rows & columns | Unstructured text, images, audio as vectors |
| Query type | Exact match / range / joins | "Find the K most similar items" (nearest neighbor search) |
| Good for | Precise structured queries (`WHERE age > 25`) | Fuzzy, meaning-based search over unstructured content |

**Key point to hammer home:** SQL WHERE clauses need you to guess the exact
words in the document. Users don't ask questions using the document's exact
vocabulary — they paraphrase. A vector database searches by **meaning**, so
it doesn't matter if the user's wording is completely different from the
document's wording.

Run `3_sql_vs_vector_demo.py` in class — it proves this with real code: a
keyword search fails to find Neptune's wind data because the query
doesn't share exact words with the document, while vector search finds it
instantly because the *meaning* overlaps.

### Could you technically do this in SQL?
Yes, with extensions like `pgvector` for Postgres — those add vector
similarity search capability ON TOP of SQL. So the real distinction isn't
"SQL vs No-SQL," it's **exact/structured matching vs. nearest-neighbor
similarity search over high-dimensional vectors.** Purpose-built vector
databases (Chroma, Pinecone, Weaviate, Milvus, Qdrant) are simply optimized
specifically for this nearest-neighbor search at scale, using indexes like
HNSW that make searching millions of vectors fast — a plain SQL table
would need to compare against every single row (brute force) to do the same
similarity search, which doesn't scale.

---

## 3. Project Setup (~5 min, do this together)

### Requirements
- Python 3.9+
- (Optional) An Anthropic API key, for the final generation step. Without it,
  the project still works and shows you the retrieved chunks — you just
  won't get a natural-language final answer.

### Install
```bash
cd rag_project
pip install -r requirements.txt
```

### Project structure
```
rag_project/
├── documents/                  <- our "knowledge base" (raw text files)
│   ├── planets.txt
│   └── space_missions.txt
├── 1_build_vector_db.py        <- STEP 1: chunk + embed + store
├── 2_query_rag.py              <- STEP 2: retrieve + augment + generate
├── 3_sql_vs_vector_demo.py     <- BONUS: proves why vector DB > SQL
├── requirements.txt
└── README.md
```

---

## 4. Live-Coding Plan (~35 min hands-on)

Walk through building `1_build_vector_db.py` and `2_query_rag.py` together,
piece by piece, in this order:

1. **Load documents** (5 min) — simple file reading, nothing RAG-specific yet.
2. **Chunking** (5 min) — explain WHY we split text (embedding a whole
   document blurs its meaning into one vector; small focused chunks retrieve
   much better). Have students try changing paragraph-based chunking to
   fixed-size chunking as an exercise.
3. **Embedding + Vector DB storage** (10 min) — introduce ChromaDB,
   explain `collection.add()`, and briefly show what a vector actually looks
   like by printing one: `embedding_function(["Mars is red"])`.
4. **Query time retrieval** (10 min) — `collection.query()`, explain that
   "distance" = how far apart two vectors are (lower = more similar).
5. **Prompt augmentation + generation** (5 min) — show the final prompt
   string being built, then send it to Claude.

### Run it
```bash
# Step 1: build the vector database (run once, or whenever documents change)
python 1_build_vector_db.py

# Step 2: ask questions
python 2_query_rag.py "Which planet has the strongest winds?"
python 2_query_rag.py "Who walked on the moon?"
python 2_query_rag.py "What is the tallest volcano in the solar system?"

# Bonus: prove why vector DB beats keyword/SQL search
python 3_sql_vs_vector_demo.py
```

To enable the final LLM-generated answer, set your API key first:
```bash
export ANTHROPIC_API_KEY="your-key-here"      # Mac/Linux
set ANTHROPIC_API_KEY=your-key-here           # Windows cmd
```

---

## 5. Suggested in-class exercises (if time remains)

- Add a new `.txt` file to `documents/` with facts about a new topic, rebuild
  the DB, and query it.
- Change `n_results` from 3 to 1 and observe answer quality drop.
- Ask a question NOT covered in the documents (e.g. "What is the capital of
  France?") and observe the model correctly says it doesn't know — this
  demonstrates how RAG + a good prompt reduces hallucination.
- Print the `distance` value for a very relevant vs. very irrelevant query to
  build intuition about similarity scores.

---

## 6. Key takeaways to leave students with

- RAG = Retrieval + Augmentation + Generation. It's a design pattern, not a
  single tool.
- Embeddings turn text into vectors that capture meaning, not just words.
- Vector databases are optimized for fast "find similar meaning" search,
  which plain SQL cannot do natively.
- RAG lets you give an LLM up-to-date, private, or domain-specific knowledge
  without retraining it.