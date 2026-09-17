"""
rag_utils.py
------------
Core building blocks for the RAG (Retrieval-Augmented Generation) tool.

This file has 4 jobs, kept deliberately separate so each one is easy to
explain on its own:
  1. chunk_text()       -> break long documents into small overlapping pieces
  2. embed_texts()      -> turn text chunks into numeric vectors
  3. retrieve()         -> find the chunks most similar to a user's question
  4. generate_answer()  -> turn those chunks + the question into a final answer
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. Load the embedding model once, at import time, so we don't reload it
#    on every function call. all-MiniLM-L6-v2 is small, fast, free, and runs
#    fully on your laptop (no API key or internet call needed at query time).
# ---------------------------------------------------------------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


# ---------------------------------------------------------------------------
# 2. Chunking: LLMs and embedding models work better on small pieces of text
#    rather than a whole document at once. We split on words and overlap
#    chunks slightly so we don't cut an idea awkwardly in half at a boundary.
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=120, overlap=20):
    """
    Split `text` into overlapping chunks of roughly `chunk_size` words.

    Example: chunk_size=120, overlap=20 means each chunk shares its last
    20 words with the start of the next chunk, so an idea that spans a
    chunk boundary isn't lost entirely from either chunk.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------------------------------------------------------------------------
# 3. Embedding: convert text -> a vector of numbers that captures meaning.
#    Texts with similar meaning end up with vectors that point in a similar
#    direction, which is what lets us do "semantic" search instead of just
#    keyword matching.
# ---------------------------------------------------------------------------
def embed_texts(texts):
    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


# ---------------------------------------------------------------------------
# 4. Retrieval: given a question, find the top-k most similar chunks using
#    cosine similarity (since embeddings are normalized, a dot product IS
#    cosine similarity - this is why we used normalize_embeddings=True above).
# ---------------------------------------------------------------------------
def retrieve(query, index, chunks, k=3):
    """
    query   : the user's question (string)
    index   : a FAISS index built from all chunk embeddings
    chunks  : the original list of text chunks, in the same order they were
              added to the index (so index position -> chunk text)
    k       : how many chunks to return
    """
    query_vec = embed_texts([query])
    scores, ids = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "score": float(score)})
    return results


# ---------------------------------------------------------------------------
# 5. Generation: combine the retrieved chunks with the question and produce
#    a final answer.
#
#    If an OPENAI_API_KEY is set, we call the OpenAI API to write a proper
#    natural-language answer grounded in the retrieved chunks (this is the
#    "G" - Generation - in RAG).
#
#    If no key is set, we fall back to simply returning the most relevant
#    chunk(s) as the "answer" (extractive mode) so the whole project still
#    runs end-to-end for free, with no API key required.
# ---------------------------------------------------------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(f"- {r['chunk']}" for r in retrieved_chunks)
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer isn't in the context, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    # Fallback: no API key -> return the top chunk directly (extractive answer)
    if not retrieved_chunks:
        return "No relevant information found in the documents."
    return (
        "[No OPENAI_API_KEY set - showing the most relevant passage instead "
        "of a generated answer]\n\n" + retrieved_chunks[0]["chunk"]
    )
