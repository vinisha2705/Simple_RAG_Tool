"""
rag_utils.py
------------
Core building blocks for the RAG (Retrieval-Augmented Generation) tool.
"""

from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
_embed_model = None


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def embed_texts(texts):
    model = get_embed_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")


def retrieve(query, index, chunks, k=3):
    query_vec = embed_texts([query])
    scores, ids = index.search(query_vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({"chunk": chunks[idx], "score": float(score)})
    return results


def generate_answer(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found in the documents."
    return retrieved_chunks[0]["chunk"]