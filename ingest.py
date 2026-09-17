"""
ingest.py
---------
Run this ONCE (or whenever you change the documents in data/) to build
the search index.

What it does, step by step:
  1. Reads every .txt file in the data/ folder
  2. Splits each document into overlapping chunks
  3. Converts every chunk into an embedding vector
  4. Builds a FAISS index (a fast similarity-search structure) from those vectors
  5. Saves the FAISS index + the raw chunk text to disk, so app.py can load
     them instantly without re-processing the documents every time

Usage:
    python ingest.py
"""

import os
import pickle
import faiss

from rag_utils import chunk_text, embed_texts

DATA_DIR = "data"
INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


def load_documents(data_dir):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                documents.append({"source": filename, "text": f.read()})
    return documents


def main():
    print(f"Loading documents from '{DATA_DIR}/'...")
    documents = load_documents(DATA_DIR)
    print(f"Found {len(documents)} document(s).")

    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for piece in pieces:
            all_chunks.append(piece)
    print(f"Split into {len(all_chunks)} chunk(s).")

    print("Generating embeddings (this downloads the model the first time)...")
    embeddings = embed_texts(all_chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # IP = inner product = cosine similarity
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"Done. Saved index to '{INDEX_PATH}' and chunks to '{CHUNKS_PATH}'.")


if __name__ == "__main__":
    main()
