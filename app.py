"""
app.py
------
The Streamlit web app. This is what you actually demo in an interview.

Run locally with:
    streamlit run app.py

Locally, you can run `python ingest.py` once beforehand to build the index.
On a fresh deploy (e.g. Streamlit Community Cloud), faiss_index.bin and
chunks.pkl won't exist yet since they're .gitignore'd - so this file builds
the index automatically on first load if it's missing (see
build_index_if_missing() below), then caches it so it isn't rebuilt on
every rerun.
"""

import os
import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv

from rag_utils import retrieve, generate_answer, chunk_text, embed_texts
from ingest import load_documents, DATA_DIR

load_dotenv()  # reads OPENAI_API_KEY from a .env file if present

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


def build_index_if_missing():
    """If the index files don't exist yet (e.g. first run on a fresh
    deploy), build them here using the same logic as ingest.py."""
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        return

    documents = load_documents(DATA_DIR)
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc["text"]))

    embeddings = embed_texts(all_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(all_chunks, f)


@st.cache_resource
def load_index_and_chunks():
    build_index_if_missing()
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks


st.set_page_config(page_title="Simple RAG Tool", page_icon="🔎")
st.title("🔎 Simple RAG Q&A Tool")
st.caption(
    "Ask a question about the documents in the data/ folder "
    "(currently: ETL basics, SQL basics, Cloud basics)."
)

with st.spinner("Preparing index (first load only)..."):
    try:
        index, chunks = load_index_and_chunks()
    except Exception as e:
        st.error(f"Failed to build or load the index: {e}")
        st.stop()

query = st.text_input("Your question:", placeholder="e.g. What is a JOIN in SQL?")
top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=5, value=3)

if st.button("Ask") and query:
    with st.spinner("Retrieving relevant chunks..."):
        results = retrieve(query, index, chunks, k=top_k)

    with st.spinner("Generating answer..."):
        answer = generate_answer(query, results)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Show retrieved chunks (what the model actually saw)"):
        for i, r in enumerate(results, start=1):
            st.markdown(f"**Chunk {i}** (similarity score: {r['score']:.3f})")
            st.write(r["chunk"])
            st.divider()