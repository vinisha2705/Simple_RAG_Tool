"""
app.py
------
The Streamlit web app. This is what you actually demo in an interview.

Run with:
    streamlit run app.py

Make sure you've run `python ingest.py` at least once first, so that
faiss_index.bin and chunks.pkl exist.
"""

import pickle
import faiss
import streamlit as st
from dotenv import load_dotenv

from rag_utils import retrieve, generate_answer

load_dotenv()  # reads OPENAI_API_KEY from a .env file if present

INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "chunks.pkl"


@st.cache_resource
def load_index_and_chunks():
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

try:
    index, chunks = load_index_and_chunks()
except Exception:
    st.error(
        "No index found. Run `python ingest.py` first to build the index "
        "from the documents in the data/ folder."
    )
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
