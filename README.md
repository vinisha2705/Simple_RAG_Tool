# Simple RAG Q&A Tool

A small Retrieval-Augmented Generation (RAG) project: ask questions in plain
English and get answers grounded in a folder of text documents.

Works with **no API key** (extractive fallback) and gets better answers if
you add a free/paid OpenAI key later. Everything runs on your laptop.

---

## How it works (the whole idea in 4 steps)

1. **Chunk** — Long documents are split into small overlapping pieces of text.
2. **Embed** — Each chunk is converted into a vector (a list of numbers) that
   captures its meaning, using a small local model (`all-MiniLM-L6-v2`).
3. **Retrieve** — When you ask a question, it's embedded the same way, and
   FAISS finds the chunks whose vectors are most similar (closest in meaning).
4. **Generate** — Those chunks + your question are sent to an LLM, which
   writes an answer using only that retrieved context (or, with no API key,
   the tool just shows you the best-matching chunk directly).

```
data/*.txt --> chunk_text() --> embed_texts() --> FAISS index (ingest.py)
                                                        |
user question --> embed_texts() --> retrieve() ---------
                                          |
                                   top-k chunks --> generate_answer() --> answer
                                                                    (app.py)
```

---

## Step-by-step setup

### 1. Install Python dependencies
```bash
cd rag_tool
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Add an OpenAI API key for generated answers
```bash
cp .env.example .env
# then open .env and paste your key after OPENAI_API_KEY=
```
If you skip this step, the app still works — it will just show you the raw
retrieved passage instead of a generated sentence.

### 3. Build the search index
```bash
python ingest.py
```
This reads everything in `data/`, chunks it, embeds it, and saves two files:
`faiss_index.bin` (the vector index) and `chunks.pkl` (the original text).

### 4. Run the app
```bash
streamlit run app.py
```
A browser tab opens. Type a question like *"What is a JOIN in SQL?"* or
*"What is AWS Lambda used for?"* and hit Ask.

### 5. Try it on your own documents
Delete the sample `.txt` files in `data/`, drop in your own text files
(notes, a company's FAQ, your own study material), and re-run `python
ingest.py` to rebuild the index.

---

## Project structure
```
rag_tool/
├── data/                 # your source documents (.txt)
│   ├── etl_basics.txt
│   ├── sql_basics.txt
│   └── cloud_basics.txt
├── rag_utils.py          # chunking, embedding, retrieval, generation logic
├── ingest.py             # one-time script: builds the FAISS index
├── app.py                # Streamlit UI
├── requirements.txt
└── .env.example
```

---

## How to explain this in an interview

Keep it to these 4 sentences, then let them ask follow-ups:

> "I built a small RAG tool. It takes a folder of text documents, splits
> them into chunks, and converts each chunk into an embedding vector using
> a local sentence-transformers model. When a user asks a question, I embed
> the question the same way and use FAISS to find the most similar chunks
> by cosine similarity. Those chunks get passed to an LLM along with the
> question so the answer is grounded in the actual documents, not just the
> model's memory — and I added a fallback mode that shows the raw retrieved
> chunk if no API key is configured, so the project runs for free end-to-end."

**Questions you should be ready for, and how to answer them simply:**

- **"Why chunk the text instead of embedding the whole document?"**
  Embedding models have a limited context size and lose precision on long
  text; smaller chunks let you retrieve just the relevant part of a
  document instead of the whole thing.

- **"Why overlap the chunks?"**
  So an idea that falls near a chunk boundary isn't cut off entirely —
  it still fully appears in at least one chunk.

- **"What is FAISS?"**
  A library from Meta for fast similarity search over large collections
  of vectors — instead of comparing your question to every chunk one by
  one, it finds the closest matches efficiently.

- **"Why normalize the embeddings?"**
  So a plain dot product between two vectors is equivalent to cosine
  similarity — it keeps the similarity score based on direction (meaning)
  rather than vector length.

- **"What would you improve if you had more time?"**
  Good honest answer: add re-ranking of retrieved chunks, support PDFs
  instead of just .txt, and add citations showing which source file each
  answer came from.

- **"What's the difference between this and just asking ChatGPT directly?"**
  Without retrieval, the model only knows what it was trained on and can
  hallucinate. RAG grounds the answer in your specific documents, so it can
  answer questions about content the model has never seen before, and you
  can point to exactly which passage it used.

---

## Notes
- The embedding model downloads (~90MB) the first time you run `ingest.py`.
- `faiss-cpu` is used since GPU isn't needed for a dataset this small.
- Swap `gpt-4o-mini` in `rag_utils.py` for any other OpenAI model, or adapt
  `generate_answer()` to call a different provider if you prefer.
