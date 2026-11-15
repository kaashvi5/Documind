<div align="center">

# DocuMind

### AI-powered RAG document assistant: ask your documents, get answers with sources

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/documind/blob/main/DocuMind_Colab.ipynb)
![Python](https://img.shields.io/badge/python-3.10%20to%203.12-3F6BF4)
![Tests](https://img.shields.io/badge/tests-89%20passing-0F7B55)
![License](https://img.shields.io/badge/cost-free%20%7C%20no%20API%20keys-7B5CF0)

</div>

DocuMind lets you upload **PDF, DOCX, TXT and Markdown** files, ask questions in plain language, and get answers
generated **only from your own documents**. Every answer shows the source file, page number, supporting passage
and similarity score, so you can check it. If the documents do not contain the answer, DocuMind says so instead of
guessing. It runs for free on Google Colab or on your own computer, with no paid API and no API keys.

![DocuMind architecture](docs/architecture.svg)

> **Screenshots:** add your own after the first run, then replace these lines with
> `![Chat](docs/screenshot_chat.png)`, `![Library](docs/screenshot_library.png)` and `![Settings](docs/screenshot_settings.png)`.

---

## Table of contents

1. [Features](#features)
2. [Quick start](#quick-start)
3. [Run on Google Colab](#run-on-google-colab)
4. [Using the app](#using-the-app)
5. [How it works](#how-it-works)
6. [Project structure](#project-structure)
7. [Configuration](#configuration)
8. [Testing](#testing)
9. [Reliability and security](#reliability-and-security)
10. [Troubleshooting](#troubleshooting)
11. [Limitations](#limitations)
12. [Future improvements](#future-improvements)

---

## Features

| Area | What you get |
|---|---|
| Documents | Multi-file upload of PDF, DOCX, TXT and Markdown; validation of type and size (15 MB); duplicate detection; per-document delete; clear repository |
| Processing | Text extraction, cleaning, sentence-aware overlapping chunks, page-level metadata, optional OCR for scanned PDFs |
| Retrieval | MiniLM embeddings, FAISS cosine search, similarity threshold, keyword reranking, configurable Top-K, document filter |
| Answers | Local FLAN-T5 model, prompt that restricts answers to the retrieved context, honest "not found" response, follow-up questions |
| Citations | Filename, page, highlighted passage, similarity score and source number in expandable cards; low-confidence warning |
| Interface | Dashboard with statistics, chat workspace, suggested questions, Settings tab where every control works |
| Robustness | Handles empty, corrupted, encrypted and oversized files; GPU out-of-memory recovery; fallback if a model fails to load |
| Quality | 89 automated tests, modular code, self-contained Colab notebook |

## Quick start

**Requirements:** Python 3.10 to 3.12 (3.12 is the tested version), about 3 GB of free disk, and an internet
connection for the first run (about 1 GB of model downloads).

```bash
git clone https://github.com/YOUR-USERNAME/documind.git
cd documind
```

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --default-timeout=300 --retries 15 -r requirements.txt
python app.py
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:7860**. Options: `python app.py --share` (public link), `--port 7861`, `--no-preload`
(load the language model on the first question), `--generation-model google/flan-t5-small`.

The first launch downloads `all-MiniLM-L6-v2` and `flan-t5-base`; later launches use the local cache.

**Try it in 30 seconds:** click **Load samples** in the sidebar, then ask
*"How much is the home-office stipend?"*

## Run on Google Colab

1. Click the **Open in Colab** badge at the top (or upload `DocuMind_Colab.ipynb` to colab.research.google.com).
2. Optional: *Runtime → Change runtime type → T4 GPU*. Without a GPU it still works, only slower.
3. Choose *Runtime → Run all*.

The notebook installs dependencies, writes every source file, generates sample documents, runs the tests, loads the
models and launches the app with a public `gradio.live` link. It needs no other files and no API key.

## Using the app

1. **Upload:** choose files in the sidebar and select **Index documents**. Each file reports *Indexed*, *Updated*,
   *Skipped (duplicate)* or *Failed* with the reason. **Load samples** indexes the bundled demo set.
2. **Ask:** type a question in the **Chat** tab or click a suggested question. Follow-ups such as
   *"What about the internet?"* are rewritten using your previous question.
3. **Verify:** expand a source to read its passage (query words highlighted), page and similarity. Filled number
   badges mark passages that support the answer.
4. **Manage:** delete a single document, or clear the repository (tick the confirmation first). Use
   **Search only in these documents** to restrict retrieval.
5. **Tune (Settings tab):**

| Setting | Effect |
|---|---|
| Chunk size / overlap | Passage length and shared text between neighbours. Applies to newly indexed files; **Re-index** applies it to all |
| Top-K | How many passages the model reads |
| Similarity threshold | Passages below this cosine similarity are ignored (default 0.25) |
| Keyword reranking | Blends keyword coverage into the ranking |
| Detailed answers | Longer, slower answers |
| OCR | Reads scanned PDFs (needs Tesseract) |
| Models | Switch FLAN-T5 Small / Base / Large and the embedding model |

Example questions with expected sources are in [`docs/example_questions.md`](docs/example_questions.md).
Sample answers are in [`docs/example_outputs.md`](docs/example_outputs.md).

## How it works

**Retrieval-Augmented Generation (RAG)** is an open-book exam for a language model: instead of answering from memory,
it first looks up relevant passages and answers from them.

```
INGESTION   Upload → Extract → Clean → Chunk → Metadata → Embeddings → FAISS
QUESTION    Question → Query embedding → Vector search → Threshold → Rerank → Context → LLM → Answer + citations
```

| Concept | Plain-language explanation |
|---|---|
| Chunk | A short passage (default 600 characters). Neighbouring chunks overlap by 100 characters so a fact on a boundary is not lost |
| Embedding | A list of 384 numbers that captures a passage's meaning. Similar meanings get similar numbers |
| Cosine similarity | How closely two embeddings point the same way: 1.0 is identical, 0 is unrelated |
| FAISS | A library that finds the nearest embeddings quickly |
| Reranking | Final score = 0.8 × similarity + 0.2 × keyword coverage |
| Grounded prompt | The model is told to answer only from the numbered passages, or reply "unanswerable" |
| Citation marking | After generation, passages whose wording matches the answer are marked as cited |

## Project structure

```
documind/
├── app.py                  Entry point: python app.py [--share]
├── config.py               Paths, limits, model lists, RAGSettings
├── requirements.txt
├── DocuMind_Colab.ipynb    Self-contained Colab notebook
├── rag/
│   ├── loaders.py          Validation, PDF/DOCX/TXT/MD extraction, OCR, cleaning
│   ├── chunking.py         Sentence-aware overlapping chunks with page numbers
│   ├── embeddings.py       Sentence-Transformers wrapper, cache, offline fallback
│   ├── vectorstore.py      FAISS + metadata registry, delete, filter, persistence
│   ├── retrieval.py        Search, threshold, keyword rerank, follow-up handling
│   ├── generator.py        FLAN-T5 prompt and token budget, OOM handling, extractive fallback
│   ├── pipeline.py         Ingestion and question-answering orchestration
│   └── textutils.py        Shared tokenising helpers
├── ui/                     components.py (theme, HTML), dashboard.py, chat.py
├── data/                   uploads/ and vector_db/ (created at runtime)
├── sample_docs/            Five sample documents (MD, DOCX, TXT, PDF)
├── scripts/                make_samples.py, generate_example_outputs.py, build_notebook.py
├── tests/                  pytest suite
└── docs/                   architecture.svg, example_questions.md, example_outputs.md
```

## Configuration

Defaults live in `config.py`: `MAX_FILE_MB` (15), `MAX_FILES_PER_UPLOAD` (25), model lists, and `RAGSettings`
(chunk size 600, overlap 100, Top-K 4, threshold 0.25). Set the `DOCUMIND_HOME` environment variable to store data
somewhere other than the project folder. No API keys or secrets are used.

## Testing

```bash
python -m pytest -q
```

89 tests run in a few seconds with no downloads. They cover extraction for all four formats (including a real OCR
test when Tesseract is installed), cleaning, chunking, embeddings (shape, normalisation, caching), FAISS indexing,
retrieval relevance, metadata preservation, empty and corrupted files, deletion, persistence across restarts,
grounded answers and honest not-found responses.

The suite uses a deterministic offline embedder and an extractive answerer so it runs anywhere. To also test the real
FLAN-T5 model (downloads it):

```bash
DOCUMIND_RUN_MODEL_TESTS=1 python -m pytest tests/test_generator.py     # Windows: set the variable first
```

## Reliability and security

- Validates extension, size, empty files, PDF and DOCX headers, password-protected and corrupted files. One bad file
  never stops the rest of a batch.
- Duplicate content is detected with SHA-256. Re-uploading a changed file with the same name replaces the old
  version, so no stale vectors remain. Deleting a document removes its vectors, metadata and stored copy.
- The index is saved atomically and reconciled on load. A corrupted index is reset with a warning. Changing the
  embedding model re-indexes stored documents.
- Uploads are copied under sanitised file names. Documents stay on the machine running the app; only the model files
  are downloaded from Hugging Face.
- GPU memory errors are handled: embeddings split their batch, generation retries with greedy decoding and finally
  moves to CPU. If the language model cannot load or crashes, answers fall back to extractive mode (copied passages)
  with a visible warning.

## Troubleshooting

| Problem | Fix |
|---|---|
| `faiss-cpu` has no matching distribution | Use Python 3.10 to 3.12 and a fresh virtual environment |
| pip fails with SSL or timeout errors | `pip install --default-timeout=300 --retries 15 -r requirements.txt`; disable VPN or proxy; retry the failing package alone |
| `requirements.txt` not found | Run commands inside the folder that contains `app.py` |
| Slow answers | Normal on CPU. Use FLAN-T5 Small in Settings, or run on a Colab GPU |
| Warnings about `HF_TOKEN` or tied weights | Harmless messages from Hugging Face; ignore them |
| Port 7860 busy | `python app.py --port 7861` |
| "Not found" for a question you expect to work | Lower the similarity threshold or raise Top-K in Settings |

## Limitations

- FLAN-T5 Base is small: good at short factual questions, weaker at long reasoning, tables, or combining many
  passages. It reads about 768 tokens of context per question.
- Scanned PDFs need OCR (Tesseract). Multi-column layouts and tables in PDFs may extract in a jumbled order.
- DOCX and TXT have no page numbers, so citations show "Page n/a".
- Search is exact (flat), suited to roughly 100,000 chunks or fewer.
- Citation marking compares wording between answer and passages, so it is a heuristic, not a proof.
- One shared repository and no login: suited to a demo or single user.

## Future improvements

Hybrid BM25 plus vector search, a cross-encoder reranker, streaming answers, per-user repositories, approximate FAISS
indexes (IVF or HNSW) for large collections, table-aware PDF parsing, and an automatic answer-quality evaluation set.

## Tech stack

Python · PyTorch · Sentence-Transformers (`all-MiniLM-L6-v2`) · FAISS · Hugging Face Transformers
(`google/flan-t5-base`) · Gradio · pypdf · python-docx · Tesseract (optional) · pytest
