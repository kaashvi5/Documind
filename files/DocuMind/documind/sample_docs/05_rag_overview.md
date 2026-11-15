# Retrieval-Augmented Generation (RAG) Overview

Retrieval-Augmented Generation, or RAG, combines a search system with a language model. Instead of relying only on what the model memorized during training, a RAG system first retrieves relevant passages from a document collection and then asks the model to answer using those passages.

## Why use RAG

Language models can produce fluent but wrong statements, which are called hallucinations. RAG reduces hallucinations because the answer is grounded in retrieved text, and the system can show the sources it used. RAG also lets a model answer questions about private or recent documents without retraining it.

## Main components

**Chunking.** Documents are split into short passages called chunks. Neighbouring chunks usually overlap by a small amount so that a fact at a boundary is not lost.

**Embeddings.** An embedding model converts each chunk into a vector of numbers. Passages with similar meaning have vectors that are close together. The same model converts the user's question into a vector.

**Vector database.** A vector database such as FAISS stores the chunk vectors and finds the nearest ones to a query vector quickly, usually with cosine similarity.

**Generator.** A language model receives the question and the retrieved chunks and writes the answer. A good prompt tells the model to use only the supplied context and to say when the answer is missing.

## Limitations

If retrieval misses the right passage, the generator cannot answer correctly. Small models may misread long contexts, and scanned documents need optical character recognition before they can be indexed.
