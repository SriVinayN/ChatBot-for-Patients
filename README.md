# 🩺 AI-Powered Healthcare Assistant using FAISS + OpenAI

This project is an intelligent, context-aware chatbot designed to answer healthcare-related questions by referencing uploaded PDFs, text files, or URLs. It uses semantic search (via FAISS), context retrieval, and OpenAI's GPT models to provide accurate and helpful responses.

---

## 🚀 Features

- ✅ Extracts text from **PDFs, text files, or URLs** (webpages or online PDFs)
- ✅ Splits documents into manageable chunks for semantic indexing
- ✅ Generates vector embeddings using **SentenceTransformers**
- ✅ Builds a **FAISS index** for fast similarity-based retrieval
- ✅ Uses **OpenAI's GPT (GPT-3.5/4)** to generate contextual responses
- ✅ Maintains **conversation history** to support follow-up questions
- ✅ **Auto-corrects user input** for better understanding

---

## 🧱 Architecture Overview

```text
Local/Online Docs ➝ Chunking ➝ Embedding ➝ FAISS Index
                               ⬇
                           User Query
                               ⬇
                   Retrieve Relevant Chunks
                               ⬇
              Build Prompt + OpenAI Completion
                               ⬇
                          Response Output
