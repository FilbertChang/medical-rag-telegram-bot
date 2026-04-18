# Medical RAG Chatbot (Telegram Bot)

A Retrieval-Augmented Generation (RAG) system built on real medical transcription data,
deployed as a Telegram bot. This project demonstrates how to combine vector search with
a local LLM to answer medical questions grounded in real clinical notes.

---

## Purpose

This project was built to explore how RAG systems can be applied in healthcare contexts,
specifically to assist with querying medical transcription data. It uses the MTSamples
dataset which contains real-world clinical transcriptions across various medical specialties.

Instead of relying on a general-purpose chatbot that may hallucinate, this system only
answers based on retrieved medical documents — making it more reliable for domain-specific use.

---

## Features

- Ask natural language questions about medical conditions, symptoms, and procedures
- Answers are grounded in 4,966 real medical transcriptions (MTSamples dataset)
- Honestly says "I don't have that information" when data is insufficient
- Runs fully locally — no paid API needed
- Deployed as a Telegram bot for easy access and demo
- REST API endpoint via FastAPI for programmatic access

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python |
| RAG Framework | LangChain |
| Vector Database | FAISS |
| Embedding Model | HuggingFace (all-MiniLM-L6-v2) |
| LLM | Ollama (Llama 3.2) |
| Interface | Telegram Bot (python-telegram-bot) |
| Dataset | MTSamples (Kaggle) |
| API Framework | FastAPI + Uvicorn |

---

## Project Structure
medical-rag/
├── data/
│   └── mtsamples.csv        # Medical transcriptions dataset
├── faiss_index/
│   ├── index.faiss          # Vector database
│   └── index.pkl            # Chunk metadata
├── ingest.py                # Loads data and builds vector database
├── bot.py                   # Telegram bot and RAG logic
├── api.py                   # FastAPI REST API endpoint
└── README.md
---

## How to Run

### 1. Install dependencies
pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers pandas python-telegram-bot

### 2. Install Ollama and pull Llama 3.2
Download Ollama from https://ollama.com then run:
ollama pull llama3.2

### 3. Download the dataset
Download MTSamples from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
and place `mtsamples.csv` inside the `data/` folder.

### 4. Build the vector database
python ingest.py

### 5. Run the Telegram bot
Add your Telegram bot token to `bot.py` then run:
python bot.py

---

## How It Works
User sends question on Telegram
→ Question is converted to a vector (embedding)
→ FAISS searches for the most relevant medical chunks
→ Top 3 chunks are passed to Llama 3.2 as context
→ Llama 3.2 generates a friendly, empathetic answer guided by those chunks
→ If context is relevant, it uses it — otherwise falls back to general medical knowledge
→ Bot sends the answer back to the user

---

## Dataset

**MTSamples** — Medical Transcription Samples  
Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions  
4,966 medical transcriptions across specialties like Surgery, Cardiology, Neurology, and more.

---

## API Usage

Start the API server: python -m uvicorn api:app --reload
Then send a POST request to `/ask`:
POST http://127.0.0.1:8000/ask
Content-Type: application/json
{
"question": "What are the symptoms of diabetes?"
}
Response:
{
"answer": "...",
"sources": ["..."]
}
Interactive API docs available at: http://127.0.0.1:8000/docs

---

## Author

Filbert  
Fresh Graduate | Aspiring AI Engineer