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
- **Dynamic knowledge base** — send PDF, TXT, DOCX, CSV, or XLSX files directly to the bot to expand its knowledge
- **Smart ingestion** — automatically adds new chunks, updates outdated ones, and skips duplicates
- **Clinical note analyzer** — extract diagnosis, symptoms, medications, and procedures from any clinical note
- **Source tracking** — every answer shows which document and medical specialty it came from
- **Section-aware chunking** — clinical notes are split by section headers (Assessment, Plan, Medications, etc.) for more precise retrieval
- **Docker support** — fully containerized with Docker and docker-compose for easy deployment

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
| Observability | LangSmith |

---

## Project Structure
```
medical-rag/
├── data/
│   └── mtsamples.csv        # Medical transcriptions dataset
├── faiss_index/
│   ├── index.faiss          # Vector database
│   └── index.pkl            # Chunk metadata
├── ingest.py                # Loads dataset and builds initial vector database
├── ingest_file.py           # Handles dynamic file ingestion (add/update/skip logic)
├── bot.py                   # Telegram bot and RAG logic
├── api.py                   # FastAPI REST API endpoint
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install langchain langchain-community langchain-ollama langchain-huggingface langchain-text-splitters faiss-cpu sentence-transformers pandas python-telegram-bot pypdf python-docx openpyxl fastapi uvicorn
```

### 2. Install Ollama and pull Llama 3.2
Download Ollama from https://ollama.com then run:
```bash
ollama pull llama3.2
```

### 3. Download the dataset
Download MTSamples from https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions
and place `mtsamples.csv` inside the `data/` folder.

### 4. Set up environment variables

Create a `.env` file in the project root:
TELEGRAM_TOKEN=your_telegram_bot_token
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=medical-rag-bot

### 5. Build the vector database
```bash
python ingest.py
```

### 6. Run the Telegram bot
```bash
python bot.py
```

---

## How It Works

### Question Answering
User sends question on Telegram
→ Question is converted to a vector (embedding)
→ FAISS searches for the most relevant medical chunks
→ Top 3 chunks are passed to Llama 3.2 as context
→ Llama 3.2 generates a friendly, empathetic answer
→ Bot sends the answer back to the user with cited sources (file name + medical specialty)

### Dynamic File Ingestion
User sends a file (PDF, TXT, DOCX, CSV, XLSX) to the bot
→ File is parsed and split into chunks
→ Each chunk is compared against existing knowledge base
→ If no similar chunk exists → ADD
→ If similar chunk exists and new one is longer → UPDATE
→ If similar chunk exists and new one is same or shorter → SKIP
→ FAISS index is saved with updates
→ Bot reports how many chunks were added, updated, or skipped

---

### Clinical Note Analyzer
```
User sends /analyze followed by a clinical note
→ Llama 3.2 extracts structured information from the note
→ Bot replies with:
   📋 Diagnosis
   🤒 Symptoms
   💊 Medications
   🏥 Procedures
```

## Bot Commands
| Command | Description |
|---|---|
| `/start` | Welcome message and usage instructions |
| `/analyze <note>` | Extract diagnosis, symptoms, medications, and procedures from a clinical note |

## Dataset
**MTSamples** — Medical Transcription Samples  
Source: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions  
4,966 medical transcriptions across specialties like Surgery, Cardiology, Neurology, and more.

---

## API Usage
Start the API server:
```bash
python -m uvicorn api:app --reload
```

Then send a POST request to `/ask`:

POST http://127.0.0.1:8000/ask
Content-Type: application/json
{
"question": "What are the symptoms of diabetes?"
}
Response:
```json
{
  "answer": "...",
  "sources": ["..."]
}
```

Interactive API docs available at: http://127.0.0.1:8000/docs

---

## Docker Deployment
Build and run with docker-compose:
```bash
docker-compose up --build
```
This starts two containers:
- `medical-rag-bot` — the Telegram bot
- `medical-rag-api` — the FastAPI REST API on port 8000

> Note: Ollama must be running on your host machine before starting the containers.

## Author
Filbert  
Fresh Graduate | Aspiring AI Engineer
