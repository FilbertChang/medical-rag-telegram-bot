from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="D:/medical-rag/.env")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""
os.environ["LANGCHAIN_PROJECT"] = "medical-rag-bot"

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Loading FAISS index...")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print("Loading Ollama LLM...")
llm = OllamaLLM(model="llama3.2")

retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = PromptTemplate.from_template("""
You are a friendly and helpful medical assistant. A user is describing their symptoms
or asking a health-related question. Use the medical knowledge from the context below
to give a clear, simple, and helpful response.

- Speak directly to the user, not about them
- Use simple language, not overly technical
- If the context is relevant, use it to guide your answer
- If the context is not relevant at all, use your general medical knowledge to help
- Never say you are reading a medical record or chart
- Always be empathetic and helpful

Context:
{context}

User message: {question}

Your response:
""")

app = FastAPI(title="Medical RAG API")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
def root():
    return {"message": "Medical RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    docs = retriever.invoke(request.question)
    context = "\n\n".join([d.page_content for d in docs])
    sources = list(set([
        d.page_content.split("\n")[0].replace("Medical Specialty:", "").strip()
        for d in docs
    ]))
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": request.question})
    return AnswerResponse(answer=answer, sources=sources)