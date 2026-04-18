from dotenv import load_dotenv
import os
load_dotenv(dotenv_path="D:/medical-rag/.env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""
os.environ["LANGCHAIN_PROJECT"] = "medical-rag-bot"
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

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

def ask_question(question):
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    chain = prompt | llm
    return chain.invoke({"context": context, "question": question})

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I am a Medical RAG Bot.\n\nAsk me anything about medical conditions, symptoms, or procedures!"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await update.message.reply_text("Searching medical data, please wait...")
    try:
        answer = ask_question(question)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    print("Bot is running...")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()