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
from ingest_file import ingest_file

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
    answer = chain.invoke({"context": context, "question": question})

    sources = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        specialty = doc.metadata.get("specialty", "Unknown")
        sources.append(f"{i}. {source} — {specialty}")

    source_text = "\n".join(sources)
    return f"{answer}\n\n📚 Sources:\n{source_text}"

async def handle_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE):
    note = " ".join(context.args)
    if not note:
        await update.message.reply_text(
            "Please provide a clinical note after the command.\n\n"
            "Example:\n/analyze Patient is a 45-year-old male with chest pain and shortness of breath. "
            "Diagnosed with acute myocardial infarction. Prescribed aspirin and nitroglycerin."
        )
        return

    await update.message.reply_text("Analyzing clinical note, please wait...")

    analyze_prompt = PromptTemplate.from_template("""
You are a clinical note analyzer. Extract the following information from the clinical note below.
Reply in exactly this format, nothing else:

📋 Diagnosis: <diagnosis or "Not mentioned">
🤒 Symptoms: <comma-separated symptoms or "Not mentioned">
💊 Medications: <comma-separated medications or "Not mentioned">
🏥 Procedure: <comma-separated procedures or "Not mentioned">

Clinical note:
{note}
""")

    try:
        chain = analyze_prompt | llm
        result = chain.invoke({"note": note})
        await update.message.reply_text(result)
    except Exception as e:
        await update.message.reply_text(f"Error analyzing note: {str(e)}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Hi! I am a Medical RAG Bot.\n\nAsk me anything about medical conditions, symptoms, or procedures!\n\nYou can also send me a PDF, TXT, DOCX, CSV, or XLSX file to add to my knowledge base."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text
    await update.message.reply_text("Searching medical data, please wait...")
    try:
        answer = ask_question(question)
        await update.message.reply_text(answer)
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Get the file from the message
    msg = update.message
    if msg.document:
        tg_file = await msg.document.get_file()
        file_name = msg.document.file_name
    else:
        await msg.reply_text("Please send a file as a document, not as a photo or media.")
        return

    # Check supported formats
    ext = os.path.splitext(file_name)[1].lower()
    if ext not in [".pdf", ".txt", ".docx", ".csv", ".xls", ".xlsx"]:
        await msg.reply_text(f"Unsupported file type: {ext}\n\nSupported: PDF, TXT, DOCX, CSV, XLS, XLSX")
        return

    await msg.reply_text(f"Received {file_name}! Processing, please wait...")

    # Download the file
    download_path = f"D:/medical-rag/uploads/{file_name}"
    os.makedirs("D:/medical-rag/uploads", exist_ok=True)
    await tg_file.download_to_drive(download_path)

    # Run ingestion
    try:
        result = ingest_file(download_path)
        await msg.reply_text(
            f"Done! Knowledge base updated:\n"
            f"✅ Added: {result['added']} chunks\n"
            f"🔄 Updated: {result['updated']} chunks\n"
            f"⏭️ Skipped: {result['skipped']} chunks"
        )
    except Exception as e:
        await msg.reply_text(f"Error processing file: {str(e)}")
    finally:
        # Clean up downloaded file
        if os.path.exists(download_path):
            os.remove(download_path)

if __name__ == "__main__":
    TOKEN = os.getenv("TELEGRAM_TOKEN")
    print("Bot is running...")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("analyze", handle_analyze))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()