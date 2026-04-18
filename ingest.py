import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

print("Loading MTSamples dataset...")
df = pd.read_csv("data/mtsamples.csv")
df = df.dropna(subset=["transcription"])

print(f"Loaded {len(df)} medical transcriptions")

texts = []
for _, row in df.iterrows():
    text = f"Medical Specialty: {row['medical_specialty']}\nDescription: {row['description']}\nTranscription: {row['transcription']}"
    texts.append(text)

print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents(texts)
print(f"Created {len(chunks)} chunks")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Building FAISS vector database...")
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("Done! Vector database saved to faiss_index/")