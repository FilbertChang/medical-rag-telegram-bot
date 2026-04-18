import re
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Clinical note section headers to split on
SECTION_HEADERS = [
    "HISTORY OF PRESENT ILLNESS",
    "PAST MEDICAL HISTORY",
    "PHYSICAL EXAMINATION",
    "REVIEW OF SYSTEMS",
    "ASSESSMENT",
    "PLAN",
    "MEDICATIONS",
    "ALLERGIES",
    "CHIEF COMPLAINT",
    "PROCEDURE",
    "DIAGNOSIS",
    "LABORATORY DATA",
    "IMPRESSION",
    "DISCHARGE INSTRUCTIONS",
]

def split_by_sections(text: str) -> list[str]:
    """Split clinical note text by section headers."""
    pattern = "|".join([re.escape(h) for h in SECTION_HEADERS])
    parts = re.split(f"({pattern})", text, flags=re.IGNORECASE)

    sections = []
    current_header = ""
    current_body = ""

    for part in parts:
        if part.strip().upper() in [h.upper() for h in SECTION_HEADERS]:
            if current_body.strip():
                sections.append(f"{current_header}\n{current_body.strip()}")
            current_header = part.strip()
            current_body = ""
        else:
            current_body += part

    if current_body.strip():
        sections.append(f"{current_header}\n{current_body.strip()}")

    return sections if sections else [text]


print("Loading MTSamples dataset...")
df = pd.read_csv("data/mtsamples.csv")
df = df.dropna(subset=["transcription"])
print(f"Loaded {len(df)} medical transcriptions")

print("Splitting into chunks by clinical sections...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

documents = []
for _, row in df.iterrows():
    specialty = str(row["medical_specialty"]).strip()
    description = str(row["description"]).strip()
    transcription = str(row["transcription"])

    # Split transcription by clinical section headers
    sections = split_by_sections(transcription)

    for section in sections:
        section_text = f"Medical Specialty: {specialty}\nDescription: {description}\n{section}"
        # Further split if section is still too long
        sub_chunks = splitter.create_documents(
            [section_text],
            metadatas=[{
                "source": "mtsamples.csv",
                "specialty": specialty,
                "description": description
            }]
        )
        documents.extend(sub_chunks)

print(f"Created {len(documents)} chunks")

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Building FAISS vector database...")
db = FAISS.from_documents(documents, embeddings)
db.save_local("faiss_index")
print("Done! Vector database saved to faiss_index/")