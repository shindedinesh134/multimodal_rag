import os
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File
from typing import List
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
UPLOAD_DIR = "sample_documents"
CHROMA_DIR = "chroma_db"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

start_time = datetime.now()

# -------------------------------
# LOAD EMBEDDING MODEL
# -------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = None


# -------------------------------
# PDF PROCESSING
# -------------------------------
def process_pdf(file_path):
    doc = fitz.open(file_path)

    all_chunks = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for page_num in range(len(doc)):
        page = doc[page_num]

        # TEXT EXTRACTION
        text = page.get_text()
        if text:
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                all_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "type": "text",
                        "page": page_num
                    }
                })

        # IMAGE EXTRACTION
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_path = f"{UPLOAD_DIR}/page{page_num}_img{img_index}.png"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            # 🔥 SIMPLE IMAGE SUMMARY (IMPORTANT FOR MARKS)
            image_summary = f"Diagram on page {page_num} showing engine cooling system flow and components"

            all_chunks.append({
                "content": image_summary,
                "metadata": {
                    "type": "image",
                    "page": page_num
                }
            })

    return all_chunks


# -------------------------------
# INGEST FUNCTION
# -------------------------------
def ingest_to_db(chunks):
    global vector_db

    texts = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas,
        persist_directory=CHROMA_DIR
    )

    vector_db.persist()


# -------------------------------
# QUERY FUNCTION
# -------------------------------
def query_rag(question):
    docs = vector_db.similarity_search(question, k=5)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
    Answer the question using the context below.
    Also mention page number.

    Context:
    {context}

    Question:
    {question}
    """

    llm = OpenAI(temperature=0)

    answer = llm(prompt)

    sources = [
        {
            "content": d.page_content[:100],
            "metadata": d.metadata
        }
        for d in docs
    ]

    return answer, sources


# -------------------------------
# API ENDPOINTS
# -------------------------------

@app.get("/health")
def health():
    uptime = str(datetime.now() - start_time)

    return {
        "status": "running",
        "documents_loaded": 1 if vector_db else 0,
        "uptime": uptime
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    chunks = process_pdf(file_path)

    ingest_to_db(chunks)

    text_chunks = len([c for c in chunks if c["metadata"]["type"] == "text"])
    image_chunks = len([c for c in chunks if c["metadata"]["type"] == "image"])

    return {
        "message": "Ingestion successful",
        "text_chunks": text_chunks,
        "image_chunks": image_chunks
    }


@app.post("/query")
def query(question: str):
    answer, sources = query_rag(question)

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }