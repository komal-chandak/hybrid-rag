from app.ingestion.store import upsert_segments
from docx2pdf import convert
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from pinecone_text.sparse import BM25Encoder
import os
from app.ingestion.processor import process_text
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust as needed

def rebuild_index(all_segments):
    # Fit BM25 once on FULL corpus
    all_texts = [segment["content"] for segment in all_segments]

    bm25 = BM25Encoder()
    bm25.fit(all_texts)
    bm25.dump("bm25_encoder.json")

    upsert_segments(all_segments, bm25=bm25)

def ingest_files(folder_path = "input_data"):
    folder_path = BASE_DIR / folder_path 
    all_segments = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".docx"):
            docx_path = os.path.join(folder_path, filename)
            pdf_path = os.path.join(folder_path, filename.replace(".docx", ".pdf"))
            convert(docx_path, pdf_path)
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(folder_path, filename)
            print(f"Ingesting: {filename}")
            segments = process_text(full_path)
            all_segments.extend(segments)

    rebuild_index(all_segments)

def add_new_file(filepath):
    bm25 = BM25Encoder().load("bm25_encoder.json")
    segments = process_text(filepath)
    upsert_segments(segments, bm25=bm25)

