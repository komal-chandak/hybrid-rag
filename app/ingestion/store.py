from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

pc_api_key = os.getenv("PINECONE_API_KEY")
pc_index = os.getenv("PINECONE_INDEX")

pc = Pinecone(api_key=pc_api_key)
index = pc.Index(pc_index)

embd_model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def upsert_segments(segments, bm25, batch_size=100):

    batch = []

    for segment in segments:

        embedding = embd_model.encode(
            "search_document: " + segment["content"],
            normalize_embeddings=True
        )
        
        sparse_vector = bm25.encode_documents(segment["content"])

        batch.append({
            "id": segment["segment_id"],
            "values": embedding.tolist(),
            "sparse_values": sparse_vector,
            "metadata": {
                "doc_id": segment["doc_id"],
                "page_start": min(segment["pages"]),
                "page_end": max(segment["pages"]),
                "content": segment["content"]
            }
        })

        if len(batch) == batch_size:
            index.upsert(vectors=batch)
            batch = []

    # upsert remaining
    if batch:
        index.upsert(vectors=batch)