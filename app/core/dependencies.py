from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder
from langchain_groq import ChatGroq
import os
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RagService
from app.services.cache_service import CacheService
from app.services.guard_service import GuardService, LlamaGuardBackend, LLMClassifierBackend
from app.services.memory_service import MemoryService
from app.services.rate_limiter import RateLimiter
from app.core.cache import llmcache, redis_client
from app.utils.llm_client import LLMClient

GUARD_MODE = "not_llama"

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
bm25 = BM25Encoder().load("app/ingestion/bm25_encoder.json")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
llm = LLMClient(ChatGroq(
    model= "llama-3.1-8b-instant",      
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("GROQ_API_KEY"))
)

judge_llm = LLMClient(ChatGroq(
    model= "llama-3.1-8b-instant",      
    temperature=0,
    max_tokens=500,
    api_key=os.getenv("GROQ_API_KEY")
))

guard_llm = LLMClient(ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    api_key=os.getenv("GROQ_API_KEY")
))

memory_llm = LLMClient(ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=500,
    api_key=os.getenv("GROQ_API_KEY")
))

# --- Services ---
retrieval_service = RetrievalService(index, model, bm25, reranker)
rag_service = RagService(llm, retrieval_service)
cache_service = CacheService(llmcache)

if GUARD_MODE == "llama":
    backend = LlamaGuardBackend()
else:
    backend = LLMClassifierBackend(guard_llm)

guard_service = GuardService(backend) 
memory_service = MemoryService(memory_llm)
rate_limiter = RateLimiter(redis_client=redis_client, limit=20,  window=60 ) # 20 requests per minute per user
