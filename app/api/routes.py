from fastapi import APIRouter, BackgroundTasks, Depends
from app.models.schemas import SearchRequest
from app.core.dependencies import cache_service, rag_service, guard_service, memory_service, rate_limiter
from app.ingestion.pipeline import ingest_files
import time
from fastapi.concurrency import run_in_threadpool
import asyncio
from app.core.security import create_token
import uuid
from app.api.dependencies import get_current_user, rate_limiter_dep, require_admin  
import logging

logger = logging.getLogger(__name__)

semaphore = asyncio.Semaphore(3)  # global semaphore to limit concurrent RAG generations, not per user. It is intentionally low (3) to prevent Groq rate-limit cascading failures

router = APIRouter()

def blocked_reponse():
    return {"answer": "I'm sorry, I can't respond to that.", "images": [], "citations": []}

@router.post("/token")
def generate_token():
    user_id = str(uuid.uuid4())
    token = create_token(user_id)
    return {"access_token": token}

@router.post("/ask")
async def ask(request: SearchRequest, background_tasks: BackgroundTasks, user_id: str = Depends(get_current_user), _=Depends(rate_limiter_dep)):

    logger.info(f"[ASK] request_received user={user_id} session={request.session_id}")
    start_total = time.time()
    memory_service.create_session_if_not_exist(user_id, request.session_id)
    t0 = time.time()
    status, categories = guard_service.guard(request.query)
    logger.info(f"[GUARD] time={(time.time()-t0)*1000:.2f}ms status={status}")
    if status == 'blocked':
        logger.info(f"User query blocked: {request.query} : {categories}")
        blocked_response = blocked_reponse()
        background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'user', request.query)       
        background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'assistant', blocked_response['answer'], images=blocked_response.get('images',[]), citations=blocked_response['citations'])
        return blocked_reponse
    
    if not request.force_refresh:
        cached = cache_service.get(request.query)
        if cached:
            logger.info("Cache hit")
            background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'user', request.query)       
            background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'assistant', cached['answer'], images=cached.get('images',[]), citations=cached['citations'])
            return cached
        logger.info("Cache miss")
        
    history = memory_service.get_history(user_id, request.session_id)

    t1 = time.time()
    logger.info("[RAG] retrieval+llm started")
    try:
        async with semaphore:
            # response, citations, context = await run_in_threadpool(rag_service.generate, request.query, history )
            response, citations, context = await rag_service.generate(request.query, history)

           
    except Exception as e:
        logger.error(f"Error during RAG generation: {e}")
        response = {"answer": "Sorry, something went wrong while processing your request. Please try again." }
        context = []
        citations = []
    
    logger.info(f"[RAG] retrieval+llm time={(time.time()-t1)*1000:.2f}ms")
    

    # check the output before returning to user
    # status, categories = guard_service.guard(request.query, response["answer"])
    # if status == 'blocked':
    #     logger.info(f"User query and assistant response blocked: {request.query}, {response['answer']} : {categories}")
    #     return blocked_reponse()

    background_tasks.add_task(cache_service.set, request.query, response, citations)
    background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'user', request.query)
    background_tasks.add_task(memory_service.save_message, user_id, request.session_id,'assistant', response['answer'], images=response.get('images',[]), citations=citations)
    background_tasks.add_task(memory_service.maybe_summarize, user_id, request.session_id)

    logger.info(f"[ASK] total_time={(time.time()-start_total)*1000:.2f}ms")
    return {"answer": response["answer"],
            "images": response.get("images", []),
            "citations": citations,
            "context": context
            }


@router.post("/ingest_folder")
def ingest(_=Depends(require_admin)):
    ingest_files("input_data")

@router.delete("/clear_cache")
def clear_cache(user=Depends(require_admin)):
    cache_service.clear()

@router.delete("/clear_history/{session_id}")
def clear_history(session_id: str, user_id: str = Depends(get_current_user)):
    return memory_service.clear_history(user_id, session_id)

@router.get("/sessions")
def get_sessions(user_id:str = Depends(get_current_user)):
    print("SESSIONS USER:", user_id)
    return memory_service.get_sessions(user_id)

@router.get("/history/{session_id}")
def get_history(session_id: str, user_id: str = Depends(get_current_user)):
    return memory_service.get_history_ui(user_id, session_id)

