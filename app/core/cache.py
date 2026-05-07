import warnings
warnings.filterwarnings('ignore')
import redis
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer
from dotenv import load_dotenv
import os
load_dotenv()

redis_url = os.getenv("REDIS_URL")
llmcache = SemanticCache(
    name="llmcache",                                          
    redis_url = redis_url,                                    
    distance_threshold=0.1,                                   
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1"), 

    )

redis_client = redis.Redis.from_url(redis_url, decode_responses=True)   # rate limiter client
