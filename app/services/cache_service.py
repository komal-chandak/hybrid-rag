# https://redis.io/docs/latest/develop/ai/redisvl/0.7.0/user_guide/llmcache/

class CacheService:
    def __init__(self, llmcache):
        self.cache = llmcache
    
    def get(self, query):
        results = self.cache.check(prompt=query)
        if results:
            return {"answer": results[0]["response"],
                "images": results[0]["metadata"]['images'],
                "citations": results[0]["metadata"]['citations']}
        return None
    
    def set(self, query, response, citations):
        self.cache.store(
        prompt=query,
        response=response["answer"],
        metadata={ "citations": citations,"images": response["images"]}
    )
        
    def clear(self):
        self.cache.clear()
        return {"status": "cache cleared"}




