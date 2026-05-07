import time

class RateLimiter:
    def __init__(self, redis_client, limit=20, window=60):
        self.redis = redis_client
        self.limit = limit
        self.window = window  # seconds

    def allow(self, user_id: str):
        now = time.time()
        key = f"rate:{user_id}"

        pipeline = self.redis.pipeline()

        # Remove old requests
        pipeline.zremrangebyscore(key, 0, now - self.window)

        # Count current requests
        pipeline.zcard(key)

        # Add current request
        pipeline.zadd(key, {str(now): now})

        # Set expiry 
        pipeline.expire(key, self.window)

        _, count, _, _ = pipeline.execute()

        if count >= self.limit:
            return False, int(self.window)

        return True, None