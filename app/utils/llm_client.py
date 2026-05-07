import time
import re

class LLMClient:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages, retries=3):
        for attempt in range(retries):
            try:
                return self.llm.invoke(messages)

            except Exception as e:
                err = str(e)

                if "rate_limit" in err or "429" in err:
                    # Groq has strict rate limits; retry backoff is capped to avoid 429 storms
                    sleep_time =  min(2 * (attempt + 1), 5)
                    time.sleep(sleep_time)
                    continue

                raise

        raise Exception("LLM retry failed")