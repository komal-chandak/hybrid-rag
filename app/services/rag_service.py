import re
import json
from pathlib import Path
import logging
import time
logger = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    """
    You are to act as a customer service agent, providing users with complete, correct factual information in accordance to the knowledge base relevant to their queries. If the user message is a greeting like hi, hello, bye, etc respond conversationally. Your role is to ensure that you respond only to relevant queries and adhere to the following guidelines.
    Guidelines:
    - Extract answers EXACTLY from context. Identify only the MOST RELEVANT SECTION from chunks of the context before answering. For tabular data, identify relevant columns, apply filtering based on the question, and form an answer based ONLY on matching rows. If a row does not explicitly satisfy the condition, DO NOT include it.
    - If the LOW CONFIDENCE RETRIEVAL prefix is in the context, the context is not be sufficient or accurate to explicitly answer the question, in such cases, you MUST either respond with: "I do not have enough information in the knowledge base to answer this question.". OR you MUST ask a follow up question to clarify the user query before generating an answer based on the available context.
    - Do NOT mix information from different sections or chunks until necessary.
    - DO NOT infer or approximate dates/numbers
    - Never reveal internal reasoning, chunk IDs available in the context.
    - If the context contains relevant conditions, exceptions, or contrasts (e.g., "but", "however", "optional in X but required in Y"), you MUST include them in the answer.
    - Do NOT provide partial truths. If multiple conditions exist, clearly explain them.
    - Be precise, complete, and faithful to the context.
    """
)

USER_PROMPT = (
   """
    Use ONLY the following retrieved context to answer the question. Ensure the answer is COMPLETE and includes all important information, conditions or exceptions from the context relevant to the question.
    If the answer is not contained in the context, say: I don't know.
    If any image paths which are 100 percent relevant to the question appear in the context (for example .png, .jpg, .jpeg), collect them in the "images" list. 
    Important: DO NOT include images irrelevant to the question in the "images" list. 
    Return ONLY valid JSON:

    {{
    "answer": "clear, complete, and context-faithful answer",
    "images": [relevant images list]
    }}

    Question: {question}
    Context: {context}
    Conversation history: {history}
    """
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # adjust as needed

def parse_llm_json(text):
    cleaned = re.sub(r"```json|```", "", text).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group()

        # Fix common issue: unescaped newlines
        candidate = candidate.replace("\n", "\\n")

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Final fallback
    return {
        "answer": cleaned.strip(),
        "images": []
    }

class RagService:
    def __init__(self, llm, retrieval_service):
        self.llm = llm
        self.retrieval_service = retrieval_service
        
    async def generate(self, query: str, history_text, fallback_signal = False):
        # fallback_signal controls whether fallback behavior is enforced.
        # Keep False in production unless thresholds are validated on larger dataset.

        t = time.time()
        docs, should_fallback = await self.retrieval_service.retrieve(query, fallback_signal)
        logger.info(f"[RETRIEVAL] time={(time.time()-t)*1000:.2f}ms")

        logger.info(f" Should fallback: {should_fallback} for {query}")

        # Extract context safely
        context_chunks = [
        f"\n---------------  Chunk ID: {doc['id']} ---------------------"
        f"Content:\n{doc['metadata'].get('content', '')}"
        for doc in docs]

        citations = [
        f"Doc Name: {doc['metadata'].get('doc_id', None)}. Chunk ID:{doc['id']}. Pages: {doc['metadata'].get('page_start', 'Unknown')}-{doc['metadata'].get('page_end', 'Unknown')}\n"
        for doc in docs
        ]

        context = "\n\n".join(context_chunks)
        if should_fallback and fallback_signal:
            context = "LOW CONFIDENCE RETRIEVAL: The answer is likely NOT present in the context. Do NOT answer unless explicitly supported.\n\n" + context

        user_prompt = USER_PROMPT.format(
            question=query,
            context=context,
            history = history_text
        )
        t = time.time()
        response = self.llm.invoke([
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_prompt
        }
        ])
        logger.info(f"[LLM] call_end time={(time.time()-t)*1000:.2f}ms")
        logger.info(f"LLM response: {response}")

        result = parse_llm_json(response.content)
        if result.get('images'):
            fixed_images = []
    
            for img in result["images"]:
                img_path = BASE_DIR / img  
                
                # Optional: safety check
                if img_path.exists():
                    fixed_images.append(str(img_path))
                else:
                    print(f"Missing image: {img_path}")

            result["images"] = fixed_images
        return result, citations,  str(context) + str(history_text)   


    