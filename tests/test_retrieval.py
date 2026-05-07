import pytest
import yaml
import requests
import uuid
import re
import time
import json
import os
from datetime import datetime
from app.core.dependencies import judge_llm
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/ask"

REPORT_FILE = os.getenv("REPORT_FILE", "tests/rag_test_report.jsonl")
MIN_SCORE = 0.80

# ---------------- LOAD TEST CASES ----------------
def load_test_cases():
    file = os.getenv("TEST_FILE", "tests/test_rag_cases.yaml")
    with open(file, "r") as f:
        return yaml.safe_load(f)


# ---------------- AUTH ----------------
def get_token():
    res = requests.post(f"{BASE_URL}/token", timeout=5)
    assert res.status_code == 200
    return res.json()["access_token"]


# ---------------- REQUEST WRAPPER ----------------

def make_request(method, url, token, retries=3, **kwargs):
    def _do_request(tok):
        headers = {"Authorization": f"Bearer {tok}"}
        return requests.request(method, url, headers=headers, **kwargs)

    res = _do_request(token)

    # Handle expired token (same as UI)
    if res.status_code == 401:
        token = get_token()
        res = _do_request(token)

    # Handle rate limiting
    for _ in range(retries):
        if res.status_code == 200:
            return res

        if res.status_code == 429:
            retry_after = int(res.headers.get("Retry-After", 1))
            time.sleep(retry_after)
            res = _do_request(token)
            continue

        assert False, f"Unexpected status: {res.status_code}, {res.text}"

    assert False, "Max retries exceeded"


# ---------------- CITATION PARSER ----------------
def parse_citation(citation_str):
    doc_match = re.search(r"Doc Name:\s*(.*?)\.", citation_str)
    chunk_match = re.search(r"Chunk ID:\s*([^\.\n]+)", citation_str)
    page_match = re.search(r"Pages:\s*(\d+)-(\d+)", citation_str)

    if not doc_match or not page_match:
        return None

    return {
        "doc_id": doc_match.group(1).strip(),
        "chunk_id": chunk_match.group(1).strip() if chunk_match else None,
        "page_start": int(page_match.group(1)),
        "page_end": int(page_match.group(2))
    }
 
# ---------------- LLM as Judge ----------------

def judge_answer(query, context, answer):
    prompt = f"""
    Question: {query}

    Context:
    {context}

    Answer:
    {answer}

    Evaluate the answer on the following metrics (0 to 1):

    1. Faithfulness:
    - Is every claim supported by the context?
    - Penalize if important context is ignored.
    - Penalize answers that include extra or irrelevant items not matching the question.
    - If the question asks for a subset, the answer must strictly filter and not over-include.

    2. Factual Accuracy:
    - Are all facts (dates, numbers, names) correct?
    - Penalize if the answer avoids stating relevant facts to the query that are clearly present in context

    3. Answer Relevance:
    - Penalize vague or evasive answers.

    4. Completeness:
    - Does it include ALL important details from the context?
    - If the question has multiple parts (e.g., what + when + why), then the answer MUST address ALL parts.
    - Penalize heavily if key conditions, exceptions, or contrasts are missing.

    5. Groundedness:
    - Does it avoid using outside knowledge?

    IMPORTANT RULES:
    - Be STRICT. Do NOT reward safe or vague answers.
    - If the answer only partially reflects the context, DO NOT give high scores.
    - Return STRICT JSON:
    {{
        "faithfulness": float,
        "factual_accuracy": float,
        "answer_relevance": float,
        "completeness": float,
        "groundedness": float,
        "reason": "brief explanation"
    }}
    Do NOT return anything else except the above response in the above JSON format. 
    """
    res = judge_llm.invoke(prompt)
    content = res.content.strip()
    content = re.sub(r"```json|```", "", content).strip()
    match = re.search(r"\{.*?\}", content, re.DOTALL)   
    try:
        candidate = match.group(0)
        return json.loads(candidate)
    except:
        return {
            "faithfulness": 0,
            "factual_accuracy": 0,
            "answer_relevance": 0,
            "completeness": 0,
            "groundedness": 0,
            "reason": f"parse failure : {content}"
        }
    
def compute_score(j):
    return (
        0.30 * j["factual_accuracy"] +
        0.25 * j["faithfulness"] +
        0.20 * j["answer_relevance"] +
        0.15 * j["completeness"] +
        0.10 * j["groundedness"]
    )

def append_report(entry):
    with open(REPORT_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

def is_blocked(answer):
    return "can't respond" in answer.lower()

def is_fallback(answer):
    fallback_phrases = [
        "i don't have the information",
        "not available in the context",
        "not explicitly mentioned",
        "do not have enough information",
        "not present in the knowledge base"
    ]
    return any(p in answer.lower() for p in fallback_phrases)

# ---------------- TEST ----------------
@pytest.mark.parametrize("case", load_test_cases())
def test_api(case):

    token = get_token()
    session_id = str(uuid.uuid4())

    payload = {
        "query": case["query"],
        "session_id": session_id,
        "force_refresh": True
    }

    # ---------------- BASE REPORT STRUCTURE ----------------
    report_entry = {
        "test_name": case.get("name"),
        "type": case.get("type", "rag"),
        "query": case["query"],
        "expected_doc": case.get("expected_doc"),
        "expected_page": case.get("expected_page"),
        "timestamp": datetime.now().isoformat(),

        # runtime fields (always present)
        "answer": None,
        "context": None,
        "citations": None,
        "images": None,
        "latency_ms": None,
        "guard_triggered": None,
        "fallback": None,
        "citation_match": None,
        "scores": None,
        "overall_score": None,

        # result
        "status": "unknown",
        "error": None
    }
    try:
        start = time.time()
        response = make_request(
            "POST",
            API_URL,
            token=token,
            json=payload,
            timeout=120
        )
        data = response.json()
        latency = (time.time() - start) * 1000  # ms
        report_entry["latency_ms"] = latency
        assert response.status_code == 200

        expected_doc = case.get("expected_doc")
        expected_page = case.get("expected_page")
        case_type = case.get("type", "rag")

        answer = data.get("answer", "")
        citations = data.get("citations", [])
        context = data.get("context", "") 
        images = data.get("images", []) 

        report_entry.update({
                "answer": answer,
                "context": context,
                "citations": citations,
                "images": images
        })
        guard_triggered = is_blocked(answer)
        fallback = is_fallback(answer)
        report_entry.update({
                "guard_triggered": guard_triggered,
                "fallback": fallback
        })

        if case_type == "guard_block":
            assert guard_triggered, f"Expected guard block but got: {answer}"

        elif case_type == "fallback":
            assert not guard_triggered, "Should not be blocked"
            assert fallback, f"Expected fallback but got: {answer}"

        elif case_type == "rag":
            assert not guard_triggered, "RAG query should not be blocked"
            assert not fallback, "RAG query should not fallback"

        # ---------- CITATION CHECK ----------
        citation_match = None
        if case_type == "rag":
            parsed_citations = [parse_citation(c) for c in citations if parse_citation(c)]

            citation_match = any(
                c["doc_id"] == expected_doc and
                c["page_start"] <= expected_page <= c["page_end"]
                for c in parsed_citations
            )
        
            assert citation_match, f"Citation mismatch for query: {case['query']}"

        # ---------- LLM JUDGE ----------
        judge_result = None
        overall_score = None
        if case_type == "rag":
            judge_result = judge_answer(case["query"], context, answer)
            overall_score = compute_score(judge_result)
            report_entry["scores"] = judge_result
            report_entry["overall_score"] = overall_score

            assert overall_score >= MIN_SCORE, (
                f"Low quality answer: {overall_score} | Query: {case['query']}"
            )
        report_entry["status"] = "passed"
    
    except AssertionError as e:
        report_entry["status"] = "failed"
        report_entry["error"] = str(e)

    except Exception as e:
        report_entry["status"] = "error"
        report_entry["error"] = str(e)

    finally:
        append_report(report_entry)

    # Fail test AFTER logging
    if report_entry["status"] != "passed":
        pytest.fail(report_entry["error"] or "Test failed")
