import os
import json
import re
from app.core.config import settings
from app.services.history_service import HistoryMessage

_llm = None

SUBJECT_SYSTEM_PROMPTS = {
    "physics": (
        "You are TutorAI, an expert IIT-JEE Physics tutor. "
        "Explain concepts using first principles, derive formulas step-by-step, "
        "and always show units. Use LaTeX math notation for all equations."
    ),
    "chemistry": (
        "You are TutorAI, an expert IIT-JEE Chemistry tutor. "
        "Explain reactions with mechanisms, balance equations, "
        "and clarify IUPAC nomenclature. Use LaTeX for chemical equations."
    ),
    "mathematics": (
        "You are TutorAI, an expert IIT-JEE Mathematics tutor. "
        "Show complete working with every algebraic step. "
        "Use LaTeX for all mathematical expressions and proofs."
    ),
    "general": (
        "You are TutorAI, an expert IIT-JEE tutor covering Physics, Chemistry, and Mathematics. "
        "Provide clear, step-by-step explanations using LaTeX for equations."
    ),
}

def get_llm():
    """Lazy-load the GGUF model (singleton)."""
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        model_path = settings.model_path
        if not os.path.exists(model_path):
            print(
                f"[LLM] Warning: GGUF model not found at '{model_path}'. "
                "Chat responses will be mocked."
            )
            return None
            
        print(f"[LLM] Loading model from: {model_path}")
        _llm = Llama(
            model_path=model_path,
            n_ctx=settings.n_ctx,
            n_gpu_layers=settings.n_gpu_layers,
            verbose=False,
        )
        print("[LLM] Model loaded successfully!")
    return _llm

def build_messages(subject: str, history: list[HistoryMessage], question: str) -> list[dict]:
    """Build the messages list for llama-cpp chat completion."""
    system_prompt = SUBJECT_SYSTEM_PROMPTS.get(subject, SUBJECT_SYSTEM_PROMPTS["general"])
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": question})
    return messages

def extract_topic_tags(llm, question: str, answer: str) -> list[str]:
    """Ask the LLM to infer topic tags from the Q&A pair."""
    if llm is None:
        return ["mock-tag"]
        
    tag_prompt = (
        f"Extract up to 3 topic tags (lowercase, hyphenated) from this Q&A. "
        f"Respond with ONLY a JSON array like [\"tag1\", \"tag2\"].\n\n"
        f"Q: {question[:200]}\nA: {answer[:300]}"
    )
    try:
        r = llm.create_chat_completion(
            messages=[{"role": "user", "content": tag_prompt}],
            max_tokens=60,
            temperature=0.1,
        )
        raw = r["choices"][0]["message"]["content"].strip()
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return []

def generate_answer(question: str, history: list[HistoryMessage], subject: str = "general") -> dict:
    """
    Generate an AI answer.
    Returns: { answer: str, topic_tags: list[str] }
    """
    llm = get_llm()
    if llm is None:
        return {
            "answer": f"Mock answer to '{question}'. (Model not loaded)",
            "topic_tags": ["mock-tag-1"]
        }
        
    messages = build_messages(subject, history, question)
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
        top_p=0.9,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    topic_tags = extract_topic_tags(llm, question, answer)
    return {"answer": answer, "topic_tags": topic_tags}

def generate_quiz(weak_topics: list[str], subject: str) -> dict:
    llm = get_llm()
    topics_str = ", ".join(weak_topics) if weak_topics else subject

    if llm is None:
        return {
            "questions": [
                {
                    "id": 1,
                    "question": f"Sample question on {topics_str} (Mock)",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "A",
                    "explanation": "Mock explanation."
                }
            ]
        }
        
    prompt = f"""Generate exactly 5 IIT-JEE level multiple choice questions on these {subject} topics: {topics_str}.

Return ONLY a valid JSON object with this exact structure:
{{
  "questions": [
    {{
      "id": 1,
      "question": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer": "A",
      "explanation": "..."
    }}
  ]
}}

Rules:
- correct_answer must be exactly one of: A, B, C, D
- options must have exactly 4 entries
- questions should be IIT-JEE difficulty level
- explanations must be concise but complete"""

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.4,
    )
    raw = response["choices"][0]["message"]["content"].strip()
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "questions": [
            {
                "id": 1,
                "question": f"Sample question on {topics_str} (AI failed to return JSON)",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "explanation": "Check AI formatting."
            }
        ]
    }
