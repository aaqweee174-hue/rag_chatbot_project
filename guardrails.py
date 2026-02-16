# guardrails.py
import re

# -----------------------------
# Input Guardrails
# -----------------------------
def validate_user_input(user_input: str) -> bool:
    """
    Block prompt injection / empty input
    """
    if not user_input or len(user_input.strip()) < 3:
        return False

    blocked_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are chatgpt",
        r"jailbreak"
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, user_input.lower()):
            return False

    return True


# -----------------------------
# Context Guardrails
# -----------------------------
def validate_context(context_chunks: list[str]) -> bool:
    """
    If no context retrieved → do NOT answer
    """
    if not context_chunks:
        return False

    combined_length = sum(len(c) for c in context_chunks)
    if combined_length < 100:
        return False

    return True


# -----------------------------
# Output Guardrails
# -----------------------------
def post_process_answer(answer: str, context_chunks: list[str]) -> str:
    """
    Final hallucination control
    """
    if not answer or len(answer.strip()) < 5:
        return "I don’t know based on the provided document."

    # If model tries to go outside context
    if "general knowledge" in answer.lower():
        return "I don’t know based on the provided document."

    return answer
