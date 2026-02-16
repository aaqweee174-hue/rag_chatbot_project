import re

# -----------------------------
# Greeting Detection
# -----------------------------
def is_greeting(user_input: str) -> bool:
    greetings = ["hi", "hello", "hey", "thanks", "thank you"]
    return user_input.lower().strip() in greetings


# -----------------------------
# Input Validation
# -----------------------------
def validate_user_input(user_input: str):
    if not user_input or len(user_input.strip()) == 0:
        return False, "Please enter a question."

    if is_greeting(user_input):
        return True, "GREETING"

    blocked_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"you are chatgpt",
        r"jailbreak",
        r"act as"
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, user_input.lower()):
            return False, "Your query contains unsafe instructions."

    return True, ""


# -----------------------------
# Context Validation
# -----------------------------
def validate_context(context_chunks, similarity_scores):
    if not context_chunks:
        return False, "I couldn't find relevant information in the document."

    if similarity_scores and max(similarity_scores) < 0.3:
        return False, "The question seems unrelated to the document."

    return True, ""


# -----------------------------
# Output Validation
# -----------------------------
def post_process_answer(answer):
    if not answer or len(answer.strip()) < 5:
        return "I don’t know based on the provided document."

    bad_phrases = [
        "general knowledge",
        "not in context",
        "outside the document"
    ]

    for p in bad_phrases:
        if p in answer.lower():
            return "I don’t know based on the provided document."

    return answer 