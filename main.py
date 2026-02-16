import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

from config import (
    API_KEY,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K
)

from gpt_memory import GPTMemory
from bm25_retriever import BM25Retriever
from guardrails import (
    validate_user_input,
    validate_context,
    post_process_answer
)

# ------------------------
# OpenAI client
# ------------------------
client = OpenAI(api_key=API_KEY)

# ------------------------
# Retrieval engines
# ------------------------
vector_memory = GPTMemory()          # Embedding + simple cosine similarity
bm25_retriever = BM25Retriever()     # Keyword-based BM25

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(
    page_title="RAG Chatbot 2026",
    page_icon="🤖"
)

st.title("RAG Chatbot 2026")
st.write("Upload a PDF/TXT document and ask questions strictly based on its content.")

# ------------------------
# Text splitter
# ------------------------
def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# ------------------------
# File upload
# ------------------------
uploaded_file = st.file_uploader(
    "Upload a document (PDF or TXT)",
    type=["pdf", "txt"]
)

document_chunks = []

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PdfReader(uploaded_file)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        text = uploaded_file.read().decode("utf-8")

    document_chunks = split_text(text)

    # Add chunks to both retrievers
    for chunk in document_chunks:
        vector_memory.add_response(chunk)

    bm25_retriever.add_documents(document_chunks)

    st.success("Document indexed successfully!")

# ------------------------
# User input
# ------------------------
user_input = st.text_input("Ask a question:")

if "history" not in st.session_state:
    st.session_state.history = []

# ------------------------
# RAG Pipeline
# ------------------------
if user_input:

    # 1️⃣ Guardrail: input validation
    valid, msg = validate_user_input(user_input)

    if not valid:
        st.warning(msg)
        st.stop()

    if msg == "GREETING":
        st.write("Hello! 👋 Please ask a question related to the uploaded document.")
        st.stop()

    if not document_chunks:
        st.warning("Please upload a document first.")
        st.stop()

    # 2️⃣ Retrieve context
    bm25_chunks = bm25_retriever.retrieve(user_input, top_k=TOP_K)
    vector_chunks = vector_memory.retrieve_topk(user_input, top_k=TOP_K)

    # Merge + deduplicate
    context_chunks = list(dict.fromkeys(bm25_chunks + vector_chunks))

    # 3️⃣ Guardrail: context validation
    valid, msg = validate_context(context_chunks, [])
    if not valid:
        answer = msg
    else:
        combined_context = "\n\n".join(context_chunks)

        system_prompt = (
            "You are a strict RAG assistant.\n"
            "Use ONLY the provided context to answer.\n"
            "If the answer is not present, say: I don’t know."
        )

        # 4️⃣ Generate Answer
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"Context:\n{combined_context}\n\nQuestion:\n{user_input}"
                }
            ],
            max_tokens=300
        )

        raw_answer = response.choices[0].message.content

        # 5️⃣ Output guardrail
        answer = post_process_answer(raw_answer)

    # 6️⃣ Update chat history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", answer))

# ------------------------
# Display chat history
# ------------------------
for role, message in st.session_state.history:
    st.markdown(f"**{role}:** {message}")
