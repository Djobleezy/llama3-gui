import os
import tempfile
import streamlit as st
from utils import (
    load_document,
    create_vector_store,
    get_qa_chain,
    summarize_text,
    rewrite_question,
    MAX_TOKENS,
)

st.set_page_config(page_title="🔐 LLaMA 3 Document Q&A", layout="wide")
st.title("🔐 LLaMA 3 Document Q&A with Password Support")

# Containers to keep the upload widgets pinned to the top
upload_area = st.container()
summary_area = st.container()

with upload_area:
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "txt", "pptx"],
        accept_multiple_files=True,
    )
    password = st.text_input("Enter PDF password (if any)", type="password")
    process_clicked = st.button("Process document")

if "summary" not in st.session_state:
    st.session_state.summary = ""
with summary_area:
    if st.session_state.summary:
        st.info(f"**Summary:** {st.session_state.summary}")

# ─── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    if st.button("New chat / reset", use_container_width=True):
        st.session_state.qa = None
        st.session_state.history = []
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []
    st.divider()
    st.header("Model")
    temp = st.slider(
        "Temperature",
        0.0,
        1.0,
        0.1,
        step=0.05,
        help="Higher values generate more random output",
    )
    tokens = st.number_input(
        "Max tokens",
        value=MAX_TOKENS,
        max_value=MAX_TOKENS,
        min_value=64,
        step=64,
        help="Limits the length of the model's response",
    )
    if st.session_state.get("history"):
        transcript = "\n".join(f"{r}: {m}" for r, m in st.session_state.history)
        st.download_button(
            "Download transcript",
            transcript,
            file_name="chat.txt",
            use_container_width=True,
        )

if "history" not in st.session_state:
    st.session_state.history = []
if "qa" not in st.session_state:
    st.session_state.qa = None
st.session_state.temperature = temp
st.session_state.max_tokens = min(int(tokens), MAX_TOKENS)

if uploaded_files and process_clicked:
    docs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            tmp_path = os.path.join(tmpdir, os.path.basename(f.name))
            with open(tmp_path, "wb") as tmp:
                os.chmod(tmp_path, 0o600)
                tmp.write(f.read())
            try:
                docs.extend(load_document(tmp_path, password))
            except Exception as e:
                st.error(f"{f.name}: {e}")
        if docs:
            st.success("✅ Documents loaded and parsed successfully.")
            text = "\n".join(d.page_content for d in docs)
            summary = summarize_text(text)
            st.session_state.summary = summary
            with summary_area:
                st.info(f"**Summary:** {summary}")
            vector_store = create_vector_store(docs, "store")
            st.session_state.qa = get_qa_chain(
                vector_store,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            )
        else:
            st.session_state.qa = None

if st.session_state.qa:
    for msg in st.session_state.history:
        with st.chat_message(msg[0]):
            st.markdown(msg[1])

    if prompt := st.chat_input("Ask a question about your document:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        clarified = rewrite_question(prompt, st.session_state.history)
        with st.spinner("Thinking…"):
            result = st.session_state.qa.invoke({"question": clarified})
            answer = result["answer"]
            docs = result.get("source_documents", [])
            if docs:
                refs = ", ".join(
                    f"p{d.metadata.get('page', '?')}" for d in docs if d.metadata.get("page")
                )
                if refs:
                    answer += f"\n\nSources: {refs}"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", answer))
