import os
import tempfile
import streamlit as st
from utils import (
    load_document,
    create_vector_store,
    get_qa_chain,
    summarize_text,
    rewrite_question,
)

st.set_page_config(page_title="🔐 LLaMA 3 Document Q&A", layout="wide")
st.title("🔐 LLaMA 3 Document Q&A with Password Support")

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
    temp = st.slider("Temperature", 0.0, 1.0, 0.1, step=0.05)
    tokens = st.number_input("Max tokens", value=256, step=64)
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
st.session_state.max_tokens = int(tokens)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "pptx"],
    accept_multiple_files=True,
)
password = st.text_input("Enter PDF password (if any)", type="password")

if uploaded_files and st.button("Process document"):
    docs = []
    for f in uploaded_files:
        ext = os.path.splitext(f.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(f.read())
            tmp_path = tmp.name
        try:
            docs.extend(load_document(tmp_path, password))
        except Exception as e:
            st.error(f"{f.name}: {e}")
        finally:
            os.unlink(tmp_path)
    if docs:
        st.success("✅ Documents loaded and parsed successfully.")
        text = "\n".join(d.page_content for d in docs)
        st.info(f"**Summary:** {summarize_text(text)}")
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
