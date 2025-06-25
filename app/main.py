import os
import tempfile
import streamlit as st
from utils import load_pdf_with_password, create_vector_store, get_qa_chain

st.set_page_config(page_title="🔐 LLaMA 3 PDF Q&A", layout="wide")
st.title("🔐 LLaMA 3 PDF Q&A with Password Support")

# ─── Sidebar controls ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Session")
    if st.button("New chat / reset", use_container_width=True):
        st.session_state.qa = None
        st.session_state.history = []
        st.experimental_rerun()
    if st.button("Clear history", use_container_width=True):
        st.session_state.history = []

if "history" not in st.session_state:
    st.session_state.history = []
if "qa" not in st.session_state:
    st.session_state.qa = None

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
password = st.text_input("Enter PDF password (if any)", type="password")

if uploaded_file and st.button("Process PDF"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    text = load_pdf_with_password(tmp_path, password)
    os.unlink(tmp_path)
    if text:
        st.success("✅ PDF loaded and parsed successfully.")
        vector_store = create_vector_store(text)
        st.session_state.qa = get_qa_chain(vector_store)
    else:
        st.session_state.qa = None

if st.session_state.qa:
    for msg in st.session_state.history:
        with st.chat_message(msg[0]):
            st.markdown(msg[1])

    if prompt := st.chat_input("Ask a question about your PDF:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking…"):
            answer = st.session_state.qa.run(prompt)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.history.append(("user", prompt))
        st.session_state.history.append(("assistant", answer))
