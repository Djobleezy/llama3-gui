import os
import tempfile
import streamlit as st
from utils import load_pdf_with_password, create_vector_store, get_qa_chain

st.set_page_config(page_title="🔐 LLaMA 3 PDF Q&A", layout="wide")
st.title("🔐 LLaMA 3 PDF Q&A with Password Support")

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
    query = st.text_input("Ask a question about your PDF:")
    if st.button("Submit") and query:
        with st.spinner("Thinking…"):
            answer = st.session_state.qa.run(query)
        st.session_state.history.append((query, answer))

if st.session_state.history:
    st.markdown("## History")
    for q, a in st.session_state.history:
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
