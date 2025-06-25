import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import tempfile
import os
import fitz  # PyMuPDF

st.set_page_config(page_title="🔐 LLaMA 3 PDF Q&A", layout="wide")
st.title("🔐 LLaMA 3 PDF Q&A with Password Support")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
password = st.text_input("Enter PDF password (if any)", type="password")

def load_pdf_with_password(path: str, password: str) -> str | None:
    try:
        doc = fitz.open(path)
        if doc.needs_pass:
            if not doc.authenticate(password):
                raise ValueError("Incorrect password")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Failed to open PDF: {e}")
        return None

if uploaded_file:
    # save to a temp file so PyMuPDF can open it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    text = load_pdf_with_password(tmp_path, password)

    if text:
        st.success("✅ PDF loaded and parsed successfully.")

        # wrap the full text into a single Document, then split into chunks
        docs = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # CPU-based embeddings for stability
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()

        # Ollama LLaMA 3 for generation
        llm = Ollama(model="llama3", base_url="http://ollama:11434")
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        query = st.text_input("Ask a question about your PDF:")
        if query:
            with st.spinner("Thinking…"):
                answer = qa.run(query)
            st.markdown("**Answer:**")
            st.write(answer)

    # clean up
    os.unlink(tmp_path)
