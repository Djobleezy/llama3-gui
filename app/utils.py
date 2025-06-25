import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.schema import Document


def load_pdf_with_password(path: str, password: str) -> str | None:
    """Return the text from a PDF, unlocking it with the given password."""
    try:
        doc = fitz.open(path)
        if doc.needs_pass:
            if not doc.authenticate(password):
                raise ValueError("Incorrect password")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        return None


def create_vector_store(text: str) -> FAISS:
    """Create a FAISS vector store from raw text."""
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def get_qa_chain(vector_store: FAISS) -> RetrievalQA:
    """Set up a RetrievalQA chain using an Ollama LLaMA 3 model."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = Ollama(model="llama3", base_url=base_url)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())

