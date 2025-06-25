import os
import fitz  # PyMuPDF
import docx
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document


def load_docx(path: str) -> str:
    """Return the text content from a DOCX file."""
    doc = docx.Document(path)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def load_txt(path: str) -> str:
    """Return the text content from a plain text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pptx(path: str) -> str:
    """Return the text content from a PowerPoint file."""
    prs = Presentation(path)
    texts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                texts.append(shape.text)
    return "\n".join(texts)


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


def load_document(path: str, password: str = "") -> str | None:
    """Load text from various document formats based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf_with_password(path, password)
    if ext == ".docx":
        return load_docx(path)
    if ext == ".txt":
        return load_txt(path)
    if ext in {".ppt", ".pptx"}:
        return load_pptx(path)
    return None


def create_vector_store(text: str) -> FAISS:
    """Create a FAISS vector store from raw text."""
    docs = [Document(page_content=text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)


def get_qa_chain(vector_store: FAISS) -> ConversationalRetrievalChain:
    """Return a conversational QA chain with memory using an Ollama LLaMA 3 model."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = Ollama(model="llama3", base_url=base_url)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
    )

