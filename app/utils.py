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


def summarize_text(text: str) -> str:
    """Return a short summary of ``text`` using LLaMA 3."""
    if not text:
        return ""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = Ollama(model="llama3", base_url=base_url, temperature=0.1)
    prompt = f"Summarize this text in a few sentences:\n\n{text[:4000]}"
    return llm.invoke(prompt).strip()


def load_docx(path: str) -> list[Document]:
    """Return the text content from a DOCX file."""
    doc = docx.Document(path)
    docs = []
    for i, paragraph in enumerate(doc.paragraphs, start=1):
        if paragraph.text.strip():
            docs.append(Document(page_content=paragraph.text, metadata={"paragraph": i}))
    return docs


def load_txt(path: str) -> list[Document]:
    """Return the text content from a plain text file.

    The function first attempts to decode the file as UTF-8. If that fails
    due to a ``UnicodeDecodeError`` it falls back to a list of common
    encodings before finally ignoring undecodable bytes. This prevents
    crashes when the text file uses a legacy code page such as Windows-1252.
    """

    with open(path, "rb") as f:
        raw = f.read()

    encodings = ["utf-8", "cp1252", "latin-1"]
    for enc in encodings:
        try:
            text = raw.decode(enc)
            return [Document(page_content=text)]
        except UnicodeDecodeError:
            continue

    # As a last resort, drop undecodable bytes.
    text = raw.decode("utf-8", errors="ignore")
    return [Document(page_content=text)]


def load_pptx(path: str) -> list[Document]:
    """Return the text content from a PowerPoint file."""
    prs = Presentation(path)
    docs = []
    for i, slide in enumerate(prs.slides, start=1):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                slide_text.append(shape.text)
        if slide_text:
            docs.append(
                Document(page_content="\n".join(slide_text), metadata={"page": i})
            )
    return docs


def load_pdf_with_password(path: str, password: str) -> list[Document]:
    """Return the text from a PDF, unlocking it with the given password."""
    doc = fitz.open(path)
    if doc.needs_pass and not doc.authenticate(password):
        raise ValueError("Incorrect password")
    docs = [
        Document(page_content=page.get_text(), metadata={"page": i + 1})
        for i, page in enumerate(doc)
    ]
    return docs


def load_document(path: str, password: str = "") -> list[Document]:
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
    return []


def create_vector_store(docs: list[Document], path: str) -> FAISS:
    """Create (or load) a persistent FAISS vector store from ``docs``."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(path)
    return vector_store


def get_qa_chain(
    vector_store: FAISS,
    temperature: float = 0.1,
    max_tokens: int = 256,
) -> ConversationalRetrievalChain:
    """Return a conversational QA chain with memory using LLaMA 3."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    llm = Ollama(
        model="llama3",
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

