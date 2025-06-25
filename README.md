# llama3-gui

This project provides a minimal interface for querying local documents using the LLaMA 3 model via [Ollama](https://ollama.ai/). It runs entirely locally and requires no external API keys.

## Quick start

1. Install [Docker](https://docs.docker.com/get-docker/) and `docker-compose`.
2. Run `docker-compose up --build` to start the Ollama server and Streamlit app.
3. Open `http://localhost:8501` in your browser and upload a PDF, DOCX, TXT or PPTX file. If you upload a PDF that is password protected, supply the password when prompted.
4. Use the chat box at the bottom to ask questions about the document. Messages appear in a chat-style layout and the sidebar lets you reset or clear history.

The app now maintains context between questions using a conversational retrieval chain with memory. Features include:

- Loading multiple documents at once.
- Automatic summaries of uploaded content.
- Persistent search indexes for faster reuse.
- Page references in answers when available.
- A button to download your chat transcript.
- Sliders to tweak model temperature and token limits.

The temperature slider controls the randomness of the model's responses:
higher values lead to more varied answers. The token limit sets the
maximum length of each reply.

## Development

The main functionality lives in `app/main.py` and helper functions in `app/utils.py`. Dependencies are listed in `app/requirements.txt`.

```
# run checks
python -m py_compile app/*.py
ruff .
pytest
```

If you encounter missing dependencies, install them with `pip install -r app/requirements.txt`.
