# llama3-gui

This project provides a minimal interface for querying PDFs using the LLaMA 3 model via [Ollama](https://ollama.ai/). It runs entirely locally and requires no external API keys.

## Quick start

1. Install [Docker](https://docs.docker.com/get-docker/) and `docker-compose`.
2. Run `docker-compose up --build` to start the Ollama server and Streamlit app.
3. Open `http://localhost:8501` in your browser and upload a PDF. If the file is password protected, supply the password when prompted.
4. Ask questions about the PDF in the text box.

The app keeps a simple history of your questions and answers for the current session.

## Development

The main functionality lives in `app/main.py` and helper functions in `app/utils.py`. Dependencies are listed in `app/requirements.txt`.

```
# run checks
python -m py_compile app/*.py
```
