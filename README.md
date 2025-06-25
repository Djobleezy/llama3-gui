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
- Sliders to tweak model temperature and token limits (up to 8192 tokens).

The temperature slider controls the randomness of the model's responses:
higher values lead to more varied answers. The token limit sets the
maximum length of each reply.

## Minimum requirements

Running the full stack locally requires a machine with at least:

- 4 CPU cores
- 8 GB of system memory
- 12 GB of free disk space for the model and indexes
- (optional) a GPU with 8 GB or more of VRAM for faster inference

The containers should run on Linux, macOS or Windows as long as Docker is available.

## Offline usage

To run in an airgapped environment you must build the Docker images while you
still have network access. The custom `Dockerfile.ollama` preloads the `llama3`
model and the app image downloads the required sentence-transformer embeddings.

```bash
docker-compose build
```

After the build completes you can disconnect from the internet and start the
stack normally:

```bash
docker-compose up
```

Once started, the stack runs entirely locally and requires no network access.
If you want to further restrict connectivity, disconnect your machine from the
internet after building the images.

## Development

The main functionality lives in `app/main.py` and helper functions in `app/utils.py`. Dependencies are listed in `app/requirements.txt`.

```
# run checks
python -m py_compile app/*.py
ruff .
pytest
```

If you encounter missing dependencies, install them with `pip install -r app/requirements.txt`.
