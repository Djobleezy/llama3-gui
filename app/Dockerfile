FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        libtesseract-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2')
PY

COPY . .

EXPOSE 8501
