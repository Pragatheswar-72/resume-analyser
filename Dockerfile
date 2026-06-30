# Resume Analyser API — container image for the FastAPI backend.
FROM python:3.11-slim

# Avoid .pyc files and buffer issues; keep logs flowing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the first request isn't slow.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

EXPOSE 8000

# GEMINI_API_KEY is provided at runtime: docker run -e GEMINI_API_KEY=...
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
