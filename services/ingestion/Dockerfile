# Build context: repo root  (docker-compose.yml sets context: .)
# All COPY paths are relative to the repo root.

FROM python:3.11-slim

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY services/ingestion/requirements.txt /app/ingestion/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/ingestion/requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY shared/ /app/shared/
COPY services/ingestion/  /app/ingestion/

# ── Pre-cache the embedding model ─────────────────────────────────────────────
ARG SKIP_MODEL_DOWNLOAD=0
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "0" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; \
                 SentenceTransformer('BAAI/bge-small-en-v1.5'); \
                 print('Model cached OK')"; \
    else \
      echo "Skipping model download (SKIP_MODEL_DOWNLOAD=1)"; \
    fi

# ── Runtime configuration ─────────────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Run pipeline once and exit (one-shot ETL job)
CMD ["python", "-m", "ingestion.pipeline"]
