# Build context: repo root  (docker-compose.yml sets context: .)
# All COPY paths are relative to the repo root.

FROM python:3.11-slim AS base

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY services/api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /app/api/requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
# Shared config package (used by both api and ingestion)
COPY shared/ /app/shared/

# Ingestion schema is imported by api at startup (for HYBRID_PIPELINE_ID)
COPY services/ingestion/__init__.py  /app/ingestion/__init__.py
COPY services/ingestion/schema.py    /app/ingestion/schema.py

# API package
COPY services/api/  /app/api/

# ── Pre-cache the embedding model ─────────────────────────────────────────────
# Set SKIP_MODEL_DOWNLOAD=1 in CI to avoid downloading ~130MB during builds.
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

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--loop", "asyncio", \
     "--access-log"]
