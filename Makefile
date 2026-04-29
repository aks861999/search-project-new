.PHONY: help up down ingest test test-cov lint clean logs smoke-test \
        mlops-up mlops-down mlops-logs flower grafana evaluate drift-check

COMPOSE      = docker compose -f infra/docker-compose.yml
COMPOSE_FULL = docker compose -f infra/docker-compose.yml -f infra/docker-compose.mlops.yml

# ── Default target ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Würth Product Search — available targets"
	@echo "────────────────────────────────────────"
	@echo "  make up            Build images and start all services"
	@echo "  make down          Stop and remove containers"
	@echo "  make ingest        Run the ETL ingestion pipeline"
	@echo "  make test          Run pytest (no live services needed)"
	@echo "  make test-cov      Run pytest with coverage report"
	@echo "  make smoke-test    Hit the live API endpoints (requires make up first)"
	@echo "  make lint          Run ruff linter"
	@echo "  make logs          Tail API service logs"
	@echo "  make clean         Remove containers, volumes, and local caches"
	@echo ""

# ── Bootstrap ──────────────────────────────────────────────────────────────────
.env.docker:
	@if [ ! -f .env.docker ]; then \
	  cp .env.example .env.docker; \
	  echo "Created .env.docker from .env.example — edit it to add GOOGLE_API_KEY"; \
	fi

# ── Infrastructure ─────────────────────────────────────────────────────────────
up: .env.docker
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  API docs  →  http://localhost:8000/docs"
	@echo "  Dashboards→  http://localhost:5601"
	@echo "  OpenSearch→  http://localhost:9200"
	@echo ""
	@echo "Run 'make ingest' to load the dataset."

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f api

# ── Ingestion ──────────────────────────────────────────────────────────────────
ingest: .env.docker
	$(COMPOSE) --profile ingest up ingestion
	@echo "Ingestion complete."

# ── Tests (no live services needed — OpenSearch is fully mocked) ───────────────
test:
	PYTHONPATH=services:. pytest tests/ -v

test-cov:
	PYTHONPATH=services:. pytest tests/ -v \
	  --cov=api --cov=ingestion --cov=mlops --cov=shared \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov
	@echo ""
	@echo "HTML coverage report: htmlcov/index.html"

# ── Smoke test (requires a running stack: make up + make ingest) ───────────────
smoke-test:
	PYTHONPATH=services:. python scripts/smoke_test.py

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	ruff check api/ ingestion/ config/ tests/ scripts/

# ── Cleanup ────────────────────────────────────────────────────────────────────
clean:
	$(COMPOSE) down -v --remove-orphans
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
	@echo "Clean complete."

# ── MLOps ──────────────────────────────────────────────────────────────────────
mlops-up: .env.docker
	$(COMPOSE_FULL) up -d --build celery-ingestion celery-evaluation celery-beat flower prometheus grafana
	@echo ""
	@echo "  Flower     →  http://localhost:5555  (admin / admin)"
	@echo "  Prometheus →  http://localhost:9090"
	@echo "  Grafana    →  http://localhost:3000  (admin / admin)"
	@echo ""

mlops-down:
	$(COMPOSE_FULL) stop celery-ingestion celery-evaluation celery-beat flower prometheus grafana

mlops-logs:
	$(COMPOSE_FULL) logs -f celery-ingestion celery-evaluation

flower:
	@open http://localhost:5555 2>/dev/null || xdg-open http://localhost:5555 2>/dev/null || echo "Open http://localhost:5555"

grafana:
	@open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Open http://localhost:3000"

evaluate:
	curl -s -X POST http://localhost:8000/mlops/evaluate | python3 -m json.tool

drift-check:
	curl -s http://localhost:8000/mlops/drift | python3 -m json.tool
