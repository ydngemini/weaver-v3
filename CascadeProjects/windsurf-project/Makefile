# ═══════════════════════════════════════════════════════════════
# Weaver v3 — Makefile (DevOps + MLOps)
# ═══════════════════════════════════════════════════════════════

SHELL := /bin/bash
PYTHON := venv/bin/python3
PIP := venv/bin/pip
PROJECT := $(shell pwd)

.DEFAULT_GOAL := help

# ── Setup ────────────────────────────────────────────────────

.PHONY: setup
setup: ## Create venv and install dependencies
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-core.txt
	@echo "✅ Setup complete. Copy .env.example to .env and fill in API keys."

.PHONY: setup-full
setup-full: ## Install all deps including training
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✅ Full setup complete (including LoRA training deps)."

.PHONY: freeze
freeze: ## Freeze current venv to requirements.txt
	$(PIP) freeze > requirements.txt
	@echo "📦 requirements.txt updated ($(shell wc -l < requirements.txt) packages)"

# ── Run ──────────────────────────────────────────────────────

.PHONY: run
run: preflight ## Full stack with VTV (mic + camera)
	$(PYTHON) weaver.py --heartbeat

.PHONY: run-headless
run-headless: preflight ## Backend only (no mic/camera)
	$(PYTHON) weaver.py --headless

.PHONY: run-phone
run-phone: preflight ## Phone bridge + quantum API only
	./start_weaver.sh --phone-only

.PHONY: run-start
run-start: preflight ## Full stack via start_weaver.sh
	./start_weaver.sh

# ── Testing ──────────────────────────────────────────────────

.PHONY: test
test: ## Run full test suite
	$(PYTHON) whole_codebase_tests.py

.PHONY: test-integration
test-integration: ## Run integration tests
	$(PYTHON) run_integration_tests.py

.PHONY: test-n8n
test-n8n: ## Test n8n workflow endpoints
	$(PYTHON) test_n8n_endpoints.py

.PHONY: test-phone
test-phone: ## Test phone bridge health
	@curl -sf http://localhost:8765/health && echo "✅ Phone Bridge OK" || echo "❌ Phone Bridge down"

.PHONY: test-quick
test-quick: ## Quick smoke test — check all lobes respond
	@echo "🔍 Quick health check..."
	@curl -sf http://localhost:9996/health >/dev/null 2>&1 && echo "  ✅ Health Dashboard" || echo "  ❌ Health Dashboard"
	@curl -sf http://localhost:9997/health >/dev/null 2>&1 && echo "  ✅ Quantum API" || echo "  ❌ Quantum API"
	@curl -sf http://localhost:9995/health >/dev/null 2>&1 && echo "  ✅ Akashic Hub API" || echo "  ❌ Akashic Hub API"
	@curl -sf http://localhost:8765/health >/dev/null 2>&1 && echo "  ✅ Phone Bridge" || echo "  ❌ Phone Bridge"
	@curl -sf http://localhost:8899/health >/dev/null 2>&1 && echo "  ✅ LoRA Server" || echo "  ❌ LoRA Server"
	@curl -sf http://localhost:5678/healthz >/dev/null 2>&1 && echo "  ✅ n8n" || echo "  ❌ n8n"

.PHONY: lint
lint: ## Syntax-check all Python files
	@echo "🔍 Checking syntax..."
	@FAIL=0; for f in *.py; do \
		$(PYTHON) -m py_compile "$$f" 2>&1 || FAIL=1; \
	done; \
	if [ $$FAIL -eq 0 ]; then echo "✅ All files clean"; else echo "❌ Syntax errors found"; exit 1; fi

# ── Docker ───────────────────────────────────────────────────

.PHONY: docker-build
docker-build: ## Build Docker image
	docker build -t weaver:latest .

.PHONY: docker-up
docker-up: ## Start full stack via docker-compose
	docker compose up -d

.PHONY: docker-down
docker-down: ## Stop all Docker services
	docker compose down

.PHONY: docker-logs
docker-logs: ## Tail Weaver logs
	docker compose logs -f weaver

.PHONY: docker-ps
docker-ps: ## Show running containers
	docker compose ps

# ── Monitoring ───────────────────────────────────────────────

.PHONY: health
health: ## Show lobe status (JSON)
	@curl -sf http://localhost:9996/api/status | python3 -m json.tool 2>/dev/null || echo "❌ Health dashboard not running"

.PHONY: quantum
quantum: ## Show current quantum state
	@curl -sf http://localhost:9997/quantum/current | python3 -m json.tool 2>/dev/null || echo "❌ Quantum API not running"

.PHONY: lobes
lobes: ## Show active Akashic Hub lobes
	@curl -sf http://localhost:9995/akashic/lobes | python3 -m json.tool 2>/dev/null || echo "❌ Akashic Hub API not running"

.PHONY: metrics
metrics: ## Show system metrics
	@curl -sf http://localhost:9996/api/metrics 2>/dev/null | python3 -m json.tool || echo "❌ Metrics not available"

.PHONY: logs
logs: ## Tail the dream state log
	@tail -f Nexus_Vault/weaver_dreams.md 2>/dev/null || echo "No dreams yet."

# ── MLOps (LoRA Training) ───────────────────────────────────

.PHONY: train-data
train-data: ## Prepare LoRA training dataset
	$(PYTHON) forge_dataset.py

.PHONY: train
train: ## Train LoRA adapter (GPU required)
	$(PYTHON) forge_soul.py

.PHONY: merge-lora
merge-lora: ## Merge LoRA adapter into base model
	$(PYTHON) merge_lora.py

.PHONY: deploy-lora
deploy-lora: ## Restart LoRA server with latest adapter
	@pkill -f lora_server.py 2>/dev/null || true
	@sleep 2
	$(PYTHON) lora_server.py &
	@echo "🧠 LoRA server restarted with latest adapter"

.PHONY: lora-pipeline
lora-pipeline: train-data train merge-lora deploy-lora ## Full MLOps pipeline: data → train → merge → deploy

# ── Maintenance ──────────────────────────────────────────────

.PHONY: preflight
preflight: ## Run preflight config validation
	@$(PYTHON) -c "from weaver_preflight import preflight_check; preflight_check()" 2>/dev/null || echo "⚠️  Preflight module not loaded (non-fatal)"

.PHONY: clean
clean: ## Kill all Weaver processes
	@pkill -f weaver.py 2>/dev/null || true
	@pkill -f twilio_weaver_bridge.py 2>/dev/null || true
	@pkill -f quantum_api.py 2>/dev/null || true
	@pkill -f health_dashboard.py 2>/dev/null || true
	@pkill -f lora_server.py 2>/dev/null || true
	@pkill -f arecord 2>/dev/null || true
	@pkill -f aplay 2>/dev/null || true
	@echo "🧹 All Weaver processes killed"

.PHONY: clean-vault
clean-vault: ## Reset Nexus_Vault (DESTRUCTIVE)
	@read -p "⚠️  This deletes all vault data. Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf Nexus_Vault/*
	mkdir -p Nexus_Vault/akashic_persist
	@echo "🗑️  Vault reset"

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
