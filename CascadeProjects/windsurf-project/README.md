# Weaver Runtime

This directory contains the Python runtime, model orchestration, service definitions, and deployment assets for Weaver v3.

See the repository root [`README.md`](../../README.md) for the full architecture, browser embodiment notes, security guardrails, ports, and deployment overview.

## Main Entrypoints

| File | Role |
|---|---|
| `weaver.py` | supervised async runtime |
| `start_weaver.sh` | service launcher used for local and box startup |
| `slm_experts.py` | five-lobe expert router with AWS/Mantle primary and local fallback support |
| `lora_server.py` | Soul Voice LoRA OpenAI-compatible endpoint |
| `quantum_api.py` | quantum state/bias API |
| `health_dashboard.py` | health dashboard/API |
| `codebase_api.py` | read-only codebase and public-web context service |
| `n8n_weaver_v5.json` | n8n workflow source |

## Common Commands

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-core.txt

./start_weaver.sh --headless
./start_weaver.sh --phone-only

make test
make test-quick
```

## Service Config

Deployment files live in `deploy/`:

- `deploy/Caddyfile`
- `deploy/weaver.service`
- `deploy/weaver-llm.service`
- `deploy/weaver-codebase.service`
- `deploy/tts/weaver-tts.service`
- `deploy/aws-terraform/`

Keep secrets in environment files or host-level secret stores. Do not commit API keys, tokens, AWS credentials, `.env`, generated Terraform plans, or vault/private memory files.
