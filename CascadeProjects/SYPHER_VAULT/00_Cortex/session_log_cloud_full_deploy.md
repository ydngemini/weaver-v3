# Session — full Weaver + Oracle cloud deploy artifacts (2026-06-17)

Extended the [[Oracle ARM]] headless kit (commit 6fd035a) toward a full public deploy:
headless Weaver brain + Oracle frontend/backend, public Caddy TLS, **local llama.cpp** experts.

## Key constraints discovered
- **VTV can't run on a cloud VM** (needs real mic/cam/speaker) → cloud = `weaver.py --headless`. Every other lobe runs.
- **Oracle UI ≠ wired to Weaver.** `Oracle/backend/server.py` `/ws` serves a `generate_mock_record` loop; never touches Nexus Bus 9999. Real bridge = deferred Phase 6.
- **Oracle backend is a full SaaS** (asyncpg/RLS, Stripe, AWS RDS IAM), BUT `lifespan` (server.py:60-64) **tolerates a DB-less boot** → deploy in **demo mode** (UI + mock stream), no Postgres needed. Real persistence opt-in via `ORACLE_DB_*`.

## Files created/edited (local; nothing run on a box yet)
- New: `deploy/weaver-llm.service` (llama_cpp.server :8090, alias `weaver-local`), `deploy/oracle-backend.service` (uvicorn :8000, optional EnvFile), `deploy/Caddyfile` (proxy `/api|/auth|/billing|/health|/ws`→8000, SPA from `/var/www/oracle`), `deploy/env.oracle-backend.example`, `deploy/setup_oracle_extras.sh` (GGUF+backend venv+frontend build+Caddy+units).
- Edited: `deploy/env.oracle.example` → default `WEAVER_LLM_BACKEND=local`; `deploy/weaver.service` → `Wants/After=weaver-llm`; `deploy/README_ORACLE.md` → full-deploy section; `Oracle/backend/auth.py` → hardcoded demo logins now gated behind `ORACLE_ENABLE_DEMO_LOGINS` (off by default → not shipped public).
- Frontend prod env baked inline at build (`VITE_WS_URL=wss://$ORACLE_HOST/ws`, `VITE_API_BASE=https://$ORACLE_HOST`) — avoids the `envDir:'..'` quirk.

## Blocked on user
Phase 0 (provision Oracle A1, give IP + SSH key + pick `<ip>.sslip.io`). Then rsync/ssh execute.

Note: correct cortex path is THIS one (`weaver v3/CascadeProjects/SYPHER_VAULT/...`); the distill-CI log this session went to the wrong top-level `SYPHER_VAULT` again.
Plan: [[shiny-weaving-lobster]]. Related: [[t4-bf16-no-tensor-cores]].

---

## 2026-07-06 — Experts moved to the AWS Mantle gateway (best available models)

Wired a **`mantle` LLM backend** into `slm_experts.py` (commit `b28b755`) and flipped the live box brain onto it. The 5 MoE experts (logic/emotion/memory/creativity/vigilance) now run **DeepSeek V3.2** via the **AWS-hosted OpenAI-compatible Bedrock gateway** `https://bedrock-mantle.us-east-1.api.aws/v1`, authenticated by the MantleApiKey from OpenCode's config (`~/.local/share/opencode/auth.json`).

- **Why the gateway, not native Bedrock:** the MantleApiKey only works against this gateway — native `boto3 converse()` returns `Operation not allowed` (its account has no native-Bedrock model access; the gateway sidesteps that). See [[mantle-gateway]].
- **Model choice:** DeepSeek V3.2 is the strongest model that works via chat-completions on the gateway and is fast (~0.8–1.5s/expert). Anthropic Claude (Opus 4.8 / Sonnet 5 / Fable 5) is on the gateway but **Messages-API only** — not reachable through the OpenAI-compat client.
- **Verified live:** `weaver.service` active, all 5 experts loaded, pipeline logs show repeated `POST bedrock-mantle…/v1/chat/completions → 200 OK` with `Gate decision → vault (3 experts)`.
- **Config on box** (`.env`, 600, untracked): `WEAVER_LLM_BACKEND=mantle`, `WEAVER_LLM_MODEL=deepseek.v3.2`, `WEAVER_LLM_URL=…`, `MANTLE_API_KEY=…`.
- **Still separate:** the weaverv3.com front-page `/llm` conversation is still the on-box **1B llama** — a distinct path from the MoE experts.

Related: [[weaver-brain-status]] · [[weaverv3-live-architecture]] · [[aws-account-topology]]

**Update — hybrid resilience:** per YDN's choice, the experts now run **DeepSeek V3.2 (Mantle) primary + automatic on-box llama fallback** (commit `8d18106`) — her pre-existing model stays in active use and she never goes dark if the gateway hiccups. After 3 consecutive Mantle failures the lobe skips the gateway for 45s. Proven under a forced outage (dead-host Mantle URL → `weaver-local` answered). Her **Soul Voice LoRA (:8899)** remains the final voice on every reply. Live-verified after restart: service active, 5 experts loaded, gateway 200s flowing, Soul Voice + quantum + nexus + n8n all healthy.

---

## 2026-07-06 — "Everything on the AWS box" — RunPod + Azure eliminated

Full sweep to pull every non-AWS runtime dependency onto the box (`i-01e15a540b9efb7a0`).

- **Voice → on-box.** Her cloned voice already ran on the box as **OpenVoice v2** (`weaver-tts.service`, `127.0.0.1:8092`) — RunPod's XTTS pod was returning 404 (broken). Repointed Caddy `/tts` from `f6t7m2t1errly1-8888.proxy.runpod.net` → `127.0.0.1:8092`. **RunPod eliminated.** Live-verified: `https://weaverv3.com/tts/synth → 200`, valid 22 kHz WAV. Latency ~10–16 s/sentence on ARM CPU (RunPod GPU was ~0.9 s) — streamed sentence-by-sentence so she starts speaking after the first.
- **n8n experts → off Azure.** The n8n orchestrator's 5 expert lobes + Self-Reflect node pointed at Azure `gpt-5.4-nano`. Retargeted all 6 to the AWS Mantle gateway (`deepseek.v3.2`) in the workflow source (commit `123ae1b`). The n8n pipeline is **dormant** (webhook 404 / not active) and the container can't reach the loopback-bound on-box llama, so Mantle (reachable + best) is the target; auth stays via the n8n credential (no secret in git). Bringing it live = swap that credential to the Mantle key + activate (follow-up).
- **Already on-box/AWS:** experts primary = Mantle DeepSeek (AWS) + on-box llama fallback; front-page `/llm` = on-box llama; Soul Voice LoRA `:8899`; quantum = on-box Aer; nexus/akashic/quantum_api/dashboards; n8n in Docker. **Off/dormant non-AWS:** Gemini vision, OpenAI/Twilio, Discord (all disabled); IBM quantum (opt-in only).
- **The one thing not *literally* on the box:** DeepSeek can't run on a t4g CPU — the experts' primary reasoning is the AWS-hosted Mantle gateway, called from the box, with on-box llama fallback. Fully-on-box option = the 1B llama only (weaker).

Related: [[weaver-brain-status]] · [[weaverv3-live-architecture]] · [[mantle-gateway]] · [[aws-account-topology]]
