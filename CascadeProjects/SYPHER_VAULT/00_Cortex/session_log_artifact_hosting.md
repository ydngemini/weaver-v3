# Artifact Hosting — 2026-06-14

## GitHub
- Pushed branch `feat/oracle-arm-free-deploy` → origin (commit a5a2fb3).
- 32 files: MoE pretrain kit (`pretrain_moe/`), `discord_bridge.py`, dashboard/bridge updates.
- `.gitignore` expanded: `*.pt/*.bin/*.gguf/*.safetensors`, `3b/`, `pretrain_moe/kaggle_*_output/`, `.claude/`, `__pycache__/` — keeps 12GB of binaries out of git.

## Hugging Face (heavy files relocated here — "like a repo")
- Auth gotcha: first fine-grained token had empty perms; re-issued with `repo.write`.
- [[weaver-v3]] model repo (public): `model.safetensors` 6.17GB + `gguf/weaver_v3_Q4_K_M.gguf` 1.93GB + tokenizer/config.
  - https://huggingface.co/ydngemini/weaver-v3
- [[weaver-moe-pretrain]] repo (public): `ckpt_latest.pt` + `ckpt_step000000006.pt` (2.44GB each).
  - https://huggingface.co/ydngemini/weaver-moe-pretrain
- Upload via `hf upload` over XET (resumable). All RC=0.

Related: [[pretrain_corpus_shards]]

# MoE Train Session 1 — COMPLETE (2026-06-15 01:39 UTC)
- Graceful wall-clock-guard exit at 11.52h (rc=0), NOT a hard kill.
- **16.4k tok/s** fp16 on T4 (beat 10-13k est; ~5x old bf16 3.3k).
- Reached step **1706/15258**, **670.8M/6000M tokens** (~11.2% of epoch).
- lm loss 2.45->2.33 declining; best_val 2.4729; aux ~2.12, z declining.
- Output ckpts: ckpt_latest.pt, ckpt_step000001500.pt, ckpt_step000001706.pt.
- Resume design (kaggle_train.py): walks /kaggle/input for any .pt -> resume; train.py find_latest_checkpoint prefers ckpt_latest.pt -> restores global_step exactly.
- SESSION 2 PLAN: chain session-1 train kernel as kernel_source (auto-discovers ckpt, no 2.3GB local pull) instead of versioning weaver-moe-ckpt dataset. ~12 sessions for full epoch at this rate.

# MoE Train Session 2 — LAUNCHED (2026-06-15 ~02:3x UTC)
- Resume path chosen: dataset-version (NOT self kernel-chain — Kaggle forbids a kernel sourcing its own output, and silent fresh-init can't be caught mid-run).
- Pulled session-1 ckpt_latest.pt (2.44GB) via output API presigned URL (per-file, avoided 7GB all-ckpt pull); valid torch zip.
- Versioned weaver-moe-ckpt -> v2 (totalBytes 2,435,003,290 = ckpt+README). Verified by construction: only .pt in /kaggle/input.
- Train kernel v4 pushed -> session 2 RUNNING. Resumes at step 1706 (find_latest_checkpoint prefers ckpt_latest.pt -> restores global_step).
- Watcher task b4bnzz5nn polling to terminal state.
- POST-RUN VERIFY: log must show "[resume] found ... resuming" + step starting ~1706 (not 0).

# Distillation Pipeline Seeded (2026-06-15)
- pretrain_moe/distill/: gen_agentic_traces.py -> agentic_traces.json (10 seed traces, probe->trap->self-heal, <|call:tool|> syntax; 2 grounded in real Weaver incidents: Akashic SO_REUSEPORT 9999 double-bind, 99%-disk ENOSPC train.bin recovery).
- sft_train.py: SFT on the base ckpt, prompt-masked loss (ignore_index=-1), same bundle format as train.py (inference/GGUF-merge compatible). fp16 on T4. Masking boundary unit-verified (prompt p-1 masked, completion+eos supervised).
- HONEST CONSTRAINT: 203M/6B-token base is GPT-2-class. Cannot beat Opus/235B at SWE (SWE-bench is frontier-only; 32B coders ~30-40%) or general agentic. Winnable target = format-correct tool-calling on Weaver's FIXED tool set at ~16ms/$0 + small-as-fast-tier with Opus escalation. Real lever = thousands of API-generated traces, not epochs.
- NEXT: (a) swap TRACES literal for Opus-API generation loop -> agentic_traces.jsonl; eval harness (BFCL subset + held-out) scoring tool-call acc + latency/cost vs Opus.

---

## 2026-06-15 — Eval harness + subscription-only generation constraint

- **User constraint (durable):** never spend paid LLM API budget; generate training data via the
  Claude Code subscription (main session / subagents / Workflow fan-out). Saved to auto-memory as
  `no-paid-api-use-subscription`. Reframes the "Opus/235B API generation loop" → subscription subagents.
- **Built `distill/eval_toolcall.py`** — tool-calling eval harness for the SFT model. Metrics:
  format validity, first-tool accuracy, ordered sequence match (LCS), multiset-F1 over tool names,
  arg key-Jaccard + exact-match; plus latency + $/1k-traces delta vs Opus (prices via CLI, nothing
  hard-coded). Pure-Python scoring layer (GPU-free, 6-check `--smoke` PASS); torch only for generation.
  Reuses sft_train's chat template (`<|user|>…<|assistant|>…<EOS>`) and the `<|call:NAME|> {json}` syntax.
- **Honesty fix:** held-out rows from `--holdout` are NOT auto-excluded from SFT (sft_train trains on
  the whole file) — README + docstring now say to train on the complement or use `--eval-file`.
- **Kaggle Session 2** (`nathaniellockwood/weaver-moe-train`): status `running` — pretraining still in progress.
- **Next:** wire subscription-based trace generation (subagents emit batches → `agentic_traces.jsonl`),
  keep a clean held-out shard for eval.

### Subscription trace-generation loop — proven (wave 1)

- **Built `distill/merge_traces.py`** — deterministic gate: validates each subagent-authored trace
  (non-empty prompt+completion, ≥1 well-formed `<|call:NAME|> {json}`), dedups by normalized-prompt
  SHA1, appends to `agentic_traces.jsonl`. `--stats` reports trace/tool coverage.
- **Wave 1:** 3 parallel `general-purpose` subagents (Claude subscription, NOT API), distinct domain
  clusters (backend/distributed, devops/infra, data+ML+app) → 12 traces each = 36, merged + 10 seeds.
- **Corpus now: `agentic_traces.jsonl` = 46 traces, 98 distinct tools.** Cross-checked through
  `eval_toolcall.parse_tool_calls`: 116 tool calls, 3–4/trace, **0 malformed JSON args**.
- Cost ≈ 38k subagent tokens / 12-trace batch (~3.2k tok/trace incl. reasoning). Scaling to a real
  SFT set (hundreds→thousands) is N more waves — pending user's target/budget.

## SFT trace gen — wave_8 (Container Networking & DNS)
- Wrote 14 diverse agentic tool-calling traces to `/tmp/gen/wave_8.json` (probe -> trap -> self-heal format).
- Angle: common weekly production K8s net/DNS incidents. 29 unique snake_case tools, 42 total calls, 2-4/trace.
- Every trace has a genuine first-response trap (errors + two misleading-success traps: empty endpoints w/ valid ClusterIP, ExternalName already-correct-but-pooled-sockets).
- Validated: valid JSON array, exact keys {domain,prompt,completion}, one-line JSON args (nested objs allowed), matched call/response counts, 14 distinct prompts.
- Related: [[Pretrain MoE Workflow]]

## SFT trace gen — wave_14 (Disk/Filesystem/ENOSPC)
- Wrote /tmp/gen/wave_14.json: 14 diverse agentic tool-calling traces, probe->trap->self-heal format.
- Angle: common weekly production incidents. Domain cluster: disk/ENOSPC.
- 30 distinct snake_case tools (e.g. check_disk, check_inodes, list_deleted_open_files, du_breakdown, reset_index_block, docker_prune_volumes, set_prometheus_flag).
- Key trap patterns covered: inode exhaustion (df lies), deleted-but-open fd, wrong mount (data dir vs /), TMPDIR/innodb_tmpdir spill to tiny /tmp, docker volumes vs images, Kafka per-topic retention override, ES latched read_only_allow_delete, k8s imagefs DiskPressure, Prometheus missing retention.size, core-dump crash loop.
- See [[Pretrain MoE Workflow]] for the broader tool-model training effort.

## 2026-06-16 — DATA LOSS postmortem + pipeline fix + pilot regen

- **Lost:** waves 2–17 (the 4:51–5:23am Jun-15 marathon: Kafka, K8s SRE, container net/DNS, TLS,
  LBs, CI/CD, Cloud IAM, autoscaling, disk/ENOSPC, systemd, Terraform, ETL/Airflow…) were staged to
  `/tmp/gen/wave_N.json` and **never run through `merge_traces.py`**. `/tmp` was reclaimed; the dir is
  gone, no backups, no stash. ~250 validated traces evaporated. Only Wave 1's 36 (+10 seeds) survived
  in `agentic_traces.jsonl` because that wave was merged. (Confirms obs #2726.)
- **Root cause:** ephemeral staging + deferred merge. Generation wrote to `/tmp`; the persistent corpus
  only updated for Wave 1. Anything not merged before the next `/tmp` cleanup is unrecoverable.
- **Fix (durable):** generate into repo-internal `distill/waves/` and run `merge_traces.py` **immediately**
  after each wave. Never stage trace output in `/tmp` again. Memory: [[distill-stage-in-repo-not-tmp]].
- **Surviving work committed** (`266e6f4`): 46-trace corpus + merge/eval/sft pipeline, now version-tracked.
- **Pilot regen (3 waves, subscription subagents, NOT API):** k8s_sre (14 traces / 22 tools / 8 misleading-
  success traps), cicd (14 / 27 / 5), disk_enospc (14 / 46 / 12). Independent validation + `merge_traces`
  gate: **+42 added, 0 dup, 0 invalid → corpus = 88 traces, 88 unique prompts, 190 distinct tools.**
- **NEXT (pending user):** finish regen of the remaining lost domains (~13 waves) the same way, OR proceed
  to SFT/eval on the 88-trace corpus. Related: [[Pretrain MoE Workflow]]
