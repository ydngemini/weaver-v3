# Session Log — Distill: 7 owed waves merged + semantic audit

**Date:** 2026-07-03
**Branch:** `feat/aws-graviton-deploy`  **Commit:** `7c3307a`
**Area:** [[Weaver v3]] MoE SFT distillation corpus (`windsurf-project/pretrain_moe/distill/`)

## What happened

Resumed the distill regen work stalled by the 2026-06-27 session-limit interrupt. The 7 owed enrichment domains had been *generated* (waves written to `distill/waves/` on Jun 27, 20:37-21:00) but never merged — `agentic_traces.jsonl` was still at 270.

1. **Validated + merged** the 7 waves (secrets_vault, grpc_streaming, postgres_locks, redis_cluster, oauth_tokens, rate_limiting, prometheus_alerting; 14 traces each). `merge_traces.py`: **+98 added / 0 dup / 0 invalid -> 368 traces / 368 unique prompts / 634 distinct tools.**
2. **LLM semantic audit** (4 parallel subagents, probe->trap->self-heal rubric): **0 HIGH, corpus safe to keep.** 7 MED quality items (correct fix direction, shaky prose/tool-naming). `postgres_locks` t1 (`balance/16` split loses remainder -> money-losing) is the only one teaching a wrong *operation*. Cross-cutting: the "first green signal is the trap" archetype is formulaic across nearly all 98 completions.
3. **Committed** corpus + 7 waves + `validate_traces.py` (the stdlib structural gate) as `7c3307a`. Did **not** push / open PR — this is the user's active deploy branch. Pre-merge 270-row backup saved to the job tmp dir.

## Owed next

- Optional MED polish pass on the 7 flagged traces (fix needs editing both `waves/*.json` and the matching `.jsonl` line — merge dedups by prompt-hash). pg t1 is the priority.
- SFT train (`sft_train.py`, fp16 on T4) + eval (`eval_toolcall.py`) with a held-out shard — needs a T4 (Colab/Kaggle).

Follows the [[distill-stage-in-repo-not-tmp]] rule: generate -> validate -> merge immediately -> commit waves + jsonl together.
Related: [[pretrain-moe-workflow]], [[no-paid-api-use-subscription]]


## Wave authored: wave_s3_object_storage.json (2026-07-04)
- 14 traces, [[distill]] SFT corpus, domain `s3_object_storage/*`. validate_traces.py -> OK.
- Covers: bucket-policy explicit-Deny precedence, SSE-KMS key-policy decrypt, presigned-URL STS session expiry, incomplete-MPU billing, delete-marker recovery, Glacier restore-tier, CORS preflight origin, CRR+KMS FAILED, Requester Pays header, Object Lock GOVERNANCE bypass, account-level BPA RestrictPublicBuckets, lifecycle empty-Prefix, VPC gateway-endpoint policy, 403-masking-404 (missing ListBucket).
- Every trace has one detected misleading probe; distribution resp1=5 / resp2=8 / resp3=1; 42 distinct S3/AWS tools.
- Links: [[Weaver v3]] [[distill]]


## 2026-07-04 — wave_oom_memory_cgroups authored
Authored `waves/wave_oom_memory_cgroups.json` — 14 SFT tool-calling traces on Linux memory/OOM/cgroup debugging for the [[distill]] corpus. Passed `validate_traces.py` (OK). 35 distinct Linux-appropriate tools; every trace carries a misleading System Response (free-vs-available, memcg-kill-despite-free-host, JVM RSS>Xmx off-heap, glibc arenas, THP fork blowup, tmpfs cgroup charge, python fork COW, PSI reclaim thrash, negative-dentry slab, mmap max_map_count, cgroup swap.max=0, sidecar global-OOM victim, oom_score_adj shielded hog, hugetlb leftover reservation). Trap position spread: 7 at resp1 / 5 at resp2 / 2 at resp3. Relates to [[distill-regen-resume-state]].

## 2026-07-04 — 8 more waves merged (480 total) + 2nd audit pass, 2 systemic HIGH bugs found+fixed

Generated and merged the next 8 enrichment domains: s3_object_storage, rabbitmq_queues, nginx_proxy, docker_builds_registry, elasticsearch_ops, mysql_replication, oom_memory_cgroups, websocket_realtime (14 traces each). `merge_traces.py`: **+112 added / 0 dup / 0 invalid -> 480 traces / 852 distinct tools.**

**Generation was rocky** — every one of the 8 generator subagents died at least once on a Claude session-limit reset mid-task; each was resumed from its saved transcript rather than restarted, which worked cleanly (transcripts preserved full context, no wasted regen).

**LLM semantic audit (4 parallel subagents, same probe->trap->self-heal rubric) found a genuine systemic bug, not just prose nits:** 3 traces across 2 different domains/generator-agents had a **Resolution claiming a bulk/high-stakes fix (data resync, IAM key rotation, cache TTL change) backed by ZERO tool call** — the model would learn to assert "fixed and verified" without ever taking the action. Root cause (self-reported by the mysql_replication agent): an automated char-budget trim pass silently deleted the actual repair/verification call while leaving the Resolution's claim text untouched.

Confirmed + fixed by inserting a genuine backing call+response before each Resolution (same technique across all 7 instances, in both the wave JSON and the already-merged `agentic_traces.jsonl` — merge dedups by prompt-hash so re-merging does NOT overwrite existing rows, had to sync those 7 corpus lines directly):
- `mysql_replication` t4, t9: claimed `pt-table-sync` repaired diverged/missing rows after only a preventive `binlog_format=ROW` flip — no repair call existed. Added `pt_table_sync` + confirming response.
- `mysql_replication` t5: claimed semi-sync "verified active end to end" after only diagnosing the misconfig — no enable call existed. Added `mysql_set_global` enabling `rpl_semi_sync_replica_enabled` + confirming response.
- `mysql_replication` t11: claimed a specific lag-to-zero endpoint state with no re-check call. Added a `mysql_show_replica_status` verification call (this one legitimately needed only a *verification* call, since "wait it out" was the correct fix).
- `s3_object_storage` t9 (object-lock-governance): claimed "9,412 versions deleted" after only ONE sample `aws_s3api_delete_object` bypass call. Added an `invoke_lambda` bulk-rerun call + confirming response.
- `docker_builds_registry` t4 (deleted-secret-in-layer-history): claimed "AKIA2E7NVXQ4 rotated in IAM" with no rotation call. Added `rotate_iam_key` + confirming response.
- `docker_builds_registry` t11 (pullthrough-mirror-stale-base): claimed "mirror TTL lowered to 24h" with no config call. Added `edit_mirror_config` + confirming response.

0 remaining HIGH findings after the fix pass. ~10 MED findings logged as optional follow-up polish (not blocking): rabbitmq t2 + several elasticsearch_ops traces cite specific post-fix metrics (throughput numbers, shard counts) never re-verified by a tool call — a milder version of the same pattern, but the *core* fix action in every one of those is genuinely backed, only the decorative numbers are unverified.

**Lesson for future wave generation:** any subagent doing a post-hoc "shrink completions to fit the char budget" pass must NOT touch call/response pairs — trim only prose in Reasoning/Recovery-reasoning lines. Worth stating explicitly in the next generation prompt.

Not yet committed as of this log entry — see [[distill-regen-resume-state]] for current corpus state and next steps.

## 2026-07-04/05 — weaverv3.com: penthouse rebuilt + her cloned voice live

[[Weaver v3]] embodiment milestones, all on `feat/aws-graviton-deploy`:
- **Penthouse** (`0f860f2`): `avatar/build_penthouse.py` — fully procedural Blender 4.5.9 headless build of `weaver_apartment.glb` (glazing+mullions, generated night-skyline/marble/walnut textures packed in-GLB, 153 meshes, 3.1MB), sized to embodiment.html's runtime-atmosphere bounds, composed for its fixed camera. Iterated via live Playwright screenshots (fixed: plane-UV corner-order smearing textures diagonally; light wattages blowing out under the page's 60-intensity clamp). Old scene → `weaver_apartment_v1_backup.glb` in the avatar bucket. Blender lives at `SYPHER_CORE2/tools/`.
- **Voice** (`e7eee01`+`5d73e6a`): OpenVoice v2 clone-once pipeline serving at `/tts/synth` (key-gated) from the box; page plays her cloned voice, browser TTS fallback. Embedding extracted once from the 12.9s ref → saved `.pth`; per-line sha1 disk cache. Measured: cached 0.3-0.8s / fresh ~12s (RTF ~2.9, 2 Graviton vCPUs). Three prod bugs found+fixed en route: numpy/PyAV pin rot (py3.10 venv + `--no-deps` + direct `extract_se()`), 0-byte poisoned cache entries (atomic writes), and **systemd MemoryHigh reclaim-hell masquerading as a network hang** (759k high-breaches, D-state at 2200M ceiling; raised to 3200M).
- Also: silent speechSynthesis failure fixed (`e1418ea`), Seymore creds rotated in.

Related: [[weaverv3-live-architecture]] memory note has the operational detail.
