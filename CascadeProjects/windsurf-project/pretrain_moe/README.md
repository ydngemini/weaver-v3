# pretrain_moe — From-Scratch MoE GPT, Multi-Session Free-Colab-T4 Pretrain

A complete, runnable kit that pretrains a **small Mixture-of-Experts (MoE) GPT
from random init** on open code+text corpora, across multiple **free Google
Colab T4** sessions, auto-resuming from a Google Drive checkpoint each time you
reconnect.

Three files do the work, plus this launch kit:

| File | Role |
|------|------|
| `model.py` | Architecture only. Random init, no pretrained weights, ever. Self-test: `python model.py`. |
| `prepare_data.py` | Streams 4 corpus shards, trains a 16k byte-level BPE, packs `train.bin`/`val.bin` (resumable cursor). |
| `train.py` | Multi-session AdamW loop + atomic Drive resume bundle. Smoke test: `python train.py --smoke`. |
| `colab_pretrain.ipynb` | The launch notebook — mounts Drive, installs deps, runs prep then train, **safe to re-run every session**. |
| `run_colab.sh` | The one-liner that drives a session from a shell (local sanity run or a Colab cell). |

---

## What this IS / what this is NOT — read this first (HONEST)

### What this IS
- A **from-scratch** pretrain. Every weight starts from random init; nothing is
  fine-tuned and no pretrained checkpoint is loaded (verified: obs #1903).
- A **small** MoE: **~202.9M total params / ~69.5M active per token** (9 layers ×
  640d, 8 experts top-2, ctx 1024, 16k vocab). It fits one **16 GB T4** with
  ~6.4 GB used and a deliberately large VRAM safety margin.
- A model that, on the planned **~6B-token** budget (~30× params, Chinchilla-plus
  for a small usable code model), will learn **code SYNTAX and simple
  completions**: balanced brackets/quotes, indentation, keyword and idiom
  patterns, short boilerplate, common library call shapes, and the difference
  between code and prose. It will produce locally-plausible tokens.
- A genuinely **multi-session** system: kill it at the ~12h Colab cap, reconnect,
  re-run the same notebook, and it resumes from the latest Drive checkpoint at the
  exact step / token-count / optimizer-state / RNG / LR-schedule position.

### What this is NOT
- **NOT a real coding assistant.** A ~70M-active model trained on ~6B tokens on
  one free T4 is a *toy-to-small* model. Expect it to autocomplete a `for` loop or
  close a bracket, **not** to implement a function from a docstring, reason about a
  bug, follow multi-step instructions, or write correct non-trivial programs. It
  has no instruction tuning, no RLHF, no tool use. Treat its output as "statistically
  plausible code-shaped text," not as correct code.
- **NOT frontier scale.** Frontier coding models are 100s of billions of params
  trained on **trillions** of tokens across large GPU clusters for weeks. Nothing
  here approaches that, and a single free T4 categorically cannot.
- **NOT an 8B-active model.** The "8" in this config is **8 experts**, never 8
  billion parameters. See the scale-up section below for why 8B-active is
  config-only but **hardware-impossible** on a T4.

### Honest expectations by budget

| Tokens trained | What you should realistically see |
|---|---|
| ~0.2–0.5B (1 session) | Loss drops off the `ln(16384)≈9.7` random-init plateau; output is byte-level mush settling into token-shaped fragments. |
| ~1–2B (a few sessions) | Recognizable code tokens, mostly-balanced brackets, plausible indentation, keyword runs. Still semantically incoherent. |
| ~6B (full budget, ~9 sessions) | Short, locally-coherent completions: finishes an obvious line, mirrors nearby style, closes delimiters. **Still not a coding assistant.** |

If you want a *useful* code model, this kit is the **learning/serving scaffold**,
not the destination — the destination needs the cluster described below.

---

## The winning config (and why)

```
9 layers × 640d   |  10 heads × head_dim 64   |  8 experts, top-2 routing
d_ff 1920 (3× n_embd, modest)  |  ctx 1024  |  vocab 16384 (trained byte-level BPE)
TOTAL ~202.9M   |   ACTIVE/token ~69.5M   |   VRAM ~6.4 GB of 16 GB on a T4
```

**Stance: maximize the VRAM safety margin.** On a free T4 across ~9 reconnects,
the real failure mode is not peak FLOP efficiency — it's an **OOM** or a **router
divergence** that burns a whole 12h session and corrupts the multi-session
schedule. This config sits mid-band on every axis on purpose:

- **Total 202.9M** sits well clear of the ~300M practical ceiling, leaving room
  for autocast graphs, flash-attn workspace, and eval-time logits spikes.
- **Active 69.5M** sits inside the 30–80M target band (MoE saves FLOPs per token,
  not memory — all 8 experts + their full fp32 AdamW state stay resident).
- **Tight 16k vocab** is the headroom lever: the `(B, T, vocab)` logits tensor is
  the single largest activation on a T4, so a small vocab both keeps that buffer
  small **and** keeps the tied tok-emb/lm-head at only ~10.49M params.

**Param arithmetic (verified, tied lm_head, head_dim 64, no biases on big mats):**
```
per layer = attn 4·640²                = 1.64M
          + 8 experts·(2·640·1920)     = 19.66M
          + router 640·8               = 0.005M   -> 21.30M / layer
×9 layers                              = 191.74M
+ tok_emb 16384·640 = 10.49M  + pos 1024·640 = 0.66M
                                       = 202.9M  TOTAL
active/token = attn 1.64M + top_2·(2.46M each = 4.92M) + router, ×9 + 10.49M tied head
                                       = 69.5M  ACTIVE  (mid 30–80M band)
```
(Self-test prints the live counts: `python model.py`.)

**VRAM budget (honest, all params resident, mixed-precision AdamW):**
```
weights bf16(2) + grad bf16(2) + fp32 master(4) + Adam m(4) + Adam v(4) = 16 B/param
  202.9M × 16 B  = 3.25 GB resident
activations @ B=48, T=1024, gradient-checkpointed:
  block-boundary inputs 0.57 GB + bf16 logits 1.61 GB + 1-block recompute ~0.75 GB ≈ 2.9 GB
MoE dispatch/combine @ capacity_factor 1.25 ≈ 0.15 GB
CUDA ctx + cuBLAS workspace + fragmentation + eval spikes ≈ 1.7 GB reserve
  TOTAL ≈ 6.4 GB of 16 GB  ->  ~9.6 GB free  (that margin IS the stance)
```

**Grafted router stack (for real expert utilization):** noisy top-2 gating
(annealed Gaussian noise for early exploration), Switch-Transformer
load-balancing aux loss (coef 0.01), router z-loss (1e-3, bounds gate logits),
and capacity_factor 1.25 with token-dropping to bound the dispatch buffers.

**Throughput / schedule (the 6B plan):** FLOPs/token ≈ 6·active ≈ 4.17e8. A T4 at
a conservative ~8 effective bf16 TFLOPS (~12% MFU after checkpoint/dataloader
stalls) ≈ 19.2k tok/s ≈ **~690M tok per ~10 usable compute-hours/session**.
**6B tokens ÷ 690M ≈ ~9 sessions.**

---

## Multi-session resume — how re-running is safe

Both stages are built to be **killed and re-run**:

- **Data prep** keeps a per-substream cursor (`data/cursor.json`, atomic
  `.tmp → os.replace`). Re-running `--build` *continues* packing where it stopped
  and never repeats or skips a shard across reconnects.
- **Training** writes a single atomic `.pt` bundle to Drive every `ckpt_interval`
  steps containing: model state, full AdamW state (m, v, step), AMP scaler,
  `global_step`, `token_count`, best-val, **all** RNG streams
  (torch/cuda/numpy/python), and config hashes. Written `.tmp → os.replace` so a
  mid-write Colab kill leaves the previous checkpoint intact.
- On launch, `train.py` auto-loads the latest **valid** bundle from `--ckpt-dir`,
  restores everything exactly, and (because the data sampler RNG is keyed by
  `(seed, step)`) the window stream is reproducible — session 3 resuming at step
  41,000 draws the same batches a single long run would. A mismatched-architecture
  hash is **refused** rather than silently corrupting the run.

So the entire multi-session protocol is: **run the notebook → it trains until the
12h cap kills it → reconnect → run the same notebook again → it resumes.** Repeat
~9 times until the 6B budget is met.

---

## Corpus (4 shards, streamed — never materialized to disk)

| Shard | Mix weight | Role |
|---|---|---|
| `bigcode/the-stack-v2-dedup` | 0.78 | Multi-language code spine (blob_id pointers → SH S3; skips cleanly if AWS creds absent). |
| `codeparrot/codeparrot-clean` | 0.62 | Python code spine (drops GPL). |
| `bigcode/starcoderdata` | 0.18 | Gated; `huggingface-cli login` with an accepted-ToS token. |
| `HuggingFaceFW/fineweb-edu` (sample-10BT) | 0.12 | Thin NL slice so the model can read/explain prose, not pure syntax. |

Weights are normalized at runtime. The 16k BPE is trained on a mix-weighted
*sampled* slice of the real corpus before packing, so the small vocab fits the
code-heavy distribution. (Shard licensing/provenance notes: see the vault note
`SYPHER_VAULT/00_Cortex/pretrain_corpus_shards.md`.)

---

## Quick start

### On Colab (the intended path)
1. Upload `colab_pretrain.ipynb` to <https://colab.research.google.com> (or open
   it from Drive / GitHub).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.** First cell mounts Drive and asks for auth; the rest
   installs deps, copies this kit to Drive, prepares data (resumable), and trains
   until the session is killed.
4. When Colab disconnects at the ~12h cap, reconnect and **Run all again** — it
   auto-resumes from the latest Drive checkpoint. Repeat ~9 times.

See **`run_colab.sh`** for the exact CLI invocation a single session runs, and the
**"How to launch"** section below for the full manual steps.

### Local smoke tests (no GPU needed, seconds)
```bash
python model.py            # param accounting + a forward/backward on random tokens
python train.py --smoke    # 12-step run -> checkpoint -> fresh-process resume 12->18
```
Both are verified to pass on `torch 2.11.0+cu130` (obs #1886).

---

## How to launch (manual steps — no free-Colab CLI exists)

> There is **no command-line / headless launcher for free Colab.**
> `colab.research.google.com` runs notebooks only, in a browser, behind an
> interactive Google sign-in. `gcloud colab` is **Colab *Enterprise*** — a
> *paid* Vertex AI runtime on a different (non-free-T4) backend, not the free tier
> this kit targets. So the launch is, by the platform's design, a browser action.
> The steps below are the real, complete launch procedure.

1. Get the kit onto Drive or GitHub so Colab can reach it. Either push
   `pretrain_moe/` to a repo, or upload the folder to
   `MyDrive/moe_pretrain/pretrain_moe/`.
2. Open `colab_pretrain.ipynb` in Colab (`File → Upload notebook`, or
   `File → Open notebook → GitHub`, or open it from Drive).
3. `Runtime → Change runtime type → Hardware accelerator: **T4 GPU**`.
4. `Runtime → Run all`. Approve the Drive-mount auth popup.
   - The notebook installs `torch`, `datasets`, `tokenizers`, `numpy`,
     `huggingface_hub`, `smart_open`, `boto3`.
   - (Optional) run `huggingface-cli login` in the auth cell for the gated
     StarCoder shard; (optional) set `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`
     for The-Stack-v2 content fetch. Both are skipped cleanly if absent.
   - `prepare_data.py` packs `train.bin`/`val.bin` into
     `MyDrive/moe_pretrain/data` (resumable).
   - `train.py` trains into `MyDrive/moe_pretrain/ckpts` (atomic resume bundles).
5. Let it run until Colab kills the session (~12h). **Reconnect and Run all
   again** — it resumes automatically. Repeat ~9 times until the 6B budget is met.

The same per-session command, runnable from a Colab cell or any GPU shell with
Drive mounted, is in `run_colab.sh`.

---

## 8B-ACTIVE scale-up — config-only, but NOT possible on a T4

The **same code** reaches a Mixtral-class **~8B-active / ~47B-total** model with a
**CONFIG-ONLY change** — keep 8 experts / top-2, but grow the shape:

```python
MoEGPTConfig(
    n_embd=4096, n_layer=32, head_dim=128, n_head=32,
    d_ff=14336, n_experts=8, top_k=2,
    vocab_size=65536, ctx_len=4096,
)   # ≈ 8B active / ≈ 47B total
```
Data side: bump vocab to ~64k, ctx to 4096+, and point `prepare_data.py` at the
**full** (non-sample) splits.

**Why this is hardware-impossible on one 16 GB T4 (not a tuning problem — a wall):**
- All 8 expert weights **plus their full fp32 AdamW optimizer state** must be
  resident on the device. MoE saves FLOPs per token, **not** memory — at ~47B
  total params, the optimizer state alone is hundreds of GB.
- It requires a **multi-GPU cluster** with **expert-parallel + tensor-parallel**
  sharding (Megatron / DeepSpeed-MoE), **ZeRO-3** optimizer-state sharding, and a
  token budget in the **trillions** (~15–20× params).

What changes for 8B-active: **only the config + the parallelism strategy.** The
architecture family, the router stack, the resume bundle, and the data pipeline
are unchanged. This kit is the small, honest, T4-runnable end of exactly that
spectrum — and it does not pretend to be the other end.

---

## File map / provenance
- Architecture + param accounting: `model.py` (verified, obs #1869).
- Data pipeline + resumable cursor: `prepare_data.py` (verified, obs #1874/#1900).
- Training loop + atomic Drive resume: `train.py` (smoke-verified live, obs #1886).
- No pretrained weights anywhere: confirmed obs #1903.
- Winner config rationale: mem obs #1848 ("B-core hardened with A+C grafts").

---

## Kaggle (headless CLI, free GPU)

Unlike free Colab (browser-only), **Kaggle Kernels run HEADLESS** and are driven
from the terminal via the `kaggle` CLI — so the entire multi-session loop can be
scripted. The corrected, **code-as-dataset** pieces live under `kaggle/` +
`run_kaggle.sh`; the free-Colab files (`run_colab.sh`) are untouched.

### Two facts that shape this design (proven on a live run)

1. **FORCE A T4.** Kaggle's torch is `2.10.0+cu128`, which has kernels for
   sm_70/75/80/86/90 but **NOT sm_60**. Kaggle may hand you a **P100 (sm_60)**,
   which then dies with *"no kernel image is available"*. Fix (validated): the
   train kernel-metadata sets `"machine_shape": "NvidiaTeslaT4"` **and** we push
   with `--accelerator NvidiaTeslaT4`. The train kernel also guards at runtime
   (loud warning if it sees no CUDA or a P100).
2. **A SCRIPT kernel runs only its single `code_file`** — sibling `.py` files
   pushed alongside are **not importable**. Fix: the project code ships as a
   Kaggle **dataset** (`weaver-moe-code`), and the kernels do
   `sys.path.insert(0, "/kaggle/input/weaver-moe-code")` before importing.

### Corpus (headless-safe — no AWS, no gated tokens)

The winner corpus can't run headless: **The-Stack-v2 rows are blob_id POINTERS**
(content needs AWS creds + a Software Heritage S3 agreement) and **starcoderdata is
GATED** (needs an accepted-ToU HF token). The `--corpus kaggle` preset in
`prepare_data.py` replaces them with **directly-streamable, non-gated** sources
(only `enable_internet=true` required):

| Source | Role | Weight |
|--------|------|--------|
| `codeparrot/codeparrot-clean` | Python code (PRIMARY) | 0.62 |
| `bigcode/the-stack-smol` | small permissive multi-language code (SECONDARY, optional) | 0.20 |
| `HuggingFaceFW/fineweb-edu` (`sample-10BT`, streaming) | thin NL slice | 0.12 |

Any source that 404s / needs auth is **skipped with a warning** and the prep
continues — it never hard-fails the whole build.

### The artifacts

| Path | Role |
|------|------|
| `kaggle/code_dataset/` | `dataset-metadata.json` + a copy of `model.py`/`prepare_data.py`/`train.py` → the **`weaver-moe-code`** dataset the kernels import from. |
| `kaggle/prep/` | `kaggle_prep.py` + `kernel-metadata.json` (CPU, internet on, source = `weaver-moe-code`). Packs `train.bin`/`val.bin`/tokenizer with `--corpus kaggle` into `/kaggle/working` → the **`weaver-moe-data`** dataset. Modest **first-run budget ~0.3B tokens** (env `KAGGLE_PREP_BUDGET_B`) so it finishes in one CPU session; resumable across pushes. |
| `kaggle/train/` | `kaggle_train.py` + `kernel-metadata.json` (GPU, **`machine_shape: NvidiaTeslaT4`**, internet on, sources = `weaver-moe-code` + `weaver-moe-data` + `weaver-moe-ckpt`). Prints GPU info first, guards against P100, resumes from the mounted ckpt (or fresh), runs `train.py --storage kaggle --max-hours 11.5`. |
| `run_kaggle.sh` | CLI launcher. **Dry-run by default** — PRINTS every quota-spending `kaggle` command; runs for real only with `RUN=1` (or `--run`). |

### How the resume loop works

- A GPU kernel runs headless up to **~12h**, then is **hard-killed**. The wall-clock
  guard stops cleanly at **~11.5h** (override env `KAGGLE_MAX_HOURS`), writes a fresh
  atomic bundle to `/kaggle/working/ckpts`, and **exits 0** before the kill.
- The latest checkpoint lives in the **`weaver-moe-ckpt`** dataset, mounted
  read-only at `/kaggle/input/weaver-moe-ckpt`. The train kernel reads it on
  startup (or starts fresh if empty). The launcher pulls the kernel output and
  pushes it as a **new version** of that dataset, so the next run resumes EXACTLY
  where it stopped.
- The packed data lives in **`weaver-moe-data`** (mounted read-only) — prepared
  **once** by the prep kernel, never re-tokenized.

### One-time setup

1. Create a Kaggle account and **phone-verify it** — *GPU access requires phone
   verification*. Without it, kernels run CPU-only.
2. Get an API token: Kaggle → **Account → Create New API Token** → save
   `kaggle.json` to `~/.kaggle/kaggle.json`, then `chmod 600 ~/.kaggle/kaggle.json`.
   (The kaggle binary used here is `$HOME/.venvs/kaggle/bin/kaggle`; override with
   `KAGGLE_BIN`.)

### The exact ordered command sequence (first real session)

`run_kaggle.sh` is **dry-run unless `RUN=1`** — every command below prints first so
you can read it; nothing touches your account until you opt in. Order matters: the
train kernel lists all three datasets in `dataset_sources`, so **all three must
exist before the first `train` push** (code → data → ckpt).

```bash
KB="$HOME/.venvs/kaggle/bin/kaggle"   # the kaggle binary run_kaggle.sh uses

# --- ONE-TIME BOOTSTRAP (in this order) ---

# 1. ship the project code as a dataset (kernels can't import sibling .py)
RUN=1 ./run_kaggle.sh bootstrap-code

# 2. pack the data ONCE on a CPU kernel; then make the weaver-moe-data dataset
RUN=1 ./run_kaggle.sh prep                       # pushes the CPU prep kernel
RUN=1 ./run_kaggle.sh status --kernel prep       # poll until COMPLETE
RUN=1 ./run_kaggle.sh output --kernel prep       # pull train.bin/val.bin/tokenizer
#   stage the pulled output + a dataset-metadata.json (id: .../weaver-moe-data):
mkdir -p ./kaggle_data_dataset
cp -rf ./kaggle_prep_output/* ./kaggle_data_dataset/
"$KB" datasets init -p ./kaggle_data_dataset      # then edit id -> weaver-moe-data
RUN=1 ./run_kaggle.sh prep                        # re-run: the create step makes the dataset
#   (the 'prep' subcommand's final step is `datasets create -p kaggle_data_dataset`)

# 3. create an EMPTY checkpoint dataset (first train run is fresh random-init)
RUN=1 ./run_kaggle.sh bootstrap-ckpt              # init + edit id -> weaver-moe-ckpt, then create

# --- PER-SESSION GPU CYCLE (repeat until the token budget is met) ---

# 4. push the GPU train kernel — FORCED T4 (--accelerator NvidiaTeslaT4)
RUN=1 ./run_kaggle.sh train

# 5. poll until the run reports complete (or error)
RUN=1 ./run_kaggle.sh status                      # default --kernel train

# 6. pull the train kernel version output (the fresh /kaggle/working/ckpts bundle)
RUN=1 ./run_kaggle.sh output                      # default --kernel train

# 7. stage the pulled checkpoint, then version weaver-moe-ckpt from it
cp -f ./kaggle_train_output/ckpts/*.pt ./kaggle_ckpt_dataset/
RUN=1 ./run_kaggle.sh version-ckpt

# 8. GOTO 4 — the next train push mounts the new ckpt version; train.py resumes exactly.
```

`./run_kaggle.sh resume-cycle` prints this whole loop.

### Honest wall-clock

- **~12h per session** (hard kill); the guard stops cleanly at ~11.5h and
  checkpoints, so you always end resumable.
- **~30h free GPU/week** → roughly **2–3 sessions/week**.
- The data prep is a **modest first-run budget (~0.3B tokens)** so it finishes in
  one CPU session, then training is **open-ended resume**: raise the budget later
  and re-prep (resumable), or just keep training on the packed `.bin`. Every GPU
  session ends with a clean atomic checkpoint, so pausing for days costs nothing.
- **GPU requires phone verification** on your Kaggle account.
