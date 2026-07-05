# Weaver MoE SFT — GPU training job (oncompute.ai C2D, or any GPU)

Packages `sft_train.py` as a self-contained containerized GPU job that fine-tunes
the pretrained Weaver MoE base on the **480-trace distill corpus** and emits the
trained checkpoint as its downloadable output. Shaped for **oncompute.ai**
Compute-to-Data (matches the format of their `Decoder Fine-Tuning` tutorial), but
the same image runs on any NVIDIA GPU unchanged.

## What's baked in vs supplied at run time
- **Baked into the image:** the corpus (`agentic_traces.jsonl`), the tokenizer,
  `model.py`, and `sft_train.py`. All small.
- **Supplied at run time:** the **base checkpoint** (`ckpt_latest.pt`, ~2.3 GB,
  gitignored). It's the model to fine-tune *from*.

## Run on oncompute.ai
oncompute is **wallet-native and human-driven** — jobs are launched from the
**Ocean Orchestrator** plugin (VS Code / Cursor / Windsurf / Antigravity) or
`ocean-cli`, and paid with your wallet / grant tokens. An agent can't submit for
you (no access to your private key). Steps **you** run:
1. Publish `kaggle/ckpt_dataset/ckpt_latest.pt` as the job's **input dataset**.
2. Publish/point the **algorithm** at this container image (build + push, or the
   Orchestrator builds from this folder).
3. Launch the C2D job from the Orchestrator; the entrypoint finds the ckpt under
   `/data/inputs/`, trains, and writes the result to `/data/outputs/` for download.

## Run on any GPU box (RunPod / Lambda / a g4dn / Colab)
```bash
cd CascadeProjects/windsurf-project/pretrain_moe
docker build -f distill/oncompute_job/Dockerfile -t weaver-sft:latest .
docker run --gpus all \
  -v /path/to/ckpt_latest.pt:/data/inputs/base.pt \
  -v $PWD/sft_out:/data/outputs \
  -e BASE_CKPT=/data/inputs/base.pt -e EPOCHS=3 -e LR=2e-4 \
  weaver-sft:latest
# trained checkpoint lands in ./sft_out/
```

## ⚠️ Verify the base checkpoint FIRST
`sft_train.py`'s own honest note: fine-tuning from an **undertrained** base
produces a model that learned nothing — you'd burn GPU credits for garbage. The
local checkpoints include a `ckpt_step000000006.pt` (6 steps), and the MoE
pretraining was compute-ceiling-limited. Before spending a real run, load
`ckpt_latest.pt` and confirm its `step`/`loss` reflect actual training (the job
prints these). If the base isn't real yet, **pretrain more first** — the SFT
corpus is ready and waiting either way.

## Scope
This is the **training** half of "GPU on oncompute" — the only half that fits a
batch C2D network. Her **live voice/brain** (real-time `/tts` + `/llm`) **cannot**
run on oncompute (no persistent endpoint); that needs a dedicated GPU (AWS g4dn /
RunPod / Lambda) or stays on the CPU box.
