# Weaver MoE — Agentic Tool-Calling Distillation

Post-training the pretrained MoE base into a tool-calling / agentic model.

## Files
- `gen_agentic_traces.py` — emits `agentic_traces.json`: 10 **seed** traces (probe → trap →
  self-heal loop) in the SFT syntax `<|call:tool_name|> {"arg": "val"}`. These define the
  *format/quality template*, not a training set.
- `agentic_traces.json` — the seed traces (2 grounded in real Weaver incidents).
- `sft_train.py` — fine-tunes the base on prompt/completion traces with **prompt-masked loss**
  (only the completion is supervised). Emits checkpoints in the same bundle format as `train.py`,
  so SFT output is inference/GGUF-merge compatible.
- `eval_toolcall.py` — scores an SFT checkpoint on a held-out trace split: format validity,
  first-tool accuracy, ordered sequence match (LCS), multiset-F1 over tool names, and argument
  fidelity, plus latency + a $/1k-traces delta vs a cloud model (prices passed on the CLI).
  Scoring is pure-Python and GPU-free (`--smoke` self-tests it); generation needs torch.

## Pipeline
1. **Generate data (the real lever).** 10 traces overfit a 203M model instantly and teach nothing
   general. Scale up the trace set to **thousands** of deduped, quality-filtered traces into
   `agentic_traces.jsonl`. **Generate via the Claude Code subscription (subagents / Workflow
   fan-out), NOT the paid API** — author batches of traces in the same probe→trap→self-heal format
   as the seeds. Keep a clean held-out shard out of training for `eval_toolcall.py`.
2. **SFT.**
   ```
   python sft_train.py \
       --base-ckpt /kaggle/input/weaver-moe-ckpt/ckpt_latest.pt \
       --data agentic_traces.jsonl \
       --tokenizer ../kaggle/code_dataset/tokenizer/tokenizer.json \
       --out ./sft_ckpts --epochs 3 --lr 2e-4
   ```
   T4 → fp16 by default (bf16 has no tensor-core path on T4). Router aux/z losses stay on.
3. **Eval.**
   ```
   python eval_toolcall.py \
       --ckpt ./sft_ckpts/ckpt_sft_final.pt \
       --eval-file holdout.jsonl \
       --tokenizer ../kaggle/code_dataset/tokenizer/tokenizer.json \
       --opus-in-price <$/Mtok> --opus-out-price <$/Mtok>
   ```
   Or carve a split from a single file with `--data agentic_traces.json --holdout 0.3`
   (held-out rows are NOT auto-excluded from SFT — train on the complement to avoid leakage).

## Self-test
- `python sft_train.py --smoke` — tiny random-init run; checks masking boundary + a few train steps.
- `python eval_toolcall.py --smoke` — pure-Python scoring self-test (parse, metrics, split); no GPU.

## Honest scope
SWE-bench-grade ability is out of reach at this scale (see vault session log). The defensible win
is **format-correct tool calling on Weaver's fixed tool set, at ~16ms/$0**, with the small model as
the fast tier and escalation to Opus for hard steps.
