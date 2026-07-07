# Code review — distill SFT/eval pipeline (2026-06-17)

Scope: `origin/feat/oracle-arm-free-deploy..HEAD` (commits 266e6f4, 9ef0a31) —
4 new Python files + trace data. Data files all valid (10 seed / 88 corpus /
3×14 waves; every row has a `<|call:NAME|>` and non-empty prompt+completion).

Linked: [[distill-stage-in-repo-not-tmp]], [[t4-bf16-no-tensor-cores]]

## Findings
1. **[bug, confirmed]** `sft_train.py` `build_example` — when a completion alone
   exceeds `ctx_len`, the `overflow >= len(prompt_ids)` branch keeps the full
   prompt at the head of `seq` but sets `prompt_ids = seq[:1]`, so **0 targets
   are masked** and the model is trained to emit the user prompt. Reproduced
   with ctx=64 / 500-tok completion: `first_supervised_idx=0, masked=0`.
   Latent today (traces < 1024 tok) but corrupts training on long traces.
2. **[altitude]** `--data` defaults to `agentic_traces.json` (10-trace seed that
   `gen_agentic_traces.py` overwrites) in both `sft_train.py:286` and
   `eval_toolcall.py:369`; the real corpus is `agentic_traces.jsonl` (88). Default
   runs silently use 10 traces.
3. **[reuse]** `eval_toolcall.generate` reimplements decode that
   `MoEGPT.generate` (`model.py:484`) already provides.
4. **[efficiency]** eval decode has no KV cache (full forward over growing seq
   each step, O(n²)); also re-encodes the prompt at `eval_toolcall.py:275`.
5. **[bug, minor]** `sft_train.train` discards the trailing partial
   grad-accum group each epoch when `grad_accum>1` (zeroed next epoch, no step).
