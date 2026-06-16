#!/usr/bin/env python3
"""Supervised fine-tuning (SFT) for the Weaver MoE base on agentic tool-calling traces.

Takes the PRETRAINED base checkpoint (the same atomic bundle train.py emits) and
fine-tunes it on prompt/completion traces (e.g. distill/agentic_traces.json), with
the LOSS MASKED on the prompt so the model only learns to *produce* the completion
— reasoning + `<|call:tool|> {...}` calls + recovery.

Design choices mirror the pretrain kit:
  * Same MoEGPT / MoEGPTConfig from ../model.py; checkpoints use the same bundle keys
    so an SFT output is loadable by inference / GGUF-merge exactly like a base ckpt.
  * fp16 autocast + GradScaler on T4 (bf16 has no tensor-core path on T4 -> ~3-4x
    slower; see project memory). Override with --bf16 only on Ampere+.
  * Router aux/z losses stay ON during SFT so the experts don't collapse.
  * targets use ignore_index = -1 (matches MoEGPT.forward), so masked positions = -1.

HONEST SCALE NOTE: 10 seed traces will overfit a 203M model in seconds and teach it
nothing general. This trainer is the *mechanism*; it needs thousands of deduped,
quality-filtered traces (generate them with the Opus/235B API into a .jsonl) to make
the model competitive at tool-calling. Quantity + diversity is the lever, not epochs.

Usage (on Kaggle / any box with torch+tokenizers):
  python sft_train.py \
      --base-ckpt /kaggle/input/weaver-moe-ckpt/ckpt_latest.pt \
      --data agentic_traces.json \
      --tokenizer ../kaggle/code_dataset/tokenizer/tokenizer.json \
      --out ./sft_ckpts --epochs 3 --lr 2e-4

  python sft_train.py --smoke      # tiny random-init self-test (no base ckpt needed)
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- import the kit's model from the parent dir ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

IGNORE = -1                      # MoEGPT.forward uses ignore_index = -1
EOS_ID = 0                       # tokenizer meta.json: eos '<|endoftext|>' id 0

# Chat template. The role markers are plain text (byte-level BPE tokenizes them as
# bytes — no special-token surgery needed). The PROMPT span ends right after the
# assistant header; everything from there on is supervised.
USER_HDR = "<|user|>\n"
ASST_HDR = "\n<|assistant|>\n"


# ----------------------------------------------------------------------------- data
def load_traces(path: str) -> List[dict]:
    """Accept either a JSON array (.json) or one-object-per-line (.jsonl)."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(ln) for ln in f if ln.strip()]
    out = []
    for r in rows:
        if r.get("prompt") and r.get("completion"):
            out.append({"prompt": str(r["prompt"]), "completion": str(r["completion"])})
    if not out:
        raise SystemExit(f"[sft] no usable prompt/completion rows in {path}")
    return out


def build_example(tok, prompt: str, completion: str, ctx_len: int
                  ) -> Optional[Tuple[List[int], List[int]]]:
    """Return (input_ids, targets) for one trace, with the prompt masked to -1.

    Full token stream:  [user_hdr + prompt + asst_hdr] + [completion] + [EOS]
    We supervise predicting the completion span + EOS only. Because MoEGPT computes
    CE(logits[t], targets[t]) on aligned positions, targets are the NEXT token; the
    last prompt position is what predicts the first completion token, so the boundary
    is (prompt_len - 1).
    """
    enc = lambda s: tok.encode(s).ids
    prompt_ids = enc(USER_HDR + prompt + ASST_HDR)
    comp_ids = enc(completion) + [EOS_ID]
    seq = prompt_ids + comp_ids
    if len(seq) < 2:
        return None
    # Truncate from the LEFT of the prompt if needed, never losing the completion.
    if len(seq) > ctx_len + 1:
        overflow = len(seq) - (ctx_len + 1)
        if overflow >= len(prompt_ids):
            # completion alone won't fit -> drop trailing completion tokens instead
            seq = seq[: ctx_len + 1]
            prompt_ids = seq[: max(1, len(prompt_ids) - overflow)]
        else:
            prompt_ids = prompt_ids[overflow:]
            seq = prompt_ids + comp_ids
            seq = seq[: ctx_len + 1]
    p = len(prompt_ids)
    x = seq[:-1]
    y = seq[1:]
    # mask everything before the first supervised prediction (index p-1)
    for i in range(min(p - 1, len(y))):
        y[i] = IGNORE
    if all(t == IGNORE for t in y):
        return None
    return x, y


def collate(batch, pad_to: int):
    import torch
    B = len(batch)
    T = min(pad_to, max(len(x) for x, _ in batch))
    X = torch.full((B, T), EOS_ID, dtype=torch.long)
    Y = torch.full((B, T), IGNORE, dtype=torch.long)
    for b, (x, y) in enumerate(batch):
        n = min(len(x), T)
        X[b, :n] = torch.tensor(x[:n], dtype=torch.long)
        Y[b, :n] = torch.tensor(y[:n], dtype=torch.long)
    return X, Y


# ----------------------------------------------------------------------------- model io
def load_base(base_ckpt: str, device: str):
    import torch
    from model import MoEGPT, MoEGPTConfig  # noqa
    bundle = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    cfg = MoEGPTConfig.from_dict(bundle["model_config"])
    model = MoEGPT(cfg)
    missing, unexpected = model.load_state_dict(bundle["model"], strict=False)
    print(f"[sft] loaded base step={bundle.get('global_step')} "
          f"tokens={bundle.get('token_count')} | missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device), cfg, bundle


def save_sft(out_dir, model, cfg, optimizer, scaler, step, chat_template, fname):
    import torch
    os.makedirs(out_dir, exist_ok=True)
    bundle = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(step),
        "token_count": 0,
        "best_val": 0.0,
        "model_config": cfg.to_dict(),
        "model_config_hash": cfg.config_hash(),
        "stage": "sft",
        "chat_template": chat_template,
    }
    tmp = os.path.join(out_dir, fname + ".tmp")
    final = os.path.join(out_dir, fname)
    torch.save(bundle, tmp)
    os.replace(tmp, final)
    print(f"[sft] saved {final}")
    return final


# ----------------------------------------------------------------------------- train
@dataclass
class SFTArgs:
    epochs: int = 3
    lr: float = 2e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    batch_size: int = 8
    grad_accum: int = 1
    ctx_len: int = 1024
    grad_clip: float = 1.0
    log_every: int = 5
    bf16: bool = False


def lr_at(step, total, peak, warmup):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.1 * peak + 0.5 * (peak - 0.1 * peak) * (1 + math.cos(math.pi * prog))


def train(model, cfg, examples, device, a: SFTArgs, out_dir, chat_template):
    import torch
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95), eps=1e-8)

    use_fp16 = (device == "cuda" and not a.bf16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    amp_dtype = torch.bfloat16 if a.bf16 else torch.float16

    steps_per_epoch = math.ceil(len(examples) / (a.batch_size * a.grad_accum))
    total_steps = max(1, steps_per_epoch * a.epochs)
    warmup = max(1, int(total_steps * a.warmup_ratio))
    print(f"[sft] examples={len(examples)} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} warmup={warmup} fp16={use_fp16} bf16={a.bf16}")

    model.train()
    step = 0
    t0 = time.time()
    for epoch in range(a.epochs):
        order = list(range(len(examples)))
        # deterministic shuffle keyed by epoch (no global RNG dependency)
        order.sort(key=lambda i: ((i * 2654435761) ^ (epoch * 40503)) & 0xFFFFFFFF)
        micro = [order[i:i + a.batch_size] for i in range(0, len(order), a.batch_size)]
        accum = 0
        optimizer.zero_grad(set_to_none=True)
        for mb in micro:
            X, Y = collate([examples[i] for i in mb], a.ctx_len)
            X, Y = X.to(device), Y.to(device)
            with torch.autocast(device_type=("cuda" if device == "cuda" else "cpu"),
                                dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss, aux = model(X, Y)
                loss = loss / a.grad_accum
            scaler.scale(loss).backward()
            accum += 1
            if accum == a.grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                for g in optimizer.param_groups:
                    g["lr"] = lr_at(step, total_steps, a.lr, warmup)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                if step % a.log_every == 0:
                    lm = float(aux["lm_loss"])
                    print(f"step {step:4d}/{total_steps}  lm {lm:.4f}  "
                          f"aux {float(aux['aux_loss']):.3f}  z {float(aux['z_loss']):.3f}  "
                          f"lr {optimizer.param_groups[0]['lr']:.2e}  "
                          f"{(step+1)/(time.time()-t0):.2f} it/s")
                step += 1
        save_sft(out_dir, model, cfg, optimizer, scaler, step, chat_template, "ckpt_latest.pt")
    save_sft(out_dir, model, cfg, optimizer, scaler, step, chat_template, "ckpt_sft_final.pt")
    print(f"[sft] done in {(time.time()-t0)/60:.2f} min, {step} steps")


# ----------------------------------------------------------------------------- smoke
def smoke():
    """Tiny random-init self-test: verifies masking + a couple of train steps run and
    loss is finite. Exercises the data path with a byte-level fallback 'tokenizer'."""
    import torch
    from model import MoEGPT, MoEGPTConfig

    class ByteTok:  # stand-in so smoke needs no tokenizer.json
        class _E:
            def __init__(self, ids): self.ids = ids
        def encode(self, s):
            return ByteTok._E([(b % 255) + 1 for b in s.encode("utf-8")])

    tok = ByteTok()
    cfg = MoEGPTConfig(vocab_size=256, ctx_len=128, n_layer=2, n_head=2,
                       n_embd=64, head_dim=32, n_experts=4, top_k=2, d_ff=128)
    model = MoEGPT(cfg)
    traces = [{"prompt": f"Failure #{i}: race in shard {i}.",
               "completion": f"Reasoning... <|call:probe|> {{\"shard\": {i}}}\n"
                             f"System Response: [ERROR] missing ctx\nFix... <|call:fix|> {{\"ok\": true}}"}
              for i in range(6)]
    exs = []
    for t in traces:
        ex = build_example(tok, t["prompt"], t["completion"], cfg.ctx_len)
        assert ex is not None
        x, y = ex
        assert len(x) == len(y)
        assert any(v == IGNORE for v in y), "prompt span must be masked"
        assert any(v != IGNORE for v in y), "completion span must be supervised"
        exs.append(ex)
    # masking sanity: first supervised target index > 0 (prompt was masked)
    x0, y0 = exs[0]
    first_sup = next(i for i, v in enumerate(y0) if v != IGNORE)
    assert first_sup > 0, "expected a masked prompt prefix"
    print(f"[smoke] masking OK: first supervised idx={first_sup}/{len(y0)}")
    train(model, cfg, exs, "cpu", SFTArgs(epochs=1, batch_size=3, log_every=1,
          ctx_len=cfg.ctx_len), out_dir="/tmp/sft_smoke", chat_template=USER_HDR + "{p}" + ASST_HDR)
    print("[smoke] PASS")


# ----------------------------------------------------------------------------- cli
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt")
    ap.add_argument("--data", default=os.path.join(_HERE, "agentic_traces.json"))
    ap.add_argument("--tokenizer",
                    default=os.path.join(_PARENT, "kaggle/code_dataset/tokenizer/tokenizer.json"))
    ap.add_argument("--out", default=os.path.join(_HERE, "sft_ckpts"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if not args.base_ckpt:
        raise SystemExit("--base-ckpt required (or use --smoke)")

    import torch
    from tokenizers import Tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    rows = load_traces(args.data)
    model, cfg, _ = load_base(args.base_ckpt, device)
    chat_template = USER_HDR + "{prompt}" + ASST_HDR + "{completion}"

    examples = []
    for r in rows:
        ex = build_example(tok, r["prompt"], r["completion"], min(args.ctx_len, cfg.ctx_len))
        if ex is not None:
            examples.append(ex)
    print(f"[sft] usable examples: {len(examples)}/{len(rows)} (ctx_len={min(args.ctx_len, cfg.ctx_len)})")
    if len(examples) < 200:
        print("[sft] WARNING: <200 examples — this WILL overfit a 203M model. "
              "Generate thousands of traces (Opus/235B API) before trusting eval numbers.")

    a = SFTArgs(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
                grad_accum=args.grad_accum, ctx_len=min(args.ctx_len, cfg.ctx_len),
                warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay, bf16=args.bf16)
    train(model, cfg, examples, device, a, args.out, chat_template)


if __name__ == "__main__":
    main()
