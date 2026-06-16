#!/usr/bin/env python3
"""Eval harness for the Weaver MoE tool-calling SFT model.

Measures how well an SFT checkpoint reproduces agentic tool-calling on a HELD-OUT
set of traces, in the SAME format sft_train.py trains on:

    <|user|>\\n{prompt}\\n<|assistant|>\\n{completion}<EOS>

where the completion interleaves reasoning with calls in the syntax

    <|call:tool_name|> {"arg": "val"}

It reports tool-calling quality (format validity, first-tool accuracy, ordered
sequence match, multiset F1 over tool names, and argument fidelity) plus wall-clock
latency, and a $/1000-traces comparison against a cloud model (Opus) whose prices you
pass on the CLI — nothing about pricing is hard-coded here.

Two layers, deliberately split so the scoring is testable with no GPU / no torch:
  * PURE PYTHON  — parse_tool_calls / score_pair / aggregate.  `--smoke` exercises these.
  * TORCH        — load_model / generate / the eval loop.  Runs on Kaggle (or any
                   box with torch + tokenizers + ../model.py).

Usage (Kaggle / venv with torch):
  python eval_toolcall.py \\
      --ckpt ./sft_ckpts/ckpt_sft_final.pt \\
      --data agentic_traces.json --holdout 0.3 --seed 1337 \\
      --tokenizer ../kaggle/code_dataset/tokenizer/tokenizer.json \\
      --max-new-tokens 512 \\
      --opus-in-price 0 --opus-out-price 0   # set real $/Mtok to get a cost delta

  python eval_toolcall.py --eval-file holdout.jsonl --ckpt ...   # explicit eval set

  python eval_toolcall.py --smoke    # pure-python scoring self-test, no torch/GPU
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from typing import List, Dict, Optional, Tuple, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

EOS_ID = 0
USER_HDR = "<|user|>\n"
ASST_HDR = "\n<|assistant|>\n"

# A call is `<|call:NAME|>` optionally followed by a JSON object argument.
# NAME is [A-Za-z0-9_.-]+ ; the JSON arg (if present) is the first balanced {...}
# run after the marker on the same logical span.
_CALL_MARKER = re.compile(r"<\|call:([A-Za-z0-9_.\-]+)\|>")


# ------------------------------------------------------------------- pure-python parse
def _extract_json_object(s: str, start: int) -> Tuple[Optional[str], int]:
    """From s[start:], skip whitespace and return (json_text, end_idx) for the first
    balanced {...} (brace-counting, string/escape aware). (None, start) if none."""
    i = start
    n = len(s)
    while i < n and s[i] in " \t":
        i += 1
    if i >= n or s[i] != "{":
        return None, start
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < n:
        c = s[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return s[i:j + 1], j + 1
        j += 1
    return None, start  # unbalanced -> treat as no args


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Return ordered list of {name, args, args_valid, raw_args} for every
    `<|call:NAME|> {...}` in text. args is the parsed dict (or None if absent/invalid)."""
    calls: List[Dict[str, Any]] = []
    for m in _CALL_MARKER.finditer(text):
        name = m.group(1)
        raw, _end = _extract_json_object(text, m.end())
        args: Optional[dict] = None
        valid = False
        if raw is not None:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    args, valid = parsed, True
            except (ValueError, TypeError):
                valid = False
        calls.append({"name": name, "args": args, "args_valid": valid, "raw_args": raw})
    return calls


def _lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _multiset_f1(ref: List[str], gen: List[str]) -> float:
    from collections import Counter
    rc, gc = Counter(ref), Counter(gen)
    tp = sum((rc & gc).values())
    if tp == 0:
        return 0.0
    prec = tp / max(1, sum(gc.values()))
    rec = tp / max(1, sum(rc.values()))
    return 2 * prec * rec / (prec + rec)


def score_pair(ref_completion: str, gen_completion: str) -> Dict[str, float]:
    """Score one generated completion against its reference. All metrics in [0,1]."""
    ref = parse_tool_calls(ref_completion)
    gen = parse_tool_calls(gen_completion)
    ref_names = [c["name"] for c in ref]
    gen_names = [c["name"] for c in gen]

    n_gen = len(gen)
    format_valid = (sum(1 for c in gen if c["args_valid"]) / n_gen) if n_gen else 0.0
    arg_valid_rate = format_valid  # alias; args_valid already means name+json ok

    first_tool = 1.0 if (ref_names and gen_names and ref_names[0] == gen_names[0]) else 0.0

    denom = max(len(ref_names), len(gen_names))
    seq_match = (_lcs_len(ref_names, gen_names) / denom) if denom else 1.0
    set_f1 = _multiset_f1(ref_names, gen_names) if (ref_names or gen_names) else 1.0

    # position-aligned argument fidelity over the overlapping prefix
    aligned = min(len(ref), len(gen))
    key_jac = 0.0
    arg_exact = 0.0
    if aligned:
        for r, g in zip(ref[:aligned], gen[:aligned]):
            rk = set((r["args"] or {}).keys())
            gk = set((g["args"] or {}).keys())
            union = rk | gk
            key_jac += (len(rk & gk) / len(union)) if union else 1.0
            arg_exact += 1.0 if (r["args"] == g["args"] and g["args_valid"]) else 0.0
        key_jac /= aligned
        arg_exact /= aligned

    return {
        "n_ref_calls": float(len(ref)),
        "n_gen_calls": float(n_gen),
        "format_valid": format_valid,
        "arg_valid_rate": arg_valid_rate,
        "first_tool_correct": first_tool,
        "tool_seq_match": seq_match,
        "tool_set_f1": set_f1,
        "arg_key_jaccard": key_jac,
        "arg_exact_match": arg_exact,
    }


def aggregate(scores: List[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    keys = scores[0].keys()
    return {k: sum(s[k] for s in scores) / len(scores) for k in keys}


# ------------------------------------------------------------------- data split
def _load_rows(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        rows = json.load(f) if head == "[" else [json.loads(ln) for ln in f if ln.strip()]
    return [{"prompt": str(r["prompt"]), "completion": str(r["completion"])}
            for r in rows if r.get("prompt") and r.get("completion")]


def holdout_split(rows: List[dict], frac: float, seed: int) -> List[dict]:
    """Deterministic held-out subset (no global RNG dependency).

    NOTE: sft_train.py trains on the WHOLE trace file, so these held-out rows are NOT
    auto-excluded from training — carving a split here only avoids leakage if you also
    train on the complement. For a clean held-out eval, generate a separate set and pass
    it via --eval-file, or split the .jsonl before SFT and train on the other shard."""
    order = list(range(len(rows)))
    order.sort(key=lambda i: ((i * 2654435761) ^ (seed * 40503)) & 0xFFFFFFFF)
    k = max(1, int(round(len(rows) * frac)))
    keep = set(order[:k])
    return [rows[i] for i in sorted(keep)]


# ------------------------------------------------------------------- torch generation
def load_model(ckpt: str, device: str):
    import torch
    from model import MoEGPT, MoEGPTConfig  # noqa
    bundle = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = MoEGPTConfig.from_dict(bundle["model_config"])
    model = MoEGPT(cfg)
    missing, unexpected = model.load_state_dict(bundle["model"], strict=False)
    print(f"[eval] loaded ckpt stage={bundle.get('stage')} step={bundle.get('global_step')} "
          f"| missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model.to(device), cfg


def generate(model, cfg, tok, prompt: str, max_new_tokens: int,
             temperature: float, top_k: int, device: str) -> Tuple[str, int]:
    """Greedy (temperature==0) or sampled autoregressive decode. Stops on EOS.
    Returns (decoded_completion_text, n_new_tokens)."""
    import torch
    ids = tok.encode(USER_HDR + prompt + ASST_HDR).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids: List[int] = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -cfg.ctx_len:]
            logits, _, _ = model(idx_cond)            # (1, 1, vocab) at inference
            logits = logits[:, -1, :]
            if temperature and temperature > 0:
                logits = logits / temperature
                if top_k:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            else:
                nxt = int(torch.argmax(logits, dim=-1).item())
            if nxt == EOS_ID:
                break
            new_ids.append(nxt)
            idx = torch.cat([idx, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(new_ids), len(new_ids)


def run_eval(args) -> Dict[str, Any]:
    import torch
    from tokenizers import Tokenizer

    if args.eval_file:
        rows = _load_rows(args.eval_file)
    else:
        rows = holdout_split(_load_rows(args.data), args.holdout, args.seed)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} held-out traces")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(args.tokenizer)
    model, cfg = load_model(args.ckpt, device)

    per_trace, latencies, gen_tok_total, prompt_tok_total = [], [], 0, 0
    for i, r in enumerate(rows):
        t0 = time.time()
        gen_text, n_new = generate(model, cfg, tok, r["prompt"], args.max_new_tokens,
                                   args.temperature, args.top_k, device)
        dt = (time.time() - t0) * 1000.0
        latencies.append(dt)
        gen_tok_total += n_new
        prompt_tok_total += len(tok.encode(USER_HDR + r["prompt"] + ASST_HDR).ids)
        s = score_pair(r["completion"], gen_text)
        per_trace.append(s)
        if args.verbose:
            print(f"  [{i}] {dt:6.0f}ms first_tool={s['first_tool_correct']:.0f} "
                  f"seq={s['tool_seq_match']:.2f} fmt={s['format_valid']:.2f} :: {gen_text[:90]!r}")

    summary = aggregate(per_trace)
    n = max(1, len(rows))
    mean_ms = sum(latencies) / n
    # cost: cloud model would consume prompt+completion tokens at the prices given.
    opus_cost_1k = (prompt_tok_total * args.opus_in_price
                    + gen_tok_total * args.opus_out_price) / 1e6 * (1000.0 / n)
    report = {
        "n_traces": len(rows),
        "metrics": summary,
        "latency_ms_mean": mean_ms,
        "throughput_traces_per_s": (1000.0 / mean_ms) if mean_ms else 0.0,
        "gen_tokens_total": gen_tok_total,
        "prompt_tokens_total": prompt_tok_total,
        "cost_per_1k_traces": {
            "local_model_usd": 0.0,
            "opus_equiv_usd": round(opus_cost_1k, 4),
            "note": "opus_equiv uses --opus-in-price/--opus-out-price ($/Mtok); "
                    "0 by default — pass real prices for a meaningful delta.",
        },
    }
    print("\n=== Weaver MoE tool-calling eval ===")
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[eval] wrote {args.out}")
    return report


# ------------------------------------------------------------------- smoke (no torch)
def smoke():
    ref = ('Reasoning: probe first.\n'
           '<|call:query_graph_db|> {"query": "SHOW COLUMNS", "scope": "all"}\n'
           'System Response: [ERROR] not found\n'
           'Fix: roll back.\n'
           '<|call:rollback_migration|> {"id": "0091", "dry_run": false}\n'
           'Resolution: done.')

    # 1. parse
    calls = parse_tool_calls(ref)
    assert [c["name"] for c in calls] == ["query_graph_db", "rollback_migration"], calls
    assert calls[0]["args"] == {"query": "SHOW COLUMNS", "scope": "all"}, calls[0]
    assert all(c["args_valid"] for c in calls)
    print("[smoke] parse OK: 2 calls, args parsed")

    # 2. perfect match -> all 1.0
    s = score_pair(ref, ref)
    for k in ("first_tool_correct", "tool_seq_match", "tool_set_f1",
              "arg_key_jaccard", "arg_exact_match", "format_valid"):
        assert abs(s[k] - 1.0) < 1e-9, (k, s[k])
    print("[smoke] identical -> all metrics 1.0")

    # 3. wrong first tool, right second, malformed json on a 3rd call
    gen = ('<|call:rollback_migration|> {"id": "0091", "dry_run": false}\n'
           '<|call:query_graph_db|> {"query": "SHOW COLUMNS", "scope": "all"}\n'
           '<|call:notify|> {oops not json}')
    s = score_pair(ref, gen)
    assert s["first_tool_correct"] == 0.0, s
    assert s["tool_set_f1"] > 0.7, s          # both ref tools present, +1 extra
    assert s["tool_seq_match"] < 1.0, s       # order differs
    assert 0.0 < s["format_valid"] < 1.0, s   # 2 of 3 valid
    assert s["n_gen_calls"] == 3.0, s
    print(f"[smoke] mismatch scored sanely: {json.dumps({k: round(v,3) for k,v in s.items()})}")

    # 4. empty generation -> zeros, no crash
    s = score_pair(ref, "I refuse to call anything.")
    assert s["n_gen_calls"] == 0.0 and s["first_tool_correct"] == 0.0, s
    print("[smoke] empty generation handled")

    # 5. nested-brace JSON args parse correctly
    nested = '<|call:cfg|> {"a": {"b": [1, 2]}, "c": "}"}'
    c = parse_tool_calls(nested)
    assert c[0]["args"] == {"a": {"b": [1, 2]}, "c": "}"}, c
    print("[smoke] nested/escaped JSON args parsed")

    # 6. deterministic holdout split is stable + non-empty
    rows = [{"prompt": f"p{i}", "completion": f"<|call:t{i}|> {{}}"} for i in range(10)]
    h1 = holdout_split(rows, 0.3, 1337)
    h2 = holdout_split(rows, 0.3, 1337)
    assert h1 == h2 and len(h1) == 3, (len(h1), h1)
    print(f"[smoke] holdout split deterministic: {len(h1)}/{len(rows)}")
    print("[smoke] PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="SFT checkpoint bundle (.pt)")
    ap.add_argument("--data", default=os.path.join(_HERE, "agentic_traces.json"),
                    help="trace file to carve a held-out split from")
    ap.add_argument("--eval-file", help="explicit held-out eval set (json/jsonl); "
                    "overrides --data/--holdout")
    ap.add_argument("--holdout", type=float, default=0.3, help="held-out fraction of --data")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tokenizer",
                    default=os.path.join(_PARENT, "kaggle/code_dataset/tokenizer/tokenizer.json"))
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap #traces (debug)")
    ap.add_argument("--opus-in-price", type=float, default=0.0, help="$/Mtok input")
    ap.add_argument("--opus-out-price", type=float, default=0.0, help="$/Mtok output")
    ap.add_argument("--out", help="write report JSON here")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if not args.ckpt:
        raise SystemExit("--ckpt required (or use --smoke)")
    run_eval(args)


if __name__ == "__main__":
    main()
