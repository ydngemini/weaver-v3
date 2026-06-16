#!/usr/bin/env python3
"""Validate + dedup agentic traces produced by Claude subagents into agentic_traces.jsonl.

Generation is done by the Claude Code subscription (subagents author batches), NOT a paid
API. Each subagent returns a JSON array of {domain, prompt, completion} (trace_id optional).
This script is the deterministic gate between "model wrote some traces" and "training set":

  * validates each trace has a non-empty prompt + completion in the SFT format
    (at least one well-formed `<|call:NAME|> {json}` call in the completion)
  * dedups against everything already in the .jsonl by a normalized prompt key
  * appends survivors as one-object-per-line JSONL (sft_train.py / eval_toolcall.py read both)

Usage:
  python merge_traces.py batch1.json batch2.json            # merge files
  some_producer | python merge_traces.py -                  # merge from stdin
  python merge_traces.py --out agentic_traces.jsonl batch*.json
  python merge_traces.py --stats                            # just report current jsonl
"""
from __future__ import annotations
import argparse, json, os, re, sys, hashlib
from typing import List, Dict, Any

_HERE = os.path.dirname(os.path.abspath(__file__))
_CALL = re.compile(r"<\|call:([A-Za-z0-9_.\-]+)\|>")


def _norm_key(prompt: str) -> str:
    return hashlib.sha1(re.sub(r"\s+", " ", prompt.strip().lower()).encode()).hexdigest()


def _valid(trace: Dict[str, Any]) -> bool:
    p, c = trace.get("prompt"), trace.get("completion")
    if not (isinstance(p, str) and isinstance(c, str) and p.strip() and c.strip()):
        return False
    return bool(_CALL.search(c))   # must contain at least one tool call


def _load_existing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1); f.seek(0)
        if head == "[":
            rows = json.load(f)
        else:
            rows = [json.loads(ln) for ln in f if ln.strip()]
    return rows


def _read_batch(src: str) -> List[Dict[str, Any]]:
    text = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # tolerate jsonl batches too
        return [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return data if isinstance(data, list) else [data]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="batch JSON/JSONL files, or - for stdin")
    ap.add_argument("--out", default=os.path.join(_HERE, "agentic_traces.jsonl"))
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    existing = _load_existing(args.out)
    seen = {_norm_key(r["prompt"]) for r in existing if r.get("prompt")}

    if args.stats:
        tools = {}
        for r in existing:
            for m in _CALL.finditer(r.get("completion", "")):
                tools[m.group(1)] = tools.get(m.group(1), 0) + 1
        print(json.dumps({"path": args.out, "n_traces": len(existing),
                          "unique_prompts": len(seen), "distinct_tools": len(tools),
                          "top_tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])[:15])},
                         indent=2))
        return

    added, dup, bad = 0, 0, 0
    with open(args.out, "a", encoding="utf-8") as out:
        for src in (args.inputs or ["-"]):
            for t in _read_batch(src):
                if not _valid(t):
                    bad += 1; continue
                k = _norm_key(t["prompt"])
                if k in seen:
                    dup += 1; continue
                seen.add(k)
                rec = {"trace_id": t.get("trace_id", k[:16]),
                       "domain": t.get("domain", "unknown"),
                       "prompt": t["prompt"], "completion": t["completion"]}
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                added += 1
    print(f"[merge] +{added} added  {dup} dup  {bad} invalid  -> {args.out} "
          f"(total {len(existing) + added})")


if __name__ == "__main__":
    main()
