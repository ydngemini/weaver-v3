#!/usr/bin/env python3
"""Structural gate for the agentic trace corpus + wave batches.

Formalizes the per-wave check that's otherwise run ad-hoc: every trace must have
a non-empty string prompt + completion, and the completion must contain at least
one well-formed `<|call:NAME|>` marker (the same regex merge_traces.py gates on).

Exit 0 if all files pass, 1 on the first file with any violation. Pure stdlib —
no torch / GPU, so it runs in the fast CI lane.

Usage:
  python validate_traces.py                 # validate the default corpus + waves/
  python validate_traces.py a.json b.jsonl  # validate explicit files
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Must match merge_traces.py / eval_toolcall.py call syntax exactly.
_CALL = re.compile(r"<\|call:([A-Za-z0-9_.\-]+)\|>")


def _load(path: str):
    """Load a JSON array (.json) or one-object-per-line (.jsonl)."""
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(ln) for ln in f if ln.strip()]


def validate_file(path: str) -> list[str]:
    """Return a list of human-readable violation strings (empty == valid)."""
    try:
        rows = _load(path)
    except Exception as e:  # noqa: BLE001 — surface any parse failure as a violation
        return [f"could not parse: {type(e).__name__}: {e}"]
    if not isinstance(rows, list) or not rows:
        return ["file holds no trace rows"]
    bad: list[str] = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            bad.append(f"row {i}: not a JSON object")
            continue
        p, c = r.get("prompt"), r.get("completion")
        if not (isinstance(p, str) and p.strip()):
            bad.append(f"row {i}: missing/empty prompt")
        if not (isinstance(c, str) and c.strip()):
            bad.append(f"row {i}: missing/empty completion")
        elif not _CALL.search(c):
            bad.append(f"row {i}: completion has no <|call:NAME|> marker")
    return bad


def _default_targets() -> list[str]:
    targets = []
    for name in ("agentic_traces.json", "agentic_traces.jsonl"):
        p = os.path.join(_HERE, name)
        if os.path.exists(p):
            targets.append(p)
    targets += sorted(glob.glob(os.path.join(_HERE, "waves", "*.json")))
    return targets


def main() -> int:
    targets = sys.argv[1:] or _default_targets()
    if not targets:
        print("[validate] no trace files found", file=sys.stderr)
        return 1
    failed = 0
    for path in targets:
        violations = validate_file(path)
        rel = os.path.relpath(path, _HERE)
        if violations:
            failed += 1
            print(f"FAIL  {rel}")
            for v in violations[:10]:
                print(f"        - {v}")
            if len(violations) > 10:
                print(f"        ... and {len(violations) - 10} more")
        else:
            n = len(_load(path))
            print(f"OK    {rel}  ({n} traces)")
    if failed:
        print(f"\n[validate] {failed} file(s) failed validation")
        return 1
    print(f"\n[validate] all {len(targets)} file(s) valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
