#!/usr/bin/env python3
"""Live contract test for the canonical Weaver n8n v6 workflow.

This test deliberately validates only the public response contract. Private
prompts, source evidence, and intermediate expert output must never be echoed.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request


BAR = "─" * 64
RESULTS: dict[str, bool] = {}
WEBHOOK_URL = (
    os.environ.get("WEAVER_N8N_WEBHOOK_URL")
    or os.environ.get("N8N_WEBHOOK_URL")
    or "http://127.0.0.1:5678/webhook/weaver-input"
)
TIMEOUT_S = float(os.environ.get("WEAVER_N8N_TEST_TIMEOUT", "125"))


def mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def check(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = bool(ok)
    print(f"  {mark(ok)} {detail}")


def http_post(url: str, payload: dict, timeout: float) -> tuple[int | None, str, str | None]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.getcode(), response.read().decode("utf-8"), None
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace"), None
    except Exception as exc:  # pragma: no cover - live diagnostic path
        return None, "", str(exc)


print(f"\n{'═' * 64}")
print("  WEAVER N8N V6 — LIVE PUBLIC-CONTRACT TEST")
print(f"{'═' * 64}")
print(f"  Endpoint: {WEBHOOK_URL}")

started = time.monotonic()
status, body, error = http_post(
    WEBHOOK_URL,
    {
        "text": "Explain how the cognition mesh keeps an embodied action safe.",
        "source_file": "integration-test.md",
        "timestamp": "2026-07-12T00:00:00Z",
        "cognition_context": {
            "awareness_confidence": 0.95,
            "fabric_pressure": 0.10,
            "immune_status": "nominal",
            "open_components": [],
        },
    },
    TIMEOUT_S,
)
elapsed_ms = (time.monotonic() - started) * 1000

if error:
    print(f"  Request error: {error}")

try:
    response = json.loads(body) if body else {}
except json.JSONDecodeError:
    response = {}

print(f"\n{BAR}\nPUBLIC CONTRACT\n{BAR}")
check(
    "http_response",
    status == 200 and isinstance(response, dict),
    f"HTTP 200 JSON response ({status=}, {elapsed_ms:.0f}ms)",
)

private_fields = {
    "raw_input",
    "original_input",
    "collapsed_response",
    "synthesis_prompt",
    "codebase_context",
    "internet_context",
    "mcp_context",
}
leaked = sorted(private_fields.intersection(response))
check("privacy", not leaked, f"No private intermediate fields leaked ({leaked or 'none'})")
check(
    "response_body",
    isinstance(response.get("manifested_response"), str)
    and bool(response.get("manifested_response", "").strip()),
    "Manifested response is non-empty",
)

print(f"\n{BAR}\nPARALLEL EXPERT PIPELINE\n{BAR}")
check(
    "v6_identity",
    response.get("pipeline_version") == "v6-parallel-cognition",
    f"Pipeline version is v6 ({response.get('pipeline_version')!r})",
)
check(
    "parallel_topology",
    response.get("pipeline_architecture") == "parallel-fanout-barrier"
    and response.get("expert_parallel") is True,
    "Five-expert fanout/barrier reports parallel execution",
)
check(
    "expert_completion",
    response.get("expert_count") == 5
    and response.get("experts_completed") == 5
    and response.get("expert_errors") == 0,
    (
        "Five experts completed "
        f"(completed={response.get('experts_completed')}, errors={response.get('expert_errors')})"
    ),
)

print(f"\n{BAR}\nCOGNITION, GEOMETRY, AND WRITEBACK\n{BAR}")
awareness = response.get("awareness_confidence")
check(
    "cognition",
    response.get("cognition_mesh_active") is True
    and isinstance(awareness, (int, float))
    and 0 <= awareness <= 1,
    f"Bounded cognition metadata is active (awareness={awareness!r})",
)
layout = response.get("qubit_layout")
check(
    "geometry",
    isinstance(layout, dict) and set(layout) == {f"q{i}" for i in range(7)},
    "Seven-register geometric layout is present",
)
check(
    "no_writeback",
    response.get("written_to_hub") is False
    and response.get("writeback_mode") == "response-metadata-only"
    and response.get("hub_lobe_id") is None,
    "Workflow truthfully reports metadata-only response with no vault writeback",
)

print(f"\n{'═' * 64}")
passed = sum(RESULTS.values())
total = len(RESULTS)
print(f"  {passed}/{total} n8n v6 public-contract checks passed")
print(f"{'═' * 64}\n")
raise SystemExit(0 if passed == total else 1)
