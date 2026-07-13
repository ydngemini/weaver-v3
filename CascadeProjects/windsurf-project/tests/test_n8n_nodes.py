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
CONTRACT_VERSION = "weaver-headless-n8n-v1"
CORRELATION_ID = "req-live-contract-test"
PUBLIC_SUCCESS_FIELDS = {
    "contract_version", "status", "error", "correlation_id",
    "manifested_response", "speaker", "speaker_boundary_applied",
    "speaker_model", "internal_draft_hidden", "reflection_applied",
    "soul_voice_active", "codebase_grounded", "expert_parallel",
    "expert_count", "experts_completed", "expert_errors",
    "expert_fanout_elapsed_ms", "execution_id", "timestamp",
    "pipeline_architecture", "pipeline_version",
}


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
        "contract_version": CONTRACT_VERSION,
        "correlation_id": CORRELATION_ID,
        "deadline_ms": 115_000,
        "text": "Explain how the cognition mesh keeps an embodied action safe.",
        "self_check": False,
        "introspect": False,
        "path_glob": "**/*",
        "search_query": "",
        "codebase_context": "",
        "quantum_pathway": "",
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
    "expert_drafts",
    "lora_error",
    "qwen3b_error",
    "qwen3b_route",
    "dominant_lobe",
    "experts_activated",
}
leaked = sorted(private_fields.intersection(response))
check("privacy", not leaked, f"No private intermediate fields leaked ({leaked or 'none'})")
check(
    "exact_fields",
    set(response) == PUBLIC_SUCCESS_FIELDS,
    f"Response has exactly the public v1 fields (extra={sorted(set(response) - PUBLIC_SUCCESS_FIELDS)})",
)
check(
    "response_body",
    response.get("contract_version") == CONTRACT_VERSION
    and response.get("status") == "ok"
    and response.get("error") is False
    and response.get("correlation_id") == CORRELATION_ID
    and isinstance(response.get("manifested_response"), str)
    and bool(response.get("manifested_response", "").strip()),
    "Version, discriminator, correlation, and manifestation are valid",
)
check(
    "speaker_boundary",
    response.get("speaker") == "weaver"
    and response.get("speaker_boundary_applied") is True
    and response.get("speaker_model") == "qwen.qwen3-235b-a22b-2507"
    and response.get("internal_draft_hidden") is True
    and response.get("reflection_applied") is True,
    "Only Weaver's reviewed speaker response crossed the boundary",
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

print(f"\n{BAR}\nBOUNDED PUBLIC TELEMETRY\n{BAR}")
check(
    "bounded_counts",
    isinstance(response.get("expert_fanout_elapsed_ms"), int)
    and 0 <= response.get("expert_fanout_elapsed_ms", -1) <= 115_000
    and isinstance(response.get("experts_completed"), int)
    and 0 <= response.get("experts_completed", -1) <= 5
    and isinstance(response.get("expert_errors"), int)
    and 0 <= response.get("expert_errors", -1) <= 5,
    "Only bounded aggregate expert telemetry is public",
)

print(f"\n{'═' * 64}")
passed = sum(RESULTS.values())
total = len(RESULTS)
print(f"  {passed}/{total} n8n v6 public-contract checks passed")
print(f"{'═' * 64}\n")
raise SystemExit(0 if passed == total else 1)
