#!/usr/bin/env python3
"""Non-destructive live smoke test for n8n and adjacent Weaver services."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import urllib.error
import urllib.request


RESULTS: dict[str, bool] = {}
BAR = "─" * 64
N8N_URL = (
    os.environ.get("WEAVER_N8N_WEBHOOK_URL")
    or os.environ.get("N8N_WEBHOOK_URL")
    or "http://127.0.0.1:5678/webhook/weaver-input"
)
N8N_TIMEOUT_S = float(os.environ.get("WEAVER_N8N_TEST_TIMEOUT", "125"))
N8N_CONTRACT_VERSION = "weaver-headless-n8n-v1"
N8N_CORRELATION_ID = "req-endpoint-smoke"
BRIDGE_HEALTH_URL = os.environ.get(
    "WEAVER_OBSIDIAN_HEALTH_URL", "http://127.0.0.1:5679/health"
)
NEXUS_HEALTH_URL = os.environ.get(
    "WEAVER_NEXUS_HEALTH_URL", "http://127.0.0.1:9998/health"
)
NEXUS_WS_URL = os.environ.get("WEAVER_NEXUS_WS_URL", "ws://127.0.0.1:9999")


def port_open(host: str, port: int, timeout: float = 2) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def http_request(
    url: str, payload: dict | None = None, timeout: float = 5
) -> tuple[int | None, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.getcode(), response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - live diagnostic path
        return None, str(exc)


def record(name: str, ok: bool, detail: str) -> None:
    RESULTS[name] = bool(ok)
    print(f"  {'✅' if ok else '❌'} {detail}")


print(f"\n{BAR}\nPORT CONNECTIVITY\n{BAR}")
for port, label in ((5678, "n8n"), (5679, "Obsidian bridge"), (9999, "Nexus Bus")):
    record(f"port_{port}", port_open("127.0.0.1", port), f"{label} port {port}")

print(f"\n{BAR}\nN8N V6 WEBHOOK\n{BAR}")
status, body = http_request(
    N8N_URL,
    {
        "contract_version": N8N_CONTRACT_VERSION,
        "correlation_id": N8N_CORRELATION_ID,
        "deadline_ms": 115_000,
        "text": "Return a concise Weaver service smoke-test acknowledgement.",
        "self_check": False,
        "introspect": False,
        "path_glob": "**/*",
        "search_query": "",
        "codebase_context": "",
        "quantum_pathway": "",
        "cognition_context": {
            "awareness_confidence": 0.75,
            "fabric_pressure": 0.1,
            "immune_status": "nominal",
            "open_components": [],
        },
    },
    N8N_TIMEOUT_S,
)
try:
    n8n_response = json.loads(body)
except json.JSONDecodeError:
    n8n_response = {}
private_fields = {
    "raw_input", "original_input", "collapsed_response", "codebase_context",
    "expert_drafts", "lora_error", "qwen3b_error", "qwen3b_route",
}
record(
    "n8n_webhook",
    status == 200
    and n8n_response.get("contract_version") == N8N_CONTRACT_VERSION
    and n8n_response.get("status") == "ok"
    and n8n_response.get("error") is False
    and n8n_response.get("correlation_id") == N8N_CORRELATION_ID
    and n8n_response.get("speaker") == "weaver"
    and n8n_response.get("speaker_boundary_applied") is True
    and n8n_response.get("internal_draft_hidden") is True
    and n8n_response.get("pipeline_version") == "v6-parallel-cognition"
    and n8n_response.get("pipeline_architecture") == "parallel-fanout-barrier"
    and not private_fields.intersection(n8n_response),
    f"Canonical n8n webhook returned the privacy-safe v6 contract ({status=})",
)

print(f"\n{BAR}\nREAD-ONLY SERVICE HEALTH\n{BAR}")
for name, url in (("bridge_health", BRIDGE_HEALTH_URL), ("nexus_health", NEXUS_HEALTH_URL)):
    health_status, health_body = http_request(url)
    try:
        health_payload = json.loads(health_body)
    except json.JSONDecodeError:
        health_payload = {}
    record(
        name,
        health_status == 200 and isinstance(health_payload, dict),
        f"{url} returned JSON health ({health_status=})",
    )


async def test_nexus_websocket() -> bool:
    try:
        import websockets

        async with websockets.connect(NEXUS_WS_URL) as websocket:
            initial = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3))
            if initial.get("type") != "sync":
                return False
            await websocket.send(json.dumps({"action": "register", "lobe_id": "smoke_test"}))
            acknowledgement = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
            if acknowledgement.get("type") != "ack":
                return False
            await websocket.send(json.dumps({"action": "ping"}))
            pong = json.loads(await asyncio.wait_for(websocket.recv(), timeout=2))
            return pong.get("type") == "pong"
    except Exception as exc:  # pragma: no cover - live diagnostic path
        print(f"  Nexus WebSocket error: {exc}")
        return False


print(f"\n{BAR}\nNEXUS WEBSOCKET\n{BAR}")
record("nexus_ws", asyncio.run(test_nexus_websocket()), "Sync/register/ping contract")

print(f"\n{'═' * 64}")
passed = sum(RESULTS.values())
total = len(RESULTS)
print(f"  {passed}/{total} non-destructive endpoint checks passed")
print(f"{'═' * 64}\n")
raise SystemExit(0 if passed == total else 1)
