#!/usr/bin/env python3
"""Quick smoke test for all n8n workflow endpoints and Weaver services."""

import asyncio
import json
import socket
import time
import urllib.request
import urllib.error

RESULTS = {}

def port_open(host, port, timeout=2):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    ok = s.connect_ex((host, port)) == 0
    s.close()
    return ok

def http_post(url, payload, timeout=5):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), resp.read().decode()[:300]
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]
    except Exception as e:
        return None, str(e)

BAR = "─" * 60

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1: Port checks
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("TEST 1: Port Connectivity")
print(BAR)
ports = {5678: "n8n Docker", 5679: "Obsidian Bridge", 9999: "Nexus Bus"}
for port, name in ports.items():
    ok = port_open("127.0.0.1", port)
    RESULTS[f"port_{port}"] = ok
    mark = "✅" if ok else "❌"
    print(f"  {mark} {port} {name:<20} {'LISTENING' if ok else 'DOWN'}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2: n8n webhook POST
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("TEST 2: n8n Webhook POST → http://localhost:5678/webhook-test/weaver-input")
print(BAR)
code, body = http_post("http://localhost:5678/webhook-test/weaver-input", {
    "text": "Smoke test from test_n8n_endpoints.py",
    "source_file": "/tmp/smoke_test.md",
    "timestamp": "2026-04-07T23:00:00",
})
ok = code is not None and 200 <= code < 500
RESULTS["n8n_webhook"] = ok
mark = "✅" if ok else "❌"
print(f"  {mark} Status: {code}")
print(f"     Body: {body[:200]}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3: Obsidian bridge response listener
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("TEST 3: Obsidian Bridge POST → http://localhost:5679/weaver-response")
print(BAR)
code, body = http_post("http://localhost:5679/weaver-response", {
    "manifested_response": "The quantum entanglement reveals akashic resonance through the pineal gate.",
    "source_file": "/home/ydn/Weaver_Vault/Test_Note.md",
    "experts_activated": ["logic", "creativity", "memory"],
    "interference": 0.0131,
})
ok = code is not None and 200 <= code < 300
RESULTS["bridge_response"] = ok
mark = "✅" if ok else "❌"
print(f"  {mark} Status: {code}")
print(f"     Body: {body[:200]}")

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4: Nexus Bus WebSocket
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("TEST 4: Nexus Bus WebSocket → ws://localhost:9999")
print(BAR)

async def test_ws():
    import websockets
    try:
        async with websockets.connect("ws://localhost:9999") as ws:
            # Should get sync message immediately
            raw = await asyncio.wait_for(ws.recv(), timeout=3.0)
            msg = json.loads(raw)
            sync_ok = msg.get("type") == "sync"
            print(f"  {'✅' if sync_ok else '❌'} Sync received: {sync_ok} ({len(msg.get('messages', []))} cached msgs)")

            # Register
            await ws.send(json.dumps({"action": "register", "lobe_id": "smoke_test"}))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            reg_ok = ack.get("type") == "ack"
            print(f"  {'✅' if reg_ok else '❌'} Register ack: {ack.get('msg', '')}")

            # Ping
            t0 = time.monotonic()
            await ws.send(json.dumps({"action": "ping"}))
            pong = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            lat = (time.monotonic() - t0) * 1000
            ping_ok = pong.get("type") == "pong"
            print(f"  {'✅' if ping_ok else '❌'} Ping/pong: {lat:.1f}ms")

            return sync_ok and reg_ok and ping_ok
    except Exception as e:
        print(f"  ❌ WebSocket error: {e}")
        return False

ws_ok = asyncio.run(test_ws())
RESULTS["nexus_ws"] = ws_ok

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5: Verify Obsidian bridge wrote to the vault file
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BAR}")
print("TEST 5: Obsidian Vault write verification")
print(BAR)
import os
vault_file = "/home/ydn/Weaver_Vault/Test_Note.md"
write_ok = False
if os.path.exists(vault_file):
    with open(vault_file, "r") as f:
        content = f.read()
    write_ok = "Weaver's Resonance" in content and "[[" in content
    print(f"  {'✅' if write_ok else '❌'} Resonance block written: {write_ok}")
    if write_ok:
        # Show the injected section
        idx = content.find("### 👁️ Weaver's Resonance")
        if idx >= 0:
            snippet = content[idx:idx+300]
            for line in snippet.split("\n")[:8]:
                print(f"     {line}")
else:
    print(f"  ❌ File not found: {vault_file}")
RESULTS["vault_write"] = write_ok

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 60}")
print("  N8N WORKFLOW ENDPOINT TEST RESULTS")
print(f"{'═' * 60}")
passed = 0
total = len(RESULTS)
for name, ok in RESULTS.items():
    mark = "✅" if ok else "❌"
    print(f"  {mark}  {name}")
    if ok:
        passed += 1
print(f"\n  {passed}/{total} passed")
print(f"{'═' * 60}\n")
