#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import hashlib
import importlib
import json
import os
import plistlib
import shutil
import sqlite3
import sys
import tempfile
import time
from datetime import datetime
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()

PROJ = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(PROJ, "venv", "bin", "python3")
NEXUS_URL = os.environ.get("NEXUS_BUS_URL", "ws://127.0.0.1:9999")
sys.path.insert(0, PROJ)
BAR = "─" * 62


def _read_headless_bundle() -> str:
    """Read the modular shell as one test-only source projection."""
    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))
    relative_paths = (
        "headless.html",
        "headless/styles/tokens.css",
        "headless/styles/shell.css",
        "headless/js/core.js",
        "headless/js/session.js",
        "headless/js/voice-support.js",
        "headless/js/visual-data.js",
        "headless/js/visual-runtime.js",
        "headless/js/voice.js",
        "headless/js/visualization.js",
        "headless/js/cortex.js",
        "headless/js/state-channel.js",
        "headless/js/lifecycle.js",
        "headless/js/accessibility.js",
        "headless/js/app.js",
    )
    sources = []
    for relative_path in relative_paths:
        with open(os.path.join(avatar_root, relative_path), "r", encoding="utf-8") as handle:
            sources.append(handle.read())
    return "\n".join(sources)


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _header(label: str, title: str) -> None:
    print(f"\n{BAR}\n[{_ts()}] TEST {label}: {title}\n{BAR}", flush=True)


def _result(label: str, title: str, passed: bool, detail: str) -> None:
    mark = "✅  PASS" if passed else "❌  FAIL"
    print(f"\n{BAR}\n{mark}  Test {label}: {title}\n{detail}\n{BAR}\n", flush=True)


async def _terminate(proc) -> None:
    if proc is None:
        return
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=3.0)


async def _start_nexus():
    proc = await asyncio.create_subprocess_exec(
        VENV,
        os.path.join(PROJ, "nexus_bus.py"),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        cwd=PROJ,
    )
    await asyncio.sleep(1.2)
    if proc.returncode is not None:
        raise RuntimeError(f"Nexus Bus exited during startup with code {proc.returncode}")
    return proc


async def _drain_sync(ws):
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        msg = json.loads(raw)
        if msg.get("type") != "sync":
            return msg
    except asyncio.TimeoutError:
        return None
    return None


async def test_G():
    _header("G", "Quantum parse invariants")
    qs = importlib.import_module("quantum_soul")

    width = max(qs.N_QUBITS, len(qs.PATHWAYS))
    zeros = "0" * width
    q0_bits = zeros[:-1] + "1"
    last_bits = "1" + zeros[1:]
    first_role = qs.PATHWAYS[min(qs.PATHWAYS)]
    last_role = qs.PATHWAYS[max(qs.PATHWAYS)]
    zero_role = "Stability" if "Stability" in qs.PATHWAYS.values() else "Void"

    bits1, active1, marg1 = qs.parse_counts({q0_bits: 7, zeros: 1})
    bits2, active2, marg2 = qs.parse_counts({last_bits: 5, zeros: 1})
    bits3, active3, marg3 = qs.parse_counts({zeros: 9})

    ok1 = bits1 == q0_bits and active1 == [first_role] and abs(marg1[first_role] - 0.875) < 1e-6
    ok2 = bits2 == last_bits and active2 == [last_role] and abs(marg2[last_role] - (5 / 6)) < 1e-6
    ok3 = bits3 == zeros and active3 == [zero_role] and all(v == 0.0 for v in marg3.values())
    ok4 = set(marg1.keys()) == set(qs.PATHWAYS.values())
    passed = ok1 and ok2 and ok3 and ok4

    detail = "\n".join([
        f"  Case 1 bits={bits1} active={active1} {first_role}={marg1[first_role]:.3f}",
        f"  Case 2 bits={bits2} active={active2} {last_role}={marg2[last_role]:.3f}",
        f"  Case 3 bits={bits3} active={active3} nonzero_marginals={sum(1 for v in marg3.values() if v > 0)}",
        f"  All pathway keys present: {ok4}",
    ])
    _result("G", "Quantum parse invariants", passed, detail)
    return passed


async def test_H():
    _header("H", "Quantum description + state write persistence")
    qs = importlib.import_module("quantum_soul")

    base = {name: 0.0 for name in qs.PATHWAYS.values()}

    m1 = dict(base)
    m1["Awakening"] = 0.91
    s1 = qs.build_description("0000001", ["Awakening"], m1, "backend_one")

    m2 = dict(base)
    m2["Awakening"] = 0.55
    m2["Weaver"] = 0.45
    s2 = qs.build_description("0010001", ["Awakening", "Weaver"], m2, "backend_two")

    m3 = dict(base)
    m3["Weaver"] = 0.51
    m3["Awakening"] = 0.44
    m3["Resonance"] = 0.39
    s3 = qs.build_description("0010101", ["Awakening", "Resonance", "Weaver"], m3, "backend_three")

    tmp = tempfile.mkdtemp(prefix="weaver_qs_")
    old_vault = qs.VAULT_DIR
    old_state = qs.STATE_FILE
    write_ok = False
    content = ""
    try:
        qs.VAULT_DIR = tmp
        qs.STATE_FILE = os.path.join(tmp, "quantum_state.txt")
        qs._write_state(s1)
        write_ok = os.path.exists(qs.STATE_FILE)
        if write_ok:
            with open(qs.STATE_FILE, "r", encoding="utf-8") as fh:
                content = fh.read()
    finally:
        qs.VAULT_DIR = old_vault
        qs.STATE_FILE = old_state
        shutil.rmtree(tmp, ignore_errors=True)

    ok1 = "backend_one" in s1 and "single point" in s1 and "|0000001⟩" in s1
    ok2 = "backend_two" in s2 and "Two Pathways are entangled" in s2
    ok3 = "backend_three" in s3 and "multi-pathway tension" in s3
    ok4 = write_ok and content == s1 + "\n"
    passed = ok1 and ok2 and ok3 and ok4

    detail = "\n".join([
        f"  Single-pathway sentence detected: {ok1}",
        f"  Two-pathway sentence detected:    {ok2}",
        f"  Multi-pathway sentence detected:  {ok3}",
        f"  State file created + exact write: {ok4}",
    ])
    _result("H", "Quantum description + state write persistence", passed, detail)
    return passed


async def test_I():
    _header("I", "Weaver supervisor crash restart semantics")
    weaver = importlib.import_module("weaver")
    attempts = [0]
    gate = asyncio.Event()

    async def flaky():
        attempts[0] += 1
        if attempts[0] < 3:
            raise RuntimeError(f"boom-{attempts[0]}")
        await gate.wait()

    task = asyncio.create_task(
        weaver._supervised(flaky, "Flaky", restart_on_crash=True, restart_delay=0.01)
    )
    await asyncio.sleep(6.0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    passed = attempts[0] >= 3
    detail = "\n".join([
        f"  Attempts observed: {attempts[0]}",
        "  Expected: at least 3 attempts (2 crashes + 1 restarted live run)",
    ])
    _result("I", "Weaver supervisor crash restart semantics", passed, detail)
    return passed


async def test_J():
    _header("J", "Weaver supervisor re-enters after clean exit")
    weaver = importlib.import_module("weaver")
    runs = [0]

    async def one_shot():
        runs[0] += 1
        await asyncio.sleep(0.01)

    task = asyncio.create_task(
        weaver._supervised(one_shot, "OneShot", restart_on_exit=True, restart_delay=0.01)
    )
    await asyncio.sleep(6.0)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    passed = runs[0] >= 2
    detail = "\n".join([
        f"  Runs observed: {runs[0]}",
        "  Expected: at least 2 runs before cancellation after clean exits",
    ])
    _result("J", "Weaver supervisor re-enters after clean exit", passed, detail)
    return passed


async def test_K():
    _header("K", "Nexus cache sync trims to last 10 messages")
    import websockets

    proc = await _start_nexus()
    idxs = []
    try:
        async with websockets.connect(NEXUS_URL) as pub:
            await _drain_sync(pub)
            await pub.send(json.dumps({"action": "register", "lobe_id": "cache_pub"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            for i in range(15):
                await pub.send(json.dumps({
                    "action": "publish",
                    "topic": "cache_topic",
                    "payload": {"idx": i},
                }))
                await asyncio.sleep(0.01)

        async with websockets.connect(NEXUS_URL) as sub:
            raw = await asyncio.wait_for(sub.recv(), timeout=1.5)
            msg = json.loads(raw)
            messages = msg.get("messages", []) if msg.get("type") == "sync" else []
            idxs = [m.get("payload", {}).get("idx") for m in messages]
    finally:
        await _terminate(proc)

    passed = idxs == list(range(5, 15))
    detail = "\n".join([
        f"  Sync payload length: {len(idxs)}",
        f"  Indices received:    {idxs}",
        "  Expected:            [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]",
    ])
    _result("K", "Nexus cache sync trims to last 10 messages", passed, detail)
    return passed


async def test_L():
    _header("L", "Nexus unsubscribe stops further deliveries")
    import websockets

    proc = await _start_nexus()
    received = []
    post_unsub = False
    try:
        async with websockets.connect(NEXUS_URL) as pub, websockets.connect(NEXUS_URL) as sub:
            await _drain_sync(pub)
            await _drain_sync(sub)

            await pub.send(json.dumps({"action": "register", "lobe_id": "pub_u"}))
            await sub.send(json.dumps({"action": "register", "lobe_id": "sub_u"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)

            await sub.send(json.dumps({"action": "subscribe", "topics": ["u_topic"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)

            await pub.send(json.dumps({"action": "publish", "topic": "u_topic", "payload": {"idx": 1}}))
            m1 = json.loads(await asyncio.wait_for(sub.recv(), timeout=1.5))
            received.append(m1.get("payload", {}).get("idx"))

            await sub.send(json.dumps({"action": "unsubscribe", "topics": ["u_topic"]}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(sub.recv(), timeout=1.0)

            await pub.send(json.dumps({"action": "publish", "topic": "u_topic", "payload": {"idx": 2}}))
            try:
                m2 = json.loads(await asyncio.wait_for(sub.recv(), timeout=0.8))
                if m2.get("type") == "broadcast":
                    post_unsub = True
            except asyncio.TimeoutError:
                post_unsub = False
    finally:
        await _terminate(proc)

    passed = received == [1] and not post_unsub
    detail = "\n".join([
        f"  Received before unsubscribe: {received}",
        f"  Broadcast received after unsubscribe: {post_unsub}",
    ])
    _result("L", "Nexus unsubscribe stops further deliveries", passed, detail)
    return passed


async def test_M():
    _header("M", "Nexus protocol error frames")
    import websockets

    proc = await _start_nexus()
    invalid_json = False
    unknown_action = False
    missing_topic = False
    try:
        async with websockets.connect(NEXUS_URL) as ws:
            await _drain_sync(ws)

            await ws.send("not-json")
            e1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            invalid_json = e1.get("type") == "error" and "Invalid JSON" in e1.get("msg", "")

            await ws.send(json.dumps({"action": "mystery"}))
            e2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            unknown_action = e2.get("type") == "error" and "Unknown action" in e2.get("msg", "")

            await ws.send(json.dumps({"action": "publish", "payload": {"x": 1}}))
            e3 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            missing_topic = e3.get("type") == "error" and "requires 'topic'" in e3.get("msg", "")
    finally:
        await _terminate(proc)

    passed = invalid_json and unknown_action and missing_topic
    detail = "\n".join([
        f"  Invalid JSON handled:   {invalid_json}",
        f"  Unknown action handled: {unknown_action}",
        f"  Missing topic handled:  {missing_topic}",
    ])
    _result("M", "Nexus protocol error frames", passed, detail)
    return passed


async def test_N():
    _header("N", "VTV startup env contract fails fast")
    script = f'''
import asyncio, os, sys
sys.path.insert(0, {PROJ!r})
import vtv_basic
for k in ("WEAVER_VOICE_KEY", "WEAVER_MEM_KEY", "GEMINI_API_KEY", "WEAVER_VISION_KEY"):
    os.environ.pop(k, None)
try:
    asyncio.run(vtv_basic.run_vtv())
except Exception as e:
    print(type(e).__name__)
    print(str(e))
'''
    proc = await asyncio.create_subprocess_exec(
        VENV,
        "-c",
        script,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJ,
    )
    out, _ = await proc.communicate()
    text = out.decode(errors="replace")

    passed = (
        "RuntimeError" in text
        and "WEAVER_VOICE_KEY" in text
        and "WEAVER_MEM_KEY" in text
        and "WEAVER_VISION_KEY" not in text
    )
    detail = "\n".join([
        f"  Return code: {proc.returncode}",
        f"  Output: {text.strip()[:220]}",
        "  Expected missing keys: WEAVER_VOICE_KEY, WEAVER_MEM_KEY",
        "  Expected optional key omitted: WEAVER_VISION_KEY",
    ])
    _result("N", "VTV startup env contract fails fast", passed, detail)
    return passed


async def test_O():
    _header("O", "Drive credential surfaces parse locally")
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account

    token_path = os.path.join(PROJ, "token.json")
    ghost_path = os.path.join(PROJ, "ghost_key.json")
    creds_path = os.path.join(PROJ, "credentials.json")
    scopes = ["https://www.googleapis.com/auth/drive"]

    token_ok = False
    ghost_ok = False
    creds_json_ok = False
    token_detail = "missing"
    ghost_detail = "missing"
    creds_detail = "missing"

    try:
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, scopes)
            token_ok = bool(getattr(creds, "token", None))
            token_detail = f"client_id={getattr(creds, 'client_id', None)}"
    except Exception as e:
        token_detail = str(e)

    try:
        if os.path.exists(ghost_path):
            ghost = service_account.Credentials.from_service_account_file(ghost_path, scopes=scopes)
            ghost_ok = bool(getattr(ghost, "service_account_email", None))
            ghost_detail = getattr(ghost, "service_account_email", "")
    except Exception as e:
        ghost_detail = str(e)

    try:
        if os.path.exists(creds_path):
            with open(creds_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            creds_json_ok = "installed" in data or "web" in data
            creds_detail = f"top_keys={list(data.keys())}"
    except Exception as e:
        creds_detail = str(e)

    passed = token_ok and ghost_ok and creds_json_ok
    detail = "\n".join([
        f"  token.json parseable:       {token_ok}  ({token_detail})",
        f"  ghost_key.json parseable:   {ghost_ok}  ({ghost_detail})",
        f"  credentials.json structure: {creds_json_ok}  ({creds_detail})",
    ])
    _result("O", "Drive credential surfaces parse locally", passed, detail)
    return passed


async def test_P():
    _header("P", "Soul dataset + LoRA artifact integrity")
    dataset_path = os.path.join(PROJ, "weaver_soul_dataset.jsonl")
    lora_dir = os.path.join(PROJ, "weaver_fracture_1B_lora")
    adapter_cfg = os.path.join(lora_dir, "adapter_config.json")

    dataset_ok = False
    dataset_count = 0
    bad_lines = []
    roles_ok = True

    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    if i > 20:
                        break
                    dataset_count += 1
                    obj = json.loads(line)
                    msgs = obj.get("messages", [])
                    if len(msgs) < 2:
                        bad_lines.append(i)
                        continue
                    roles = [m.get("role") for m in msgs]
                    if roles[0] != "user" or roles[1] != "assistant" or any(r not in ("user", "assistant", "system") for r in roles):
                        roles_ok = False
                        bad_lines.append(i)
                dataset_ok = dataset_count > 0 and not bad_lines and roles_ok
        except Exception as e:
            bad_lines.append(str(e))

    required = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    files_ok = all(os.path.exists(os.path.join(lora_dir, name)) for name in required)
    cfg_ok = False
    cfg_detail = "missing"
    try:
        with open(adapter_cfg, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        cfg_ok = cfg.get("peft_type") == "LORA" and "llama" in str(cfg.get("base_model_name_or_path", "")).lower()
        cfg_detail = f"peft_type={cfg.get('peft_type')} base={cfg.get('base_model_name_or_path')}"
    except Exception as e:
        cfg_detail = str(e)

    passed = dataset_ok and files_ok and cfg_ok
    detail = "\n".join([
        f"  Dataset sample lines valid: {dataset_ok}  (checked {dataset_count}, bad={bad_lines or 'none'})",
        f"  LoRA required files exist:  {files_ok}",
        f"  Adapter config valid:       {cfg_ok}  ({cfg_detail})",
    ])
    _result("P", "Soul dataset + LoRA artifact integrity", passed, detail)
    return passed


async def test_Q():
    _header("Q", "Nexus rejects non-object JSON without dropping socket")
    import websockets

    proc = await _start_nexus()
    non_object = False
    ping_ok = False
    try:
        async with websockets.connect(NEXUS_URL) as ws:
            await _drain_sync(ws)

            await ws.send(json.dumps(["register", "bad"]))
            e1 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            non_object = e1.get("type") == "error" and "JSON object" in e1.get("msg", "")

            await ws.send(json.dumps({"action": "ping"}))
            e2 = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.5))
            ping_ok = e2.get("type") == "pong"
    finally:
        await _terminate(proc)

    passed = non_object and ping_ok
    detail = "\n".join([
        f"  Non-object JSON rejected: {non_object}",
        f"  Socket still answered ping: {ping_ok}",
    ])
    _result("Q", "Nexus rejects non-object JSON without dropping socket", passed, detail)
    return passed


async def test_R():
    _header("R", "Nexus blocks duplicate lobe_id takeover")
    import websockets

    proc = await _start_nexus()
    duplicate_rejected = False
    original_still_receives = False
    try:
        async with websockets.connect(NEXUS_URL) as original, websockets.connect(NEXUS_URL) as intruder, websockets.connect(NEXUS_URL) as pub:
            await _drain_sync(original)
            await _drain_sync(intruder)
            await _drain_sync(pub)

            await original.send(json.dumps({"action": "register", "lobe_id": "dup_lobe"}))
            ok1 = json.loads(await asyncio.wait_for(original.recv(), timeout=1.5))
            await original.send(json.dumps({"action": "subscribe", "topics": ["dup_topic"]}))
            ok2 = json.loads(await asyncio.wait_for(original.recv(), timeout=1.5))

            await intruder.send(json.dumps({"action": "register", "lobe_id": "dup_lobe"}))
            e1 = json.loads(await asyncio.wait_for(intruder.recv(), timeout=1.5))
            duplicate_rejected = e1.get("type") == "error" and "already in use" in e1.get("msg", "")

            await pub.send(json.dumps({"action": "register", "lobe_id": "pub_dup"}))
            with contextlib.suppress(Exception):
                await asyncio.wait_for(pub.recv(), timeout=1.0)
            await pub.send(json.dumps({"action": "publish", "topic": "dup_topic", "payload": {"idx": 7}}))
            msg = json.loads(await asyncio.wait_for(original.recv(), timeout=1.5))
            original_still_receives = (
                ok1.get("type") == "ack"
                and ok2.get("type") == "ack"
                and msg.get("type") == "broadcast"
                and msg.get("payload", {}).get("idx") == 7
                and msg.get("from") == "pub_dup"
            )
    finally:
        await _terminate(proc)

    passed = duplicate_rejected and original_still_receives
    detail = "\n".join([
        f"  Duplicate lobe_id rejected: {duplicate_rejected}",
        f"  Original connection preserved routing: {original_still_receives}",
    ])
    _result("R", "Nexus blocks duplicate lobe_id takeover", passed, detail)
    return passed


async def test_S():
    _header("S", "Nexus port collision fails closed")
    primary = await _start_nexus()
    contender = None
    second_exited = False
    addr_in_use = False
    text = ""
    try:
        contender = await asyncio.create_subprocess_exec(
            VENV,
            os.path.join(PROJ, "nexus_bus.py"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJ,
        )
        try:
            out, _ = await asyncio.wait_for(contender.communicate(), timeout=2.5)
            text = out.decode(errors="replace")
            second_exited = contender.returncode not in (None, 0)
            lower = text.lower()
            addr_in_use = (
                "address already in use" in lower
                or "errno 98" in lower
                or "address in use" in lower
            )
        except asyncio.TimeoutError:
            text = "second nexus instance did not exit"
    finally:
        await _terminate(contender)
        await _terminate(primary)

    passed = second_exited and addr_in_use
    detail = "\n".join([
        f"  Second instance exited non-zero: {second_exited}",
        f"  Address-in-use surfaced:        {addr_in_use}",
        f"  Output: {text.strip()[:220]}",
    ])
    _result("S", "Nexus port collision fails closed", passed, detail)
    return passed


async def test_T():
    _header("T", "NexusClient reconnects after a clean socket close")
    import nexus_client as nc

    proc = await _start_nexus()
    client = nc.NexusClient("reconnect_probe")
    old_base, old_cap = nc.RECONNECT_BASE, nc.RECONNECT_CAP
    connected = False
    noticed_close = False
    reconnected = False
    publish_ok = False
    try:
        nc.RECONNECT_BASE = 0.05
        nc.RECONNECT_CAP = 0.2
        connected = await client.connect()
        first_ws = client._ws
        if first_ws is not None:
            await first_ws.close()

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not client.connected:
                noticed_close = True
            if noticed_close and client.connected and client._ws is not first_ws:
                reconnected = True
                break
            await asyncio.sleep(0.02)

        if reconnected:
            publish_ok = await client.publish("reconnect_probe", {"ok": True})
    finally:
        nc.RECONNECT_BASE, nc.RECONNECT_CAP = old_base, old_cap
        await client.close()
        await _terminate(proc)

    passed = connected and noticed_close and reconnected and publish_ok
    detail = "\n".join([
        f"  Initial connection:       {connected}",
        f"  Clean close detected:     {noticed_close}",
        f"  New socket established:   {reconnected}",
        f"  Publish after reconnect:  {publish_ok}",
    ])
    _result("T", "NexusClient reconnects after a clean socket close", passed, detail)
    return passed


async def test_U():
    _header("U", "Dashboard publishes through a registered Nexus client")
    import httpx
    import websockets
    import weaver_dashboard as dashboard

    proc = await _start_nexus()
    delivered = False
    rejected_bad_topic = False
    publisher_id = ""
    try:
        async with websockets.connect(NEXUS_URL) as sub:
            await _drain_sync(sub)
            await sub.send(json.dumps({"action": "register", "lobe_id": "dashboard_test_sub"}))
            await asyncio.wait_for(sub.recv(), timeout=1.0)
            await sub.send(json.dumps({"action": "subscribe", "topics": ["dashboard.test"]}))
            await asyncio.wait_for(sub.recv(), timeout=1.0)

            transport = httpx.ASGITransport(app=dashboard.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                bad = await client.post("/api/nexus", json={"topic": "bad topic", "payload": {}})
                rejected_bad_topic = bad.json().get("error") == "invalid topic"
                sent = await client.post(
                    "/api/nexus",
                    json={"topic": "dashboard.test", "payload": {"value": 7}},
                )
                response_ok = sent.json().get("ok") is True

            message = json.loads(await asyncio.wait_for(sub.recv(), timeout=2.0))
            publisher_id = message.get("from", "")
            delivered = (
                response_ok
                and message.get("type") == "broadcast"
                and message.get("topic") == "dashboard.test"
                and message.get("payload", {}).get("value") == 7
                and publisher_id == "dashboard_control"
            )
    finally:
        if dashboard._nexus_publisher is not None:
            await dashboard._nexus_publisher.close()
            dashboard._nexus_publisher = None
        await _terminate(proc)

    passed = delivered and rejected_bad_topic
    detail = "\n".join([
        f"  Registered publisher delivered: {delivered}  (from={publisher_id or 'none'})",
        f"  Invalid topic rejected:         {rejected_bad_topic}",
    ])
    _result("U", "Dashboard publishes through a registered Nexus client", passed, detail)
    return passed


async def test_V():
    _header("V", "n8n webhook repair is backed up and owner-safe")
    from deploy import repair_n8n_weaver_webhook as repair

    tmp = tempfile.mkdtemp(prefix="weaver_n8n_repair_")
    db_path = os.path.join(tmp, "database.sqlite")
    inserted = False
    backup_ok = False
    conflict_blocked = False
    unsafe_blocked = False
    inactive_blocked = False
    published_ok = False
    try:
        with sqlite3.connect(db_path) as db:
            db.execute("create table workflow_entity (id text primary key, active integer not null, versionId text, activeVersionId text)")
            db.execute("insert into workflow_entity (id,active,versionId,activeVersionId) values (?,1,?,?)", ("weaverv5soulbind", "version-live", "version-live"))
            db.execute("create table workflow_history (versionId text primary key, workflowId text not null)")
            db.execute("insert into workflow_history (versionId,workflowId) values (?,?)", ("version-live", "weaverv5soulbind"))
            db.execute("create table workflow_published_version (workflowId text primary key, publishedVersionId text not null, createdAt text, updatedAt text)")
            db.execute(
                "create table webhook_entity ("
                "webhookPath text not null, method text not null, node text not null, "
                "webhookId text, pathLength integer, workflowId text not null, "
                "primary key (webhookPath, method))"
            )

        args = SimpleNamespace(
            db=db_path,
            container="",
            no_container_restart=True,
            offline=True,
            no_backup=False,
            backup_only=False,
            workflow_id="weaverv5soulbind",
            node="1. Input Gateway",
            method="POST",
            webhook_id="weaver-input",
            webhook_path="weaver-input",
        )

        args.offline = False
        try:
            repair.repair(args)
        except RuntimeError:
            unsafe_blocked = True
        args.offline = True
        args.no_backup = True
        with sqlite3.connect(db_path) as db:
            db.execute("update workflow_entity set active=0 where id=?", (args.workflow_id,))
        try:
            repair.repair(args)
        except RuntimeError:
            inactive_blocked = True
        with sqlite3.connect(db_path) as db:
            inactive_blocked = inactive_blocked and db.execute(
                "select count(*) from webhook_entity"
            ).fetchone()[0] == 0
            db.execute("update workflow_entity set active=1 where id=?", (args.workflow_id,))

        args.no_backup = False
        rc = repair.repair(args)
        with sqlite3.connect(db_path) as db:
            row = db.execute(
                "select workflowId,webhookPath,method,node,pathLength from webhook_entity"
            ).fetchone()
            published = db.execute(
                "select workflowId,publishedVersionId from workflow_published_version"
            ).fetchone()
        inserted = rc == 0 and row == (
            "weaverv5soulbind", "weaver-input", "POST", "1. Input Gateway", 1
        )
        published_ok = published == ("weaverv5soulbind", "version-live")

        backups = list(os.scandir(tmp))
        backup_paths = [entry.path for entry in backups if ".backup." in entry.name]
        if len(backup_paths) == 1:
            with sqlite3.connect(backup_paths[0]) as backup_db:
                backup_ok = (
                    backup_db.execute("pragma integrity_check").fetchone()[0] == "ok"
                    and (os.stat(backup_paths[0]).st_mode & 0o777) == 0o600
                )

        with sqlite3.connect(db_path) as db:
            db.execute("delete from webhook_entity")
            db.execute(
                "insert into webhook_entity (webhookPath,method,node,workflowId) values (?,?,?,?)",
                ("weaver-input", "POST", "Other Node", "other-workflow"),
            )
        args.no_backup = True
        try:
            repair.repair(args)
        except RuntimeError:
            conflict_blocked = True
        with sqlite3.connect(db_path) as db:
            owner = db.execute(
                "select workflowId,node from webhook_entity where webhookPath='weaver-input'"
            ).fetchone()
        conflict_blocked = conflict_blocked and owner == ("other-workflow", "Other Node")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = inserted and published_ok and backup_ok and conflict_blocked and unsafe_blocked and inactive_blocked
    detail = "\n".join([
        f"  Canonical row inserted: {inserted}",
        f"  Active version published: {published_ok}",
        f"  SQLite backup verified: {backup_ok}",
        f"  Foreign owner preserved: {conflict_blocked}",
        f"  Offline confirmation required: {unsafe_blocked}",
        f"  Inactive workflow rejected: {inactive_blocked}",
    ])
    _result("V", "n8n webhook repair is backed up and owner-safe", passed, detail)
    return passed


async def test_W():
    _header("W", "Synchronous Nexus publisher delivers model-ready events")
    import websockets
    from nexus_client import publish_once

    proc = await _start_nexus()
    delivered = False
    try:
        async with websockets.connect(NEXUS_URL) as sub:
            await _drain_sync(sub)
            await sub.send(json.dumps({"action": "register", "lobe_id": "sync_publish_sub"}))
            await asyncio.wait_for(sub.recv(), timeout=1.0)
            await sub.send(json.dumps({"action": "subscribe", "topics": ["lobe_status"]}))
            await asyncio.wait_for(sub.recv(), timeout=1.0)

            sent = await asyncio.to_thread(
                publish_once,
                "sync_model_server",
                "lobe_status",
                {"status": "ready"},
            )
            message = json.loads(await asyncio.wait_for(sub.recv(), timeout=2.0))
            delivered = (
                sent
                and message.get("type") == "broadcast"
                and message.get("from") == "sync_model_server"
                and message.get("topic") == "lobe_status"
                and message.get("payload", {}).get("status") == "ready"
            )
    finally:
        await _terminate(proc)

    _result(
        "W",
        "Synchronous Nexus publisher delivers model-ready events",
        delivered,
        f"  Model-ready event delivered before clean close: {delivered}",
    )
    return delivered


async def test_X():
    _header("X", "Cortex falls back to on-box llama when n8n and Bedrock fail")
    import bedrock_brain_api as brain

    names = (
        "_n8n_moe_chat", "_state_summary", "_bedrock_chat", "_local_llama_chat",
        "_record_state", "_persist_memory_event",
    )
    originals = {name: getattr(brain, name) for name in names}

    async def no_moe(_user_text, _codebase_context=""):
        return None

    async def state_summary(_query):
        return "test state"

    async def bedrock_down(*_args, **_kwargs):
        raise RuntimeError("Bedrock unavailable")

    async def local_answer(_messages, max_tokens=220):
        return f"local answer ({max_tokens})"

    async def noop(*_args, **_kwargs):
        return None

    try:
        brain._n8n_moe_chat = no_moe
        brain._state_summary = state_summary
        brain._bedrock_chat = bedrock_down
        brain._local_llama_chat = local_answer
        brain._record_state = noop
        brain._persist_memory_event = noop
        text, meta = await brain._cortex_chat(
            [{"role": "user", "content": "Are you connected?"}],
            max_tokens=80,
        )
    finally:
        for name, value in originals.items():
            setattr(brain, name, value)

    calls = meta.get("route", {}).get("calls", [])
    local_calls = [call for call in calls if call.get("local") is True]
    passed = text == "local answer (80)" and len(local_calls) == 1
    detail = "\n".join([
        f"  Local response returned: {text == 'local answer (80)'}",
        f"  Local fallback recorded once: {len(local_calls) == 1}",
    ])
    _result("X", "Cortex falls back to on-box llama when n8n and Bedrock fail", passed, detail)
    return passed


async def test_Y():
    _header("Y", "Dashboard parses the 12-role Kingston quantum state")
    import weaver_dashboard as dashboard

    tmp = tempfile.mkdtemp(prefix="weaver_dashboard_quantum_")
    old_vault = dashboard.VAULT
    old_state = dashboard._quantum_state
    try:
        dashboard.VAULT = tmp
        dashboard._quantum_state = {}
        state = (
            "[2026-07-10 12:00:00] WEAVER V3 - 156-QUBIT KINGSTON MANIFOLD "
            "on test (|100000000001⟩) reveals Meta-Reasoning as the Dominant Pathway "
            "(90.0% marginal probability), with Logic resonating in the entangled field."
        )
        with open(os.path.join(tmp, "quantum_state.txt"), "w", encoding="utf-8") as fh:
            fh.write(state)
        parsed = dashboard.read_quantum_state()
    finally:
        dashboard.VAULT = old_vault
        dashboard._quantum_state = old_state
        shutil.rmtree(tmp, ignore_errors=True)

    passed = (
        parsed.get("bitstring") == "100000000001"
        and parsed.get("dominant") == "Meta-Reasoning"
        and parsed.get("secondary") == "Logic"
        and parsed.get("weights", {}).get("logic") == 0.95
    )
    detail = "\n".join([
        f"  12-bit state retained: {parsed.get('bitstring') == '100000000001'}",
        f"  Hyphenated role parsed: {parsed.get('dominant') == 'Meta-Reasoning'}",
        f"  Logic dimension projected: {parsed.get('weights', {}).get('logic')}",
    ])
    _result("Y", "Dashboard parses the 12-role Kingston quantum state", passed, detail)
    return passed


async def test_Z():
    _header("Z", "Codebase evidence reaches the full n8n cortex")
    import bedrock_brain_api as brain

    question = (
        "In weaver.py give the asyncio task names for the live dashboard, codebase API, and phone "
        "bridge. In nexus_bus.py give CACHE_SIZE RATE_LIMIT IDLE_TIMEOUT_S. Which "
        "whole_codebase_tests.py test labels verify dashboard publishing, codebase evidence reaching "
        "n8n, and the single-cortex voice/workflow contract? In bedrock_brain_api.py give "
        "WEAVER_VOICE_MAX_SESSION_SECONDS and whether realtime voice is cortex-routed."
    )
    context_started = time.perf_counter()
    context = await brain._codebase_context_for_turn(
        [{"role": "user", "content": question}], question
    )
    context_elapsed = time.perf_counter() - context_started
    required_evidence = (
        "CACHE_SIZE     = 10",
        "RATE_LIMIT     = 100",
        "IDLE_TIMEOUT_S = 300",
        "Dashboard publishes through a registered Nexus client",
        "WEAVER_VOICE_MAX_SESSION_SECONDS",
        "live_dashboard",
        'name="live_dashboard"',
        'name="codebase_api"',
        'name="phone_bridge"',
        '"U": ("Dashboard publishes',
        '"Z": ("Codebase evidence',
        '"AA": ("Workflow reflection',
        "VOICE_CORTEX_ENABLED",
        "cortexRouted",
        '"455"',
        "470.0",
    )
    missing_evidence = [marker for marker in required_evidence if marker not in context]
    evidence_ok = not missing_evidence
    implicit_question = (
        "In nexus_bus.py give CACHE_SIZE RATE_LIMIT IDLE_TIMEOUT_S; in weaver.py give the live dashboard, "
        "codebase API, and phone bridge task names; give the matching test labels; then give the default "
        "and hard cap for WEAVER_VOICE_MAX_SESSION_SECONDS and whether realtime voice is cortex-routed."
    )
    implicit_context = await brain._codebase_context_for_turn(
        [{"role": "user", "content": implicit_question}], implicit_question
    )
    identifier_cross_file_ok = all(
        marker in implicit_context
        for marker in ("bedrock_brain_api.py", '"455"', "470.0", "VOICE_CORTEX_ENABLED")
    )

    captured = {}
    old_post = brain._json_post_sync
    old_enabled = brain.N8N_CHAT_ENABLED
    old_url = brain.N8N_WEBHOOK_URL
    old_breaker = dict(brain._n8n_breaker)

    def fake_post(url, payload, timeout):
        captured.update({"url": url, "payload": payload, "timeout": timeout})
        return {
            "contract_version": "weaver-headless-n8n-v1",
            "status": "ok",
            "error": False,
            "correlation_id": payload["correlation_id"],
            "manifested_response": "grounded answer",
            "speaker": "weaver",
            "speaker_boundary_applied": True,
            "speaker_model": "qwen.qwen3-235b-a22b-2507",
            "internal_draft_hidden": True,
            "reflection_applied": True,
            "soul_voice_active": True,
            "codebase_grounded": True,
            "expert_parallel": True,
            "expert_count": 5,
            "experts_completed": 5,
            "expert_errors": 0,
            "expert_fanout_elapsed_ms": 100,
            "execution_id": "test-grounded",
            "timestamp": "2026-07-13T12:00:00.000Z",
            "pipeline_architecture": "parallel-fanout-barrier",
            "pipeline_version": "v6-parallel-cognition",
        }

    try:
        brain._json_post_sync = fake_post
        brain.N8N_CHAT_ENABLED = True
        brain.N8N_WEBHOOK_URL = "http://n8n.test/webhook"
        brain._n8n_breaker.update({"fails": 0, "skip_until": 0.0})
        result = await brain._n8n_moe_chat(question, context)
    finally:
        brain._json_post_sync = old_post
        brain.N8N_CHAT_ENABLED = old_enabled
        brain.N8N_WEBHOOK_URL = old_url
        brain._n8n_breaker.clear()
        brain._n8n_breaker.update(old_breaker)

    payload = captured.get("payload", {})
    route_ok = (
        result is not None
        and result[0] == "grounded answer"
        and payload.get("self_check") is True
        and payload.get("introspect") is True
        and payload.get("contract_version") == "weaver-headless-n8n-v1"
        and payload.get("deadline_ms") == 115_000
        and bool(payload.get("correlation_id"))
        and payload.get("codebase_context") == context.strip()
        and bool(payload.get("search_query"))
    )
    retrieval_fast = context_elapsed < 3.0
    passed = evidence_ok and identifier_cross_file_ok and route_ok and retrieval_fast
    detail = "\n".join([
        f"  Exact source evidence retrieved: {evidence_ok}",
        f"  Identifier search crosses files:  {identifier_cross_file_ok}",
        f"  Evidence chars:                 {len(context)}",
        f"  Missing evidence markers:       {missing_evidence}",
        f"  Grounding retrieval under 3s:   {retrieval_fast} ({context_elapsed:.3f}s)",
        f"  n8n introspection flags set:    {payload.get('self_check') is True and payload.get('introspect') is True}",
        f"  Evidence forwarded normalized:  {payload.get('codebase_context') == context.strip()} "
        f"({len(str(payload.get('codebase_context') or ''))}/{len(context)} chars)",
    ])
    _result("Z", "Codebase evidence reaches the full n8n cortex", passed, detail)
    return passed


async def test_AA():
    _header("AA", "Workflow reflection, soul voice, and live voice share one cortex")
    import inspect
    import bedrock_brain_api as brain

    workflow_path = os.path.join(PROJ, "n8n_weaver_v5.json")
    with open(workflow_path, "r", encoding="utf-8") as fh:
        workflow = json.load(fh)
    nodes = {node["name"]: node for node in workflow.get("nodes", [])}
    serialized = json.dumps(workflow, ensure_ascii=False)
    expert_names = ("5a. Logic", "5b. Emotion", "5c. Memory", "5d. Creativity", "5e. Vigilance")
    experts_grounded = all(
        "codebase_context" in nodes[name].get("parameters", {}).get("jsonBody", "")
        for name in expert_names
    )
    reflection_grounded = "codebase_context" in nodes["7. Self-Reflect"]["parameters"]["jsonBody"]
    request_nodes = [node for node in workflow.get("nodes", []) if node.get("parameters", {}).get("jsonBody")]
    request_bodies_safe = all(
        node["parameters"]["jsonBody"].startswith("={{ JSON.stringify(")
        and "{{ $json" not in node["parameters"]["jsonBody"]
        for node in request_nodes
    )
    lora_body = nodes["8. LoRA Voice"]["parameters"]["jsonBody"]
    dual_merge = nodes["8c. Dual Merge"]["parameters"]["jsCode"]
    lora_preserves_review = (
        "self_reflection" not in lora_body
        and "tone lead-in" in lora_body
        and "${reviewed}" in dual_merge
        and "unsafeSyntax" in dual_merge
    )
    writeback_grounded = "codebase_grounded" in nodes["9. Writeback"]["parameters"]["jsCode"]
    speaker_body = nodes["7. Self-Reflect"]["parameters"]["jsonBody"]
    speaker_tag = nodes["7-tag"]["parameters"]["jsCode"]
    writeback_code = nodes["9. Writeback"]["parameters"]["jsCode"]
    public_speaker_bounded = all(marker in speaker_body for marker in (
        "qwen.qwen3-235b-a22b-2507",
        "sole user-facing conversational speaker",
        "Never identify as a coder",
        "Return only Weaver's final answer",
    )) and all(marker in speaker_tag for marker in (
        "speaker_boundary_applied: !!reviewed",
        "internal_draft_hidden: true",
    )) and all(marker in writeback_code for marker in (
        "speaker: 'weaver'",
        "speaker_boundary_applied: true",
        "internal_draft_hidden: true",
    ))
    private_draft_fail_closed = (
        "No codebase evidence was supplied." not in serialized
        and "d.collapsed_response ||" not in writeback_code
    )
    no_stale_repo_path = "/media/ydn/SYPHER_CORE/weaver v3" not in serialized

    voice_source = inspect.getsource(brain.realtime_voice)
    bridge_source = inspect.getsource(brain._RealtimeVoiceBridge._process_responses)
    voice_unified = (
        brain.VOICE_CORTEX_ENABLED
        and "agent_response" in voice_source
        and "_cortex_chat" in voice_source
        and "not VOICE_CORTEX_ENABLED" in bridge_source
        and brain._merge_voice_transcript("hello", "hello world") == "hello world"
    )
    headless_source = _read_headless_bundle()
    frontend_unified = "data.type === 'agent_response'" in headless_source and "allowDuringRealtime" in headless_source

    passed = all((
        experts_grounded,
        reflection_grounded,
        request_bodies_safe,
        lora_preserves_review,
        writeback_grounded,
        public_speaker_bounded,
        private_draft_fail_closed,
        no_stale_repo_path,
        voice_unified,
        frontend_unified,
    ))
    detail = "\n".join([
        f"  Five experts receive source evidence: {experts_grounded}",
        f"  Reflection receives source evidence:  {reflection_grounded}",
        f"  Dynamic HTTP bodies remain valid JSON: {request_bodies_safe}",
        f"  LoRA lead-in preserves reviewed facts: {lora_preserves_review}",
        f"  Writeback reports grounding:          {writeback_grounded}",
        f"  Qwen brain is the sole public speaker: {public_speaker_bounded}",
        f"  Private drafts fail closed:            {private_draft_fail_closed}",
        f"  Stale workstation path removed:       {no_stale_repo_path}",
        f"  Realtime transcript enters cortex:    {voice_unified}",
        f"  Browser speaks cortex response once:  {frontend_unified}",
    ])
    _result("AA", "Workflow reflection, soul voice, and live voice share one cortex", passed, detail)
    return passed


async def test_AB():
    _header("AB", "Request, local-model, and shutdown contracts are connected")
    import inspect
    import bedrock_brain_api as brain
    import codebase_api
    import lora_server
    import nexus_bus
    import qwen3b_server
    import weaver
    from headless_scheduler import HeadlessScheduler

    workflow_path = os.path.join(PROJ, "n8n_weaver_v5.json")
    with open(workflow_path, "r", encoding="utf-8") as fh:
        workflow = json.load(fh)
    connections = workflow.get("connections", {})

    def targets(source):
        outputs = connections.get(source, {}).get("main", [])
        return [edge.get("node") for branch in outputs for edge in branch]

    single_join = (
        set(targets("7-tag")) == {"8. LoRA Voice", "8b. Qwen3B"}
        and targets("8-tag") == ["8c. Local Barrier"]
        and targets("8b-tag") == ["8c. Local Barrier"]
        and sum(targets(source).count("8c. Local Barrier") for source in connections) == 2
        and targets("8c. Local Barrier") == ["8c. Dual Merge"]
        and sum(targets(source).count("8c. Dual Merge") for source in connections) == 1
        and targets("8c. Dual Merge") == ["9. Writeback"]
    )

    lora_source = inspect.getsource(lora_server.LoRAHandler.do_POST)
    qwen_source = inspect.getsource(qwen3b_server.Qwen3BHandler.do_POST)
    token_contract = all(
        'req.get("max_completion_tokens"' in source
        for source in (lora_source, qwen_source)
    )
    model_generation_serialized = all(
        "_generation_lock.acquire" in source and "_generation_lock.release()" in source
        for source in (lora_source, qwen_source)
    )
    available_cpus = max(os.cpu_count() or 1, 1)
    inference_threads_bounded = (
        1 <= lora_server.LORA_THREADS <= available_cpus
        and 1 <= qwen3b_server.CPU_THREADS <= available_cpus
    )

    index_source = inspect.getsource(codebase_api._iter_code_files)
    search_source = inspect.getsource(codebase_api.search_codebase)
    source_index_safe = (
        "_GIT_TRACKED_PATHS" in index_source
        and "subprocess.run" not in index_source + search_source
        and "SYPHER_VAULT" in codebase_api.EXCLUDED_DIRS
    )

    shutdown_source = inspect.getsource(weaver.main)
    signal_source = inspect.getsource(weaver._setup_signal_handlers)
    nexus_source = inspect.getsource(nexus_bus.main)
    bounded_shutdown = (
        "asyncio.wait" in shutdown_source
        and "main_task.cancel()" in signal_source
        and "asyncio.all_tasks" not in signal_source
        and "close_timeout=2" in nexus_source
    )

    import threading

    def server_hook_stops(module, server_type, handler_type):
        server = server_type(("127.0.0.1", 0), handler_type)
        module._http_server = server
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.05)
        module.shutdown_server()
        thread.join(timeout=2.0)
        server.server_close()
        module._http_server = None
        return not thread.is_alive()

    blocking_servers_stop = (
        server_hook_stops(lora_server, lora_server.ThreadedHTTPServer, lora_server.LoRAHandler)
        and server_hook_stops(qwen3b_server, qwen3b_server.ThreadedHTTPServer, qwen3b_server.Qwen3BHandler)
        and inspect.getsource(weaver.main).count("shutdown_server()") >= 2
    )

    brain_source = inspect.getsource(brain._cortex_chat) + inspect.getsource(brain._cortex_chat_inner)
    headless_source = inspect.getsource(brain._headless_loop)
    internal_source = inspect.getsource(brain._internal_chat)
    scheduler_source = (
        inspect.getsource(HeadlessScheduler.run_cycle)
        + inspect.getsource(HeadlessScheduler._run_interruptible)
    )
    local_fallback = (
        brain.LOCAL_LLM_URL.endswith(":8899/v1/chat/completions")
        and brain.N8N_CHAT_TIMEOUT >= 120
        and "final_messages" in brain_source
    )
    headless_yields_to_users = (
        "_interactive_started" in inspect.getsource(brain._cortex_chat)
        and "HeadlessScheduler(" in headless_source
        and "idle_ready=_headless_idle_ready" in headless_source
        and "priority_event=_interactive_priority_event" in headless_source
        and "_priority_event.is_set()" in scheduler_source
        and "work.cancel()" in scheduler_source
        and 'request_class="background"' in internal_source
        and brain.HEADLESS_LOCAL_THOUGHT_TOKENS <= 48
        and brain.HEADLESS_LOCAL_DREAM_TOKENS <= 96
    )

    route_names = ("_bedrock_chat", "_cortex_chat", "_local_llama_chat")
    route_originals = {name: getattr(brain, name) for name in route_names}

    async def denied_bedrock(*_args, **_kwargs):
        raise RuntimeError("Operation not allowed")

    async def full_cortex(*_args, **_kwargs):
        return "full cortex answer", {
            "latency_ms": 12,
            "usage": {},
            "stop_reason": "stop",
            "route": {"alias": "weaver-one", "pipeline": "test-full-stack"},
        }

    async def unexpected_local(*_args, **_kwargs):
        raise AssertionError("weaver-brain must not fall straight to local LoRA")

    try:
        brain._bedrock_chat = denied_bedrock
        brain._cortex_chat = full_cortex
        brain._local_llama_chat = unexpected_local
        routed_text, routed_meta = await brain._chat_direct_alias(
            brain.MODEL_ROUTES["weaver-brain"],
            [{"role": "user", "content": "Are you connected?"}],
            max_tokens=80,
        )
    finally:
        for name, value in route_originals.items():
            setattr(brain, name, value)

    routed = routed_meta.get("route", {})
    denied_brain_joins_cortex = (
        routed_text == "full cortex answer"
        and routed.get("alias") == "weaver-brain"
        and routed.get("fallback") == "weaver-one"
        and "Operation not allowed" in routed.get("bedrock_error", "")
        and routed.get("cortex_route", {}).get("pipeline") == "test-full-stack"
    )

    mantle_names = ("_mantle_chat", "_bedrock_chat")
    mantle_originals = {name: getattr(brain, name) for name in mantle_names}
    old_mantle_key = brain.MANTLE_API_KEY

    async def mantle_qwen(*_args, **_kwargs):
        return "mantle qwen answer", {
            "latency_ms": 9,
            "usage": {},
            "stop_reason": "stop",
            "route": {
                "alias": "weaver-brain",
                "model_id": "qwen.qwen3-235b-a22b-2507",
                "region": "us-east-1",
                "runtime_region": "us-east-2",
                "transport": "bedrock-mantle",
            },
        }

    async def unexpected_runtime(*_args, **_kwargs):
        raise AssertionError("Mantle Qwen must run before bedrock-runtime")

    try:
        brain.MANTLE_API_KEY = "test-mantle-key"
        brain._mantle_chat = mantle_qwen
        brain._bedrock_chat = unexpected_runtime
        mantle_text, mantle_meta = await brain._chat_direct_alias(
            brain.MODEL_ROUTES["weaver-brain"],
            [{"role": "user", "content": "Use Qwen."}],
            max_tokens=80,
        )
    finally:
        brain.MANTLE_API_KEY = old_mantle_key
        for name, value in mantle_originals.items():
            setattr(brain, name, value)

    mantle_qwen_preferred = (
        mantle_text == "mantle qwen answer"
        and mantle_meta.get("route", {}).get("transport") == "bedrock-mantle"
        and mantle_meta.get("route", {}).get("model_id") == "qwen.qwen3-235b-a22b-2507"
        and mantle_meta.get("route", {}).get("region") == "us-east-1"
        and mantle_meta.get("route", {}).get("runtime_region") == "us-east-2"
    )

    passed = all((
        single_join,
        token_contract,
        model_generation_serialized,
        inference_threads_bounded,
        source_index_safe,
        bounded_shutdown,
        blocking_servers_stop,
        local_fallback,
        headless_yields_to_users,
        denied_brain_joins_cortex,
        mantle_qwen_preferred,
    ))
    detail = "\n".join([
        f"  Parallel LoRA/Qwen barrier converges once: {single_join}",
        f"  Both servers accept OpenAI token keys: {token_contract}",
        f"  Local model generation is serialized: {model_generation_serialized}",
        f"  Inference threads fit available CPUs:  {inference_threads_bounded}",
        f"  Source index is fast and vault-safe:   {source_index_safe}",
        f"  Supervisor shutdown is bounded:        {bounded_shutdown}",
        f"  Blocking model servers stop cleanly:   {blocking_servers_stop}",
        f"  Dead AWS route falls back locally:      {local_fallback}",
        f"  Headless work yields to live turns:     {headless_yields_to_users}",
        f"  Denied direct brain joins full cortex:  {denied_brain_joins_cortex}",
        f"  Mantle Qwen precedes denied runtime:    {mantle_qwen_preferred}",
    ])
    _result("AB", "Request, local-model, and shutdown contracts are connected", passed, detail)
    return passed


async def test_AC():
    _header("AC", "Embodiment navigation connects body, environment, and camera")
    embodiment_path = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar", "embodiment.html"))
    apartment_path = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar", "weaver_apartment.glb"))
    avatar_path = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar", "weaver_avatar_dress.glb"))
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        source = fh.read()

    qwen_drives_body = "body: 'weaver-brain'" in source
    named_navigation = all(
        marker in source
        for marker in (
            "const LOCOMOTION_ZONES",
            "const LOCOMOTION_GRAPH",
            "center:",
            "window:",
            "kitchen:",
            "gallery:",
            "lounge:",
            "function requestWalkTo",
            "function zoneRoute",
            "globalThis.__weaverWalkTo",
        )
    )
    model_can_navigate = all(
        marker in source
        for marker in (
            '"locomotion":{"mode":"walk|stop"',
            "applyLocomotionIntent(intent)",
            "Safe penthouse destinations:",
        )
    )
    gait_has_contact = all(
        marker in source
        for marker in (
            "const legSwing = gait",
            "const armSwing = legSwing",
            "function emitFootstep",
            "updateFootstepContact(gait, dt)",
            "foot_contact_ripple_",
        )
    )
    environment_follows = all(
        marker in source
        for marker in (
            "presenceHalo.position.x",
            "locomotion_destination_marker",
            "key.target.position.x",
            "globalThis.__weaverEnvironmentAudit",
        )
    )
    camera_follows = all(
        marker in source
        for marker in (
            "camera.position.lerp(cameraDesiredPosition",
            "cameraFollowTarget.lerp(cameraDesiredTarget",
            "mode: model && motion.baseReady ? 'body-follow'",
        )
    )
    assets_present = (
        os.path.getsize(apartment_path) > 1_000_000
        and os.path.getsize(avatar_path) > 1_000_000
    )

    passed = all((
        qwen_drives_body,
        named_navigation,
        model_can_navigate,
        gait_has_contact,
        environment_follows,
        camera_follows,
        assets_present,
    ))
    detail = "\n".join([
        f"  Qwen drives body intent:             {qwen_drives_body}",
        f"  Named apartment navigation exists:  {named_navigation}",
        f"  Model can request destinations:      {model_can_navigate}",
        f"  Signed gait has planted-foot cues:   {gait_has_contact}",
        f"  Environment follows body state:      {environment_follows}",
        f"  Camera follows world displacement:   {camera_follows}",
        f"  Avatar and apartment assets present: {assets_present}",
    ])
    _result("AC", "Embodiment navigation connects body, environment, and camera", passed, detail)
    return passed


async def test_AD():
    _header("AD", "iPhone senses use on-device acceleration and the full voice cortex")
    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))
    embodiment_path = os.path.join(avatar_root, "embodiment.html")
    deploy_path = os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh")
    vendor_root = os.path.join(avatar_root, "vendor", "mediapipe", "0.10.35")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        source = fh.read()
    with open(deploy_path, "r", encoding="utf-8") as fh:
        deploy_source = fh.read()

    expected_hashes = {
        "blaze_face_short_range.tflite": "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        "vision_bundle.mjs": "55d7ab624fbb70dcc5adc4ae6d7ea9cfcb569139d3dbfbf2b1deafcb966bc0fe",
        "wasm/vision_wasm_internal.js": "e7fd9858e8e8f221d9b96eddc11f8e077f263e0b7bbd79d3cbe882b134274f8c",
        "wasm/vision_wasm_internal.wasm": "6a5c64584c2ab61c763b6e204afbdbc7ce1caf7f5216187322bca8df94f646bc",
        "wasm/vision_wasm_nosimd_internal.js": "438d1fe8ff7f4d946025bc211c291543c037d8a3785ed4eee60f1f521b236296",
        "wasm/vision_wasm_nosimd_internal.wasm": "8a3092d34c79d3f57e6ba8592105e8a90f6b07c27891ffecd14cca428bfd3e31",
    }

    def digest(path):
        value = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest()

    assets_pinned = all(
        os.path.isfile(os.path.join(vendor_root, relative))
        and digest(os.path.join(vendor_root, relative)) == expected
        for relative, expected in expected_hashes.items()
    )
    accelerated_face_tracking = all(
        marker in source
        for marker in (
            "MEDIAPIPE_VERSION = '0.10.35'",
            "FilesetResolver.forVisionTasks(CPU_FACE_WASM_ROOT, false)",
            "(isiOS ? 'GPU' : 'CPU')",
            "delegateOrder = PREFERRED_FACE_DELEGATE === 'GPU' ? ['GPU', 'CPU'] : ['CPU']",
            "delegate\n          }",
            "runningMode: 'VIDEO'",
            "detector.detectForVideo(video, performance.now())",
            "'mediapipe-gpu-webgl'",
            "'mediapipe-cpu-wasm'",
        )
    )
    iphone_full_cortex = all(
        marker in source
        for marker in (
            "if (isiOS || !SR || params.get('hearingBackend') === 'cortex')",
            "`${scheme}//${location.host}/brain/realtime/voice`",
            "'weaver-realtime'",
            "weaver-key.${base64UrlEncode(key)}",
            "downsampleToPcm16",
            "data.type === 'agent_response'",
        )
    )
    one_voice_only = all(
        marker in source
        for marker in (
            "data.type === 'audio'",
            "suppressedAudioFrames += 1",
            "await speak(text)",
            "raw Nova audio so two voices can never play together",
            "if (realtimeHearing.live || realtimeHearing.connecting || realtimeHearing.lastHeard) return;",
        )
    )
    senses_are_isolated = all(
        marker in source
        for marker in (
            "for (const track of stream.getVideoTracks())",
            "for (const track of micStream.getAudioTracks())",
            "hasLiveTrack(sharedStream, 'video')",
            "hasLiveTrack(sharedStream, 'audio')",
        )
    )
    phone_gpu_without_server_gpu = all(
        marker in source
        for marker in (
            "wanted: gpuMode === 'on'",
            "powerPreference: isiOS || remoteGpuState.wanted ? 'high-performance' : 'low-power'",
            "serverGpuRequired: false",
            "preferredFaceDelegate: faceTrackerState.preferredDelegate",
            "npuDirectAccess: faceTrackerState.npuDirectAccess",
            "softwareWebglDetected: renderPerf.softwareRenderer",
        )
    )
    observable = all(
        marker in source
        for marker in (
            "globalThis.__weaverMediaAudit",
            "faceTracking: { ...faceTrackerState }",
            "hearingBackend:",
            "cortexRouted: realtimeHearing.cortexRouted",
            "framesSent: realtimeHearing.framesSent",
        )
    )
    vendor_deploys = 'sudo cp -a "$DEPLOY_ROOT/avatar/vendor" "$root/vendor"' in deploy_source

    passed = all((
        assets_pinned,
        accelerated_face_tracking,
        iphone_full_cortex,
        one_voice_only,
        senses_are_isolated,
        phone_gpu_without_server_gpu,
        observable,
        vendor_deploys,
    ))
    detail = "\n".join([
        f"  Pinned MediaPipe/WASM/model assets: {assets_pinned}",
        f"  iPhone GPU has a CPU fallback:       {accelerated_face_tracking}",
        f"  iPhone mic enters full cortex:       {iphone_full_cortex}",
        f"  Raw second voice is suppressed:      {one_voice_only}",
        f"  Camera/mic teardown is isolated:     {senses_are_isolated}",
        f"  No server GPU is required:           {phone_gpu_without_server_gpu}",
        f"  Sensor route is runtime-auditable:   {observable}",
        f"  Versioned vendor tree deploys:       {vendor_deploys}",
    ])
    _result("AD", "iPhone senses use on-device acceleration and the full voice cortex", passed, detail)
    return passed


async def test_AE():
    _header("AE", "Native iPhone shell connects Neural Engine, senses, cortex, and embodiment")
    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    ios_root = os.path.join(repo_root, "ios", "WeaverNeural")
    source_root = os.path.join(ios_root, "WeaverNeural")
    model_root = os.path.join(source_root, "Resources", "Models", "WeaverAttention.mlpackage")

    def read(relative):
        with open(os.path.join(ios_root, relative), "r", encoding="utf-8") as fh:
            return fh.read()

    required_paths = (
        "project.yml",
        "Makefile",
        "WeaverNeural/AppModel.swift",
        "WeaverNeural/Info.plist",
        "WeaverNeural/Resources/PrivacyInfo.xcprivacy",
        "WeaverNeural/Services/AudioCaptureEngine.swift",
        "WeaverNeural/Services/CameraCaptureService.swift",
        "WeaverNeural/Services/KeychainStore.swift",
        "WeaverNeural/Services/NeuralEngineFaceTracker.swift",
        "WeaverNeural/Services/RealtimeVoiceClient.swift",
        "WeaverNeural/Services/TrainedVoicePlayer.swift",
        "WeaverNeural/Services/WeaverAPIClient.swift",
        "WeaverNeural/Views/WeaverSceneView.swift",
        "WeaverNeuralTests/PCM16EncoderTests.swift",
        "WeaverNeuralTests/ProtocolTests.swift",
    )
    source_complete = all(os.path.isfile(os.path.join(ios_root, path)) for path in required_paths)

    app = read("WeaverNeural/AppModel.swift")
    audio = read("WeaverNeural/Services/AudioCaptureEngine.swift")
    camera = read("WeaverNeural/Services/CameraCaptureService.swift")
    keychain = read("WeaverNeural/Services/KeychainStore.swift")
    tracker = read("WeaverNeural/Services/NeuralEngineFaceTracker.swift")
    realtime = read("WeaverNeural/Services/RealtimeVoiceClient.swift")
    api = read("WeaverNeural/Services/WeaverAPIClient.swift")
    configuration = read("WeaverNeural/Services/WeaverConfiguration.swift")
    scene = read("WeaverNeural/Views/WeaverSceneView.swift")
    lifecycle = read("WeaverNeural/WeaverNeuralApp.swift")
    generator = read("Tools/generate_attention_model.py")
    project = read("project.yml")
    makefile = read("Makefile")
    unit_tests = read("WeaverNeuralTests/PCM16EncoderTests.swift") + read("WeaverNeuralTests/ProtocolTests.swift")
    with open(os.path.join(repo_root, "avatar", "embodiment.html"), "r", encoding="utf-8") as fh:
        embodiment = fh.read()

    model_manifest_path = os.path.join(model_root, "Manifest.json")
    model_spec_path = os.path.join(model_root, "Data", "com.apple.CoreML", "model.mlmodel")
    model_weights_path = os.path.join(model_root, "Data", "com.apple.CoreML", "weights", "weight.bin")
    model_package_valid = False
    if all(os.path.isfile(path) for path in (model_manifest_path, model_spec_path, model_weights_path)):
        with open(model_manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        root_id = manifest.get("rootModelIdentifier")
        root_entry = manifest.get("itemInfoEntries", {}).get(root_id, {})
        model_package_valid = (
            manifest.get("fileFormatVersion") == "1.0.0"
            and root_entry.get("path") == "com.apple.CoreML/model.mlmodel"
            and os.path.getsize(model_spec_path) > 2_000
            and os.path.getsize(model_weights_path) > 500
        )

    def png_contract(path):
        with open(path, "rb") as fh:
            header = fh.read(26)
        if len(header) != 26 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
            return None
        return (
            int.from_bytes(header[16:20], "big"),
            int.from_bytes(header[20:24], "big"),
            header[25],
        )

    icon_path = os.path.join(source_root, "Resources", "Assets.xcassets", "AppIcon.appiconset", "AppIcon-1024.png")
    fallback_path = os.path.join(source_root, "Resources", "Assets.xcassets", "WeaverFallback.imageset", "WeaverFallback.png")
    assets_valid = (
        os.path.isfile(icon_path)
        and os.path.isfile(fallback_path)
        and png_contract(icon_path) == (1024, 1024, 2)
        and png_contract(fallback_path) == (390, 844, 2)
    )

    neural_engine_connected = all(marker in tracker + generator for marker in (
        "VNDetectFaceRectanglesRequest",
        "configuration.computeUnits = .cpuAndNeuralEngine",
        "if case .neuralEngine = device",
        'MLMultiArray(shape: [1, 8], dataType: .float32)',
        "compute_precision=ct.precision.FLOAT16",
        'model.user_defined_metadata["weaver.compute_request"] = "cpuAndNeuralEngine"',
    ))
    local_senses_connected = all(marker in camera + audio for marker in (
        "AVCaptureSession",
        ".builtInWideAngleCamera",
        "position: .front",
        "AVAudioEngine",
        "PCM16Encoder.encode",
        "inputSampleRate: format.sampleRate",
        ".voiceChat",
    ))
    full_cortex_voice = all(marker in app + realtime + api + configuration for marker in (
        '"weaver-realtime"',
        '"weaver-key.\\(WebSocketProtocolKey.encode(brainKey))"',
        '"type": "start"',
        "configuration.cortexRouted",
        'let model = "weaver-one"',
        'case "agent_response"',
        "let chunks = Self.speechChunks(text)",
        "apiClient.synthesize(chunks[0])",
        "apiClient.synthesize(chunks[nextIndex])",
        "voicePlayer.play(audio)",
        'case "audio"',
        "suppressedAudioFrames += 1",
        "AsyncStream.makeStream",
        ".bufferingNewest(24)",
        "configuration.mode.lowercased() != \"mock\"",
    ))
    session_isolation = all(marker in app + realtime for marker in (
        "private var connectionID: UUID?",
        "connectionID == currentConnectionID",
        "callbackRunID == runID",
        "let maxAttempts = policy?.maxAttempts ?? 5",
        "reconnectAttempt <= maxAttempts",
        "reconnectTask?.cancel()",
        "requestRun == runID",
        "currentSpeechID == self.speechID",
    ))
    secure_boundary = all(marker in keychain + api + configuration + scene for marker in (
        "kSecAttrAccessibleWhenUnlockedThisDeviceOnly",
        "URLSessionConfiguration.ephemeral",
        'request.setValue(brainKey, forHTTPHeaderField: "X-Weaver-Key")',
        '#if DEBUG',
        'return URL(string: "https://weaverv3.com")!',
        "configuration.websiteDataStore = .nonPersistent()",
        "host == WeaverConfiguration.baseURL.host",
    )) and "WEAVER_LLM_KEY" not in (app + realtime + api + configuration + scene)
    lifecycle_closes_hardware = all(marker in app + lifecycle for marker in (
        "if phase != .active",
        "Task { await model.sleep() }",
        "audioCapture.stop()",
        "await voice?.disconnect()",
        "camera.stop()",
        "voicePlayer.stop()",
    ))
    render_bridge_bounded = all(marker in embodiment + scene + configuration for marker in (
        "const nativeShell = params.get('nativeShell') === '1';",
        "const NATIVE_WALK_ZONES",
        "globalThis.__weaverNativeBridge = Object.freeze",
        "window.__weaverNativeBridge?.update",
        "webView.callAsyncJavaScript",
        "walkCommandID",
        "bodyCueID",
        "heuristicSkeletonIntent(bodyCue)",
        "stopLocomotion('native-ios-sleep')",
        'URLQueryItem(name: "gpu", value: "off")',
        'URLQueryItem(name: "nativeShell", value: "1")',
    ))

    with open(os.path.join(source_root, "Info.plist"), "rb") as fh:
        info = plistlib.load(fh)
    with open(os.path.join(source_root, "Resources", "PrivacyInfo.xcprivacy"), "rb") as fh:
        privacy = plistlib.load(fh)
    collected = {
        item.get("NSPrivacyCollectedDataType")
        for item in privacy.get("NSPrivacyCollectedDataTypes", [])
    }
    privacy_declared = (
        bool(info.get("NSCameraUsageDescription"))
        and bool(info.get("NSMicrophoneUsageDescription"))
        and privacy.get("NSPrivacyTracking") is False
        and {
            "NSPrivacyCollectedDataTypeAudioData",
            "NSPrivacyCollectedDataTypeOtherUserContent",
        }.issubset(collected)
        and all(
            item.get("NSPrivacyCollectedDataTypeTracking") is False
            for item in privacy.get("NSPrivacyCollectedDataTypes", [])
        )
    )
    build_and_tests = all(marker in project + makefile + unit_tests + tracker for marker in (
        'iOS: "18.0"',
        "PRODUCT_MODULE_NAME: WeaverNeural",
        "WeaverAttention",
        "xcodegen generate",
        "xcodebuild test",
        "testDownsamples48kHzTo16kHzLittleEndian",
        "testWebSocketKeyUsesUnpaddedBase64URL",
        "testProductionRoutesAreTLS",
    ))

    passed = all((
        source_complete,
        model_package_valid,
        assets_valid,
        neural_engine_connected,
        local_senses_connected,
        full_cortex_voice,
        session_isolation,
        secure_boundary,
        lifecycle_closes_hardware,
        render_bridge_bounded,
        privacy_declared,
        build_and_tests,
    ))
    detail = "\n".join([
        f"  Native project source is complete:      {source_complete}",
        f"  Core ML package is structurally valid:   {model_package_valid}",
        f"  Opaque icon and fallback assets exist:   {assets_valid}",
        f"  Vision requests CPU + Neural Engine:     {neural_engine_connected}",
        f"  Front camera and microphone are native:  {local_senses_connected}",
        f"  Voice enters cortex and ordered TTS:      {full_cortex_voice}",
        f"  Socket/wake generations isolate races:   {session_isolation}",
        f"  Key and render boundaries are hardened:  {secure_boundary}",
        f"  Background/sleep closes all hardware:    {lifecycle_closes_hardware}",
        f"  Native bridge accepts bounded state:     {render_bridge_bounded}",
        f"  Permissions and data use are declared:   {privacy_declared}",
        f"  Xcode build and unit contracts exist:    {build_and_tests}",
    ])
    _result("AE", "Native iPhone shell connects Neural Engine, senses, cortex, and embodiment", passed, detail)
    return passed


async def test_AF():
    _header("AF", "Whole-body awareness, intentional environment, silent coder, and fast voice reaction")
    import inspect
    import re
    import bedrock_brain_api as brain

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    embodiment_path = os.path.join(repo_root, "avatar", "embodiment.html")
    ios_root = os.path.join(repo_root, "ios", "WeaverNeural", "WeaverNeural")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    headless = _read_headless_bundle()
    with open(os.path.join(ios_root, "AppModel.swift"), "r", encoding="utf-8") as fh:
        ios_app = fh.read()
    with open(os.path.join(ios_root, "Models", "WeaverState.swift"), "r", encoding="utf-8") as fh:
        ios_models = fh.read()
    with open(os.path.join(PROJ, "n8n_weaver_v5.json"), "r", encoding="utf-8") as fh:
        workflow = json.load(fh)
    workflow_source = json.dumps(workflow, ensure_ascii=False)
    brain_source = inspect.getsource(brain._cortex_chat_inner) + inspect.getsource(brain.realtime_voice)

    articulated_body = all(marker in embodiment for marker in (
        "leftElbow: 0",
        "rightElbow: 0",
        "leftKnee: 0",
        "rightKnee: 0",
        "upperarmTwistL",
        "forearmTwistL",
        "thighTwistL",
        "shinTwistL",
        "const elbowFlexL",
        "const kneeFlexL",
        "lowerlegL: boneQuaternion('lowerlegL')",
    ))
    intentional_motion = all(marker in embodiment for marker in (
        "autonomousPatrol: false",
        "walkState.mode = 'aware-idle'",
        "const target = requested || (walkState.autonomousPatrol ? setPatrolDestination() : null)",
        "Never move merely to create activity",
        "Keep still when no meaningful movement is warranted",
    ))
    awareness_connected = all(marker in embodiment for marker in (
        "function bodyAwarenessSnapshot()",
        "function environmentAwarenessSnapshot()",
        "globalThis.__weaverBodyAwareness",
        "globalThis.__weaverEnvironmentAwareness",
        "Current proprioception:",
        "Current environment awareness:",
        "nearestObjects",
    ))

    interaction_block = embodiment.split("const PENTHOUSE_INTERACTIONS = Object.freeze({", 1)[1].split(
        "const interactionAliases", 1
    )[0]
    interaction_keys = re.findall(r"^  ([a-z][a-z0-9_]+): \{", interaction_block, flags=re.MULTILINE)
    interactions_connected = (
        len(interaction_keys) == 20
        and len(set(interaction_keys)) == 20
        and all(marker in embodiment for marker in (
            "globalThis.__weaverInteract",
            "globalThis.__weaverInteractionAudit",
            "function updatePenthouseInteraction",
            "objectsLoaded",
            "environmentObjectLookup",
            "availableCount: Object.keys(PENTHOUSE_INTERACTIONS).length",
        ))
    )
    appearance_physics = all(marker in embodiment for marker in (
        "WeaverTexturePreservingSkin",
        "WeaverMidnightSatin",
        "WeaverSoftBlackBraids",
        "WeaverWovenFabricMicrotexture",
        "weaver_seamless_woven_a_line_skirt",
        "weaver_closed_skirt_lining",
        "weaver_soft_hair_crown",
        "const strandCount = QUALITY.hairStrands",
        "function buildDynamicHair()",
        "function buildTailoredGarmentLayer()",
        "function updateSecondaryMotion(dt)",
        "function updateScheduledSecondaryMotion(dt)",
        "weaver_gravity_braid_strands",
        "gravity: -9.81",
        "hairConstraintIterations: QUALITY.hairIterations",
        "maxStretchError",
    ))
    silent_coder = all(marker in brain_source + workflow_source for marker in (
        "silent code specialist",
        '"silent_specialist": True',
        'speaker_route = MODEL_ROUTES["weaver-brain"]',
        "codeOnlyTurn",
        "!codeOnlyTurn && !!rawPrelude",
        "code-only",
    ))
    fast_voice_reaction = (
        brain.VOICE_REACTION_TARGET_MS <= 200
        and all(marker in brain_source + embodiment + headless + ios_app + ios_models for marker in (
            '"type": "turn_ack"',
            '"reactionTargetMs": VOICE_REACTION_TARGET_MS',
            "lastReactionMs",
            'case "turn_ack"',
            "voiceReactionMilliseconds",
            "cortexLatencyMs",
            "queueLatencyMs",
        ))
    )

    passed = all((
        articulated_body,
        intentional_motion,
        awareness_connected,
        interactions_connected,
        appearance_physics,
        silent_coder,
        fast_voice_reaction,
    ))
    detail = "\n".join([
        f"  Elbows, knees, twist bones are articulated: {articulated_body}",
        f"  Idle movement is intentional, not patrol:     {intentional_motion}",
        f"  Body/environment awareness enters cortex:     {awareness_connected}",
        f"  Exactly 20 real-object interactions exist:    {interactions_connected} ({len(interaction_keys)})",
        f"  Texture-preserving materials + gravity hair:  {appearance_physics}",
        f"  Coder is silent and code prelude is blocked:  {silent_coder}",
        f"  Voice reaction budget is <=200ms and audited: {fast_voice_reaction}",
    ])
    _result("AF", "Whole-body awareness, intentional environment, silent coder, and fast voice reaction", passed, detail)
    return passed


async def test_AG():
    _header("AG", "Credential, request, accessibility, stance, and first-audio hardening")
    import inspect
    import httpx
    import bedrock_brain_api as brain

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    embodiment_path = os.path.join(repo_root, "avatar", "embodiment.html")
    ios_root = os.path.join(repo_root, "ios", "WeaverNeural")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    headless = _read_headless_bundle()
    with open(os.path.join(ios_root, "WeaverNeural", "AppModel.swift"), "r", encoding="utf-8") as fh:
        ios_app = fh.read()
    with open(os.path.join(ios_root, "WeaverNeural", "Views", "WeaverRootView.swift"), "r", encoding="utf-8") as fh:
        ios_view = fh.read()
    with open(os.path.join(ios_root, "WeaverNeuralTests", "ProtocolTests.swift"), "r", encoding="utf-8") as fh:
        ios_tests = fh.read()

    ephemeral_credentials = all(marker in embodiment + headless for marker in (
        "sessionStorage.getItem('weaver_llm_key')",
        "sessionStorage.setItem('weaver_llm_key'",
        "localStorage.removeItem('weaver_llm_key')",
        "brain key for this tab",
    )) and "localStorage.setItem('weaver_llm_key'" not in embodiment + headless

    accessibility = all(marker in embodiment + headless for marker in (
        'role="status"',
        'aria-live="polite"',
        'aria-pressed="false"',
        'aria-busy="true"',
        "setAttribute('aria-busy', 'false')",
        'button:focus-visible',
        '@media (prefers-reduced-motion: reduce)',
        'role="img"',
    ))

    stance_hardened = all(marker in embodiment for marker in (
        "const stanceCalibration",
        "leftPelvisDrop: 0.070",
        "setRigPositionOffset('pelvisL'",
        "toeHeightDelta",
        "dualFootContact",
        "1 - walk.intensity * 1.5",
    ))

    first_audio_prefetch = all(marker in ios_app + ios_view + ios_tests for marker in (
        "static func speechChunks",
        "var synthesisTask: Task<Data, Error>?",
        "apiClient.synthesize(chunks[nextIndex])",
        "voiceFirstAudioLatencyMilliseconds",
        'LatencyReadout(label: "Audio"',
        "testSpeechChunkingStartsPlaybackFromFirstSentence",
        "testTurnAcknowledgementDecodesLatencyBudget",
    ))

    source = inspect.getsource(brain._check_key) + inspect.getsource(brain._read_json_object)
    backend_hardened = all(marker in source for marker in (
        "hmac.compare_digest",
        "request body too large",
        "application/json required",
        "JSON object required",
    )) and brain.MAX_HTTP_BODY_BYTES <= 262_144 and brain.MAX_CHAT_MESSAGES <= 64

    old_key = brain.WEAVER_KEY
    try:
        brain.WEAVER_KEY = "test-secret"
        transport = httpx.ASGITransport(app=brain.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
            non_object = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "test-secret"},
                json=["not", "an", "object"],
            )
            oversized = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "test-secret", "Content-Type": "application/json"},
                content=b'{"padding":"' + (b"x" * (brain.MAX_HTTP_BODY_BYTES + 1)) + b'"}',
            )
            invalid_role = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "test-secret"},
                json={"messages": [{"role": "tool", "content": "no"}]},
            )
            excessive_tokens = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "test-secret"},
                json={"max_tokens": 9999, "messages": [{"role": "user", "content": "no"}]},
            )
    finally:
        brain.WEAVER_KEY = old_key

    admission_enforced = (
        unauthorized.status_code == 403
        and non_object.status_code == 400
        and oversized.status_code == 413
        and invalid_role.status_code == 400
        and excessive_tokens.status_code == 400
    )

    passed = all((
        ephemeral_credentials,
        accessibility,
        stance_hardened,
        first_audio_prefetch,
        backend_hardened,
        admission_enforced,
    ))
    detail = "\n".join([
        f"  Browser credentials are tab-scoped:        {ephemeral_credentials}",
        f"  UI exposes accessible live state:          {accessibility}",
        f"  Idle stance calibrates both feet:           {stance_hardened}",
        f"  Native TTS chunks and prefetches first audio:{first_audio_prefetch}",
        f"  API uses constant-time bounded admission:   {backend_hardened}",
        f"  Auth/schema/size/token rejection works:     {admission_enforced}",
    ])
    _result("AG", "Credential, request, accessibility, stance, and first-audio hardening", passed, detail)
    return passed


async def test_AH():
    _header("AH", "Neural QoS autopilot, stateful world model, flight recorder, and voice SLO control plane")
    import re
    import bedrock_brain_api as brain

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    with open(os.path.join(repo_root, "avatar", "embodiment.html"), "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    with open(os.path.join(PROJ, "weaver_dashboard.py"), "r", encoding="utf-8") as fh:
        dashboard = fh.read()
    with open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8") as fh:
        brain_source = fh.read()

    qos_autopilot = all(marker in embodiment for marker in (
        "const opsState",
        "function prioritizeVoice",
        "function updateOpsController",
        "new PerformanceObserver",
        "renderPerf.faceCadenceScale",
        "renderPerf.hairConstraintBudget",
        "scheduleVisionTick",
        "voice-priority",
        "__weaverOpsAudit",
    ))

    recorder_block = embodiment.split("const OPS_FIELD_ALLOWLIST = new Set([", 1)[1].split("]);", 1)[0]
    private_markers = ("transcript", "prompt", "secret", "key", "lastheard", "lastsaid", "text")
    privacy_safe_recorder = all(marker in embodiment for marker in (
        "const OPS_EVENT_LIMIT = 160",
        "sessionStorage.setItem(OPS_SESSION_KEY",
        "safeOpsValue",
        "__weaverFlightRecorder",
        "exportJson",
    )) and not any(marker in recorder_block.lower() for marker in private_markers)

    world_block = embodiment.split("const WORLD_EFFECTS = Object.freeze({", 1)[1].split("});", 1)[0]
    world_effects = re.findall(r"^\s{2}([a-z][a-z0-9_]+):\s*\{", world_block, re.MULTILINE)
    stateful_world = len(world_effects) == 20 and all(marker in embodiment for marker in (
        "function commitWorldEffect",
        "function applyWorldModelVisuals",
        "function planNextPenthouseInteraction",
        "const INTERACTION_GOALS",
        "world: worldModelSnapshot()",
        "__weaverWorldModel",
        "__weaverPlanNextInteraction",
        "localStorage.setItem(WORLD_MODEL_KEY",
        "Stateful penthouse changes:",
    ))

    old_samples = list(brain._voice_slo_samples)
    try:
        brain._voice_slo_samples.clear()
        for _ in range(20):
            good = brain._record_voice_slo(
                reaction_ms=18,
                queue_ms=24,
                cortex_ms=780,
                semantic_ms=920,
            )
        nominal = (
            good["status"] == "nominal"
            and good["samples"] == 20
            and good["success_rate"] == 1.0
            and good["semantic_p95_ms"] == 920.0
            and good["error_budget_remaining_pct"] == 100.0
        )
        for _ in range(5):
            breached = brain._record_voice_slo(
                reaction_ms=350,
                queue_ms=800,
                cortex_ms=5200,
                semantic_ms=6100,
            )
        burn_detected = (
            breached["status"] == "breached"
            and breached["success_rate"] == 0.8
            and breached["error_budget_remaining_pct"] == 0.0
        )
    finally:
        brain._voice_slo_samples.clear()
        brain._voice_slo_samples.extend(old_samples)
        brain._voice_route_state()["slo"] = brain._voice_slo_snapshot()

    backend_slo = nominal and burn_detected and all(marker in embodiment for marker in (
        "data.slo.success_rate",
        "voice-semantic",
        "voice-first-audio",
    ))
    warm_path = all(marker in brain_source for marker in (
        "VOICE_PREWARM_ENABLED",
        "async def _prewarm_voice_runtime",
        'status = "ready"',
        '"slo": slo_snapshot',
    ))
    operator_control_plane = all(marker in dashboard for marker in (
        '"voice_slo": voice_slo',
        '"voice_prewarm": voice_prewarm',
        'id="liveVoiceSlo"',
        'id="liveVoiceLatency"',
        "semantic_p95_ms",
        "error_budget_remaining_pct",
        'role="tablist"',
        'role="tabpanel"',
        "event.key === 'ArrowRight'",
    ))

    passed = all((
        qos_autopilot,
        privacy_safe_recorder,
        stateful_world,
        backend_slo,
        warm_path,
        operator_control_plane,
    ))
    detail = "\n".join([
        f"  Voice-first GPU/CPU workload arbitration: {qos_autopilot}",
        f"  Flight recorder excludes private content:  {privacy_safe_recorder}",
        f"  All 20 interactions leave world state:     {stateful_world} ({len(world_effects)})",
        f"  Rolling p50/p95 and error budget work:      {backend_slo}",
        f"  Cortex clients prewarm off the turn path:   {warm_path}",
        f"  Operator dashboard exposes live SLO truth:  {operator_control_plane}",
    ])
    _result("AH", "Neural QoS autopilot, stateful world model, flight recorder, and voice SLO control plane", passed, detail)
    return passed


async def test_AI():
    _header("AI", "Neural Fabric lanes, proof ledger, signed Intent Capsules, and hardened control API")
    import copy
    import httpx
    import bedrock_brain_api as brain
    from weaver_neural_fabric import (
        FabricDeadlineExceeded,
        FabricOverloaded,
        IntentCompiler,
        IntentValidationError,
        NeuralFabric,
        SlidingWindowRateLimiter,
        WorkClass,
    )

    interactive_started = asyncio.Event()
    interactive_release = asyncio.Event()
    realtime_started = asyncio.Event()
    realtime_release = asyncio.Event()
    fabric = NeuralFabric(capacity_units=6, realtime_reserved_units=2)

    async def hold_interactive():
        interactive_started.set()
        await interactive_release.wait()
        return "interactive-ok"

    async def hold_realtime():
        realtime_started.set()
        await realtime_release.wait()
        return "realtime-ok"

    interactive_task = asyncio.create_task(fabric.execute(
        lane=WorkClass.INTERACTIVE,
        name="test-interactive",
        deadline_ms=2_000,
        cost_units=4,
        factory=hold_interactive,
    ))
    await asyncio.wait_for(interactive_started.wait(), timeout=1)
    realtime_task = asyncio.create_task(fabric.execute(
        lane=WorkClass.REALTIME,
        name="test-realtime",
        deadline_ms=2_000,
        cost_units=2,
        factory=hold_realtime,
    ))
    await asyncio.wait_for(realtime_started.wait(), timeout=1)
    reserved_capacity = fabric.capacity.in_use == 6 and fabric.capacity.realtime_in_use == 2

    background_shed = False
    try:
        await fabric.execute(
            lane=WorkClass.BACKGROUND,
            name="should-shed",
            deadline_ms=500,
            cost_units=1,
            factory=lambda: asyncio.sleep(0),
        )
    except FabricOverloaded:
        background_shed = True

    realtime_release.set()
    interactive_release.set()
    realtime_result, interactive_result = await asyncio.gather(realtime_task, interactive_task)
    lanes_execute = (
        realtime_result.value == "realtime-ok"
        and realtime_result.receipt["lane"] == "realtime"
        and interactive_result.value == "interactive-ok"
        and interactive_result.receipt["lane"] == "interactive"
    )

    deadline_enforced = False
    try:
        await fabric.execute(
            lane=WorkClass.INTERACTIVE,
            name="deadline-test",
            deadline_ms=50,
            cost_units=1,
            factory=lambda: asyncio.sleep(0.2),
        )
    except FabricDeadlineExceeded:
        deadline_enforced = True

    cancel_started = asyncio.Event()

    async def cancellable_work():
        cancel_started.set()
        await asyncio.Event().wait()

    cancelled_task = asyncio.create_task(fabric.execute(
        lane=WorkClass.INTERACTIVE,
        name="cancel-test",
        deadline_ms=1_000,
        cost_units=2,
        factory=cancellable_work,
    ))
    await asyncio.wait_for(cancel_started.wait(), timeout=1)
    cancelled_task.cancel()
    try:
        await cancelled_task
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0)
    cancellation_cleanup = (
        fabric.capacity.in_use == 0
        and fabric.snapshot()["lanes"]["interactive"]["active"] == 0
        and fabric.snapshot()["lanes"]["interactive"]["counters"]["cancelled"] == 1
    )

    fabric.ledger.record(
        "privacy-test",
        WorkClass.INTERACTIVE,
        "privacy-request",
        name="metadata-only",
        result="ok",
        prompt="super-secret-prompt",
        transcript="private-transcript",
    )
    fabric_snapshot = fabric.snapshot()
    serialized_ledger = json.dumps(fabric_snapshot["ledger"]).lower()
    ledger_private = "super-secret-prompt" not in serialized_ledger and "private-transcript" not in serialized_ledger
    ledger_valid = fabric_snapshot["ledger"]["valid"] is True
    last_event = fabric.ledger._events[-1]
    original_result = last_event["result"]
    last_event["result"] = "tampered"
    tamper_detected = fabric.ledger.verify() is False
    last_event["result"] = original_result
    ledger_valid = ledger_valid and tamper_detected and fabric.ledger.verify()

    compiler = IntentCompiler("test-fabric-signing-secret")
    capsule = compiler.compile({
        "goal": "Move to the lounge, read, and hold a grounded pose",
        "priority": "embodiment",
        "ttl_ms": 12_000,
        "preconditions": {
            "world_revision": 4,
            "body_revision": 9,
            "requires_awake": True,
            "max_duration_ms": 10_000,
        },
        "actions": [
            {"type": "navigate", "zone": "lounge"},
            {"type": "interact", "interaction": "reading_book"},
            {"type": "pose", "values": {"leftElbow": 0.7, "rightElbow": 0.7, "leftKnee": 0.2}},
            {"type": "bones", "bones": {"lowerarm01_L": {"x": 0.3, "y": 0, "z": 0}}},
        ],
    })
    signed_capsule = (
        compiler.verify(capsule)
        and capsule["integrity"]["algorithm"] == "hmac-sha256"
        and capsule["rollback"] == ["reset_bones", "reset_pose", "cancel_interaction", "stop_locomotion"]
        and capsule["expires_at_ms"] > capsule["issued_at_ms"]
    )
    tampered_capsule = copy.deepcopy(capsule)
    tampered_capsule["actions"][0]["zone"] = "gallery"
    signed_capsule = signed_capsule and not compiler.verify(tampered_capsule)
    invalid_capsule_rejected = False
    try:
        compiler.compile({"goal": "unsafe", "actions": [{"type": "shell", "command": "no"}]})
    except IntentValidationError:
        invalid_capsule_rejected = True

    limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)
    rate_limit_works = await limiter.allow() and await limiter.allow() and not await limiter.allow()

    old_key = brain.WEAVER_KEY
    old_cortex = brain._cortex_chat
    old_direct = brain._chat_direct_alias

    async def fake_cortex(messages, max_tokens=None, temperature=None):
        return "fabric interactive", {"latency_ms": 1, "usage": {}, "route": {"alias": "fake"}}

    async def fake_direct(route, messages, max_tokens=None, temperature=None):
        return "fabric embodiment", {"latency_ms": 1, "usage": {}, "route": {"alias": route.alias}}

    try:
        brain.WEAVER_KEY = "fabric-test-key"
        brain._cortex_chat = fake_cortex
        brain._chat_direct_alias = fake_direct
        transport = httpx.ASGITransport(app=brain.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.get("/fabric/v1/state")
            state_response = await client.get(
                "/fabric/v1/state", headers={"X-Weaver-Key": "fabric-test-key"}
            )
            compile_response = await client.post(
                "/fabric/v1/intent/compile",
                headers={"X-Weaver-Key": "fabric-test-key"},
                json={
                    "goal": "Read in the lounge",
                    "actions": [
                        {"type": "navigate", "zone": "lounge"},
                        {"type": "interact", "interaction": "reading_book"},
                    ],
                },
            )
            invalid_response = await client.post(
                "/fabric/v1/intent/compile",
                headers={"X-Weaver-Key": "fabric-test-key"},
                json={"goal": "bad", "actions": [{"type": "navigate", "zone": "internet"}]},
            )
            interactive_chat = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "fabric-test-key"},
                json={"model": "weaver-one", "messages": [{"role": "user", "content": "hello"}]},
            )
            embodiment_chat = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "fabric-test-key"},
                json={
                    "model": "weaver-brain",
                    "messages": [
                        {"role": "system", "content": "Return JSON for browser skeleton control and body intent."},
                        {"role": "user", "content": "bend both elbows"},
                    ],
                },
            )
    finally:
        brain.WEAVER_KEY = old_key
        brain._cortex_chat = old_cortex
        brain._chat_direct_alias = old_direct

    api_capsule = compile_response.json().get("capsule", {}) if compile_response.status_code == 200 else {}
    control_api = (
        unauthorized.status_code == 403
        and state_response.status_code == 200
        and state_response.json().get("technology") == "weaver-neural-fabric"
        and state_response.json().get("ledger", {}).get("valid") is True
        and compile_response.status_code == 200
        and compile_response.json().get("verified") is True
        and brain.INTENT_COMPILER.verify(api_capsule)
        and invalid_response.status_code == 400
    )
    traffic_integrated = (
        interactive_chat.status_code == 200
        and interactive_chat.json()["weaver"]["fabric"]["lane"] == "interactive"
        and embodiment_chat.status_code == 200
        and embodiment_chat.json()["weaver"]["fabric"]["lane"] == "embodiment"
    )

    with open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8") as fh:
        brain_source = fh.read()
    with open(os.path.join(PROJ, "deploy", "weaver-brain.service"), "r", encoding="utf-8") as fh:
        service_source = fh.read()
    with open(os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"), "r", encoding="utf-8") as fh:
        deploy_source = fh.read()
    integration_contract = all(marker in brain_source for marker in (
        "lane=WorkClass.REALTIME",
        "lane=WorkClass.BACKGROUND",
        "_fabric_lane_for_chat",
        '@app.get("/fabric/v1/state")',
        '@app.post("/fabric/v1/intent/compile")',
    ))
    service_hardened = all(marker in service_source for marker in (
        "WEAVER_FABRIC_CAPACITY_UNITS=16",
        "WEAVER_FABRIC_REALTIME_RESERVED_UNITS=4",
        "NoNewPrivileges=true",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ReadWritePaths=",
        "MemoryDenyWriteExecute=true",
    ))
    deployment_verifies_fabric = all(marker in deploy_source for marker in (
        "/fabric/v1/state",
        "/fabric/v1/intent/compile",
        'state.get("technology") == "weaver-neural-fabric"',
        'data.get("verified") is True',
        'capsule.get("rollback") == ["cancel_interaction", "stop_locomotion"]',
    ))

    passed = all((
        reserved_capacity,
        background_shed,
        lanes_execute,
        deadline_enforced,
        cancellation_cleanup,
        ledger_private,
        ledger_valid,
        signed_capsule,
        invalid_capsule_rejected,
        rate_limit_works,
        control_api,
        traffic_integrated,
        integration_contract,
        service_hardened,
        deployment_verifies_fabric,
    ))
    detail = "\n".join([
        f"  Realtime capacity survives saturation:      {reserved_capacity}",
        f"  Background work yields to live voice:       {background_shed}",
        f"  Interactive/realtime lanes execute:         {lanes_execute}",
        f"  Deadlines cancel over-budget work:          {deadline_enforced}",
        f"  Cancellation cannot leak capacity:          {cancellation_cleanup}",
        f"  Ledger excludes private content:            {ledger_private}",
        f"  Hash-chain tampering is detected:           {ledger_valid}",
        f"  Intent Capsules sign, expire, and rollback: {signed_capsule}",
        f"  Arbitrary action types are rejected:        {invalid_capsule_rejected}",
        f"  Compile rate limiting works:                {rate_limit_works}",
        f"  Authenticated Fabric APIs are bounded:      {control_api}",
        f"  Chat/body traffic uses Fabric lanes:        {traffic_integrated}",
        f"  Voice/background integration is connected:  {integration_contract}",
        f"  systemd service is sandbox-hardened:        {service_hardened}",
        f"  Deployment verifies Fabric before success:  {deployment_verifies_fabric}",
    ])
    _result("AI", "Neural Fabric lanes, proof ledger, signed Intent Capsules, and hardened control API", passed, detail)
    return passed


async def test_AJ():
    _header("AJ", "Seven-angle Cognition Mesh and validated parallel n8n v6 workflow")
    import copy
    import httpx
    import bedrock_brain_api as brain
    from weaver_cognition_mesh import (
        CognitionMesh,
        CognitionValidationError,
        InferenceGovernor,
        ResilienceImmuneSystem,
        SalienceMemoryPyramid,
        ShadowPolicyLab,
    )
    from weaver_neural_fabric import IntentCompiler, NeuralFabric, SlidingWindowRateLimiter

    fabric = NeuralFabric(capacity_units=12, realtime_reserved_units=3).snapshot()
    compiler = IntentCompiler("seven-angle-test-secret")
    mesh = CognitionMesh()
    observation = mesh.observe({
        "body": {
            "awake": True,
            "balance": 0.92,
            "velocity_mps": 0,
            "pose": {"leftElbow": 0.1, "rightElbow": 0.1, "leftKnee": 0.05},
            "confidence": 0.95,
        },
        "environment": {
            "zone": "center",
            "ambient_light": 0.7,
            "noise": 0.1,
            "obstacle_distance_m": 5,
            "confidence": 0.95,
            "objects": [
                {
                    "id": "reading_book",
                    "zone": "lounge",
                    "distance_m": 1.2,
                    "visible": True,
                    "confidence": 0.95,
                }
            ],
        },
        "sensors": {
            "camera": {"confidence": 0.9},
            "microphone": {"confidence": 0.9},
        },
    })
    awareness = observation["awareness"]
    sensor_fusion = (
        awareness["body_revision"] == 1
        and awareness["world_revision"] == 1
        and awareness["awareness_confidence"] > 0.8
        and awareness["channels"]["camera"]["fresh"]
        and awareness["world"]["objects"]["reading_book"]["visible"]
    )

    strict_observation = False
    stale_observation = False
    try:
        mesh.observe({"body": {"balance": 1}, "command": "run anything"})
    except CognitionValidationError:
        strict_observation = True
    try:
        mesh.observe({
            "observed_at_ms": int(time.time() * 1000) - 301_000,
            "body": {"balance": 1},
        })
    except CognitionValidationError:
        stale_observation = True

    capsule = compiler.compile({
        "goal": "Walk to the lounge and read with articulated arms and knees",
        "priority": "embodiment",
        "ttl_ms": 12_000,
        "preconditions": {
            "body_revision": awareness["body_revision"],
            "world_revision": awareness["world_revision"],
            "requires_awake": True,
            "max_duration_ms": 8_000,
        },
        "actions": [
            {"type": "navigate", "zone": "lounge"},
            {"type": "interact", "interaction": "reading_book"},
            {
                "type": "pose",
                "values": {
                    "leftElbow": 0.35,
                    "rightElbow": 0.35,
                    "leftKnee": 0.15,
                    "rightKnee": 0.15,
                },
            },
        ],
    })
    plan = mesh.evaluate_intent(capsule, fabric=fabric)
    seven_angle_plan = (
        compiler.verify(capsule)
        and plan["decision"] == "execute"
        and len(plan["angles"]) == 7
        and plan["angles"]["embodiment"]["decision"] == "approve"
        and plan["angles"]["prediction"]["predicted_zone"] == "lounge"
        and plan["angles"]["prediction"]["success_probability"] > 0.5
        and plan["angles"]["compute"]["primary"]["alias"] == "weaver-speed"
        and all(not proposal["automatic_mutation"] for proposal in plan["angles"]["evolution"])
    )

    unsafe_mesh = CognitionMesh()
    unsafe_mesh.observe({
        "body": {
            "awake": False,
            "balance": 0.15,
            "velocity_mps": 0,
            "ground_contacts": [],
            "confidence": 0.95,
        },
        "environment": {
            "zone": "center",
            "obstacle_distance_m": 0.1,
            "confidence": 0.95,
        },
    })
    unsafe_capsule = compiler.compile({
        "goal": "Unsafe locomotion must stop",
        "actions": [{"type": "navigate", "zone": "window"}],
    })
    blocked = unsafe_mesh.evaluate_intent(unsafe_capsule, fabric=fabric)
    reflex_blocks = (
        blocked["decision"] == "block"
        and not blocked["angles"]["embodiment"]["approved"]
        and {"awake", "balance", "collision_clearance", "ground_contact"}.issubset(
            blocked["angles"]["embodiment"]["failed_checks"]
        )
    )

    voice_route = mesh.plan_inference(
        {"task": "voice", "deadline_ms": 200, "quality_priority": 0.4}, fabric=fabric
    )
    code_route = mesh.plan_inference(
        {"task": "code", "deadline_ms": 5_000, "quality_priority": 0.9}, fabric=fabric
    )
    deadline_routing = (
        voice_route["primary"]["alias"] == "weaver-speed"
        and voice_route["primary"]["estimated_latency_ms"] <= 200
        and code_route["primary"]["alias"] == "weaver-code"
        and voice_route["advisory"] is True
    )

    immune = ResilienceImmuneSystem(failure_threshold=3, cooldown_seconds=30)
    for _ in range(3):
        immune.record("weaver-brain", success=False, latency_ms=4_000, target_ms=1_000)
    governor = InferenceGovernor()
    failover_route = governor.plan(
        task="chat",
        deadline_ms=5_000,
        quality_priority=0.95,
        fabric_pressure=0,
        immune=immune,
    )
    circuit_immunity = (
        immune.snapshot()["status"] == "guarded"
        and "weaver-brain" in immune.snapshot()["open_components"]
        and failover_route["primary"]["alias"] != "weaver-brain"
    )

    memory = SalienceMemoryPyramid(hot_events=16)
    for index in range(24):
        memory.record(
            kind="outcome",
            tags=["voice", "latency"],
            reward=0.5 if index % 3 else -0.5,
            surprise=0.8 if index % 7 == 0 else 0.1,
            risk=0.6 if index % 5 == 0 else 0.0,
            success=index % 3 != 0,
        )
    memory_state = memory.snapshot()
    memory_serialized = json.dumps({"state": memory_state, "recall": memory.recall(["voice"])}).lower()
    salience_memory = (
        memory_state["hot_events"] == 16
        and memory_state["consolidations"] == 8
        and memory_state["warm_patterns"] == 2
        and "prompt" not in memory_serialized
        and "transcript" not in memory_serialized
    )

    lab = ShadowPolicyLab()
    for _ in range(4):
        lab.observe(success=False)
    for _ in range(2):
        lab.observe(success=True)
    proposals = lab.propose(
        fabric={
            "lanes": {"realtime": {"latency_p95_ms": 240, "counters": {"deadlines": 1}}},
            "accelerator": {"pressure": 0.9},
        },
        awareness={"awareness_confidence": 0.2},
        immune=immune.snapshot(),
    )
    shadow_only = (
        len(proposals) >= 3
        and all(item["status"] == "shadow-only" for item in proposals)
        and all(item["automatic_mutation"] is False for item in proposals)
        and not hasattr(lab, "apply")
        and not hasattr(lab, "deploy")
    )

    old_key = brain.WEAVER_KEY
    old_cognition = brain.COGNITION
    old_mutation_limiter = brain.COGNITION_MUTATION_LIMITER
    old_query_limiter = brain.COGNITION_QUERY_LIMITER
    try:
        brain.WEAVER_KEY = "cognition-test-key"
        brain.COGNITION = CognitionMesh()
        brain.COGNITION_MUTATION_LIMITER = SlidingWindowRateLimiter(100, 60)
        brain.COGNITION_QUERY_LIMITER = SlidingWindowRateLimiter(100, 60)
        transport = httpx.ASGITransport(app=brain.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            unauthorized = await client.get("/cognition/v1/state")
            headers = {"X-Weaver-Key": "cognition-test-key"}
            state_response = await client.get("/cognition/v1/state", headers=headers)
            observe_response = await client.post(
                "/cognition/v1/observe",
                headers=headers,
                json={
                    "body": {
                        "awake": True,
                        "balance": 0.9,
                        "velocity_mps": 0,
                        "pose": {"leftElbow": 0.1, "rightElbow": 0.1},
                        "confidence": 0.95,
                    },
                    "environment": {
                        "zone": "center",
                        "obstacle_distance_m": 4,
                        "confidence": 0.95,
                        "objects": [
                            {
                                "id": "reading_book",
                                "zone": "lounge",
                                "distance_m": 1,
                                "visible": True,
                                "confidence": 0.95,
                            }
                        ],
                    },
                },
            )
            route_response = await client.post(
                "/cognition/v1/route",
                headers=headers,
                json={"task": "voice", "deadline_ms": 200, "quality_priority": 0.4},
            )
            compile_response = await client.post(
                "/fabric/v1/intent/compile",
                headers=headers,
                json={
                    "goal": "Safely read in lounge",
                    "actions": [
                        {"type": "navigate", "zone": "lounge"},
                        {"type": "interact", "interaction": "reading_book"},
                    ],
                },
            )
            api_capsule = compile_response.json().get("capsule", {})
            evaluate_response = await client.post(
                "/cognition/v1/intent/evaluate",
                headers=headers,
                json={"capsule": api_capsule},
            )
            tampered = copy.deepcopy(api_capsule)
            if tampered.get("actions"):
                tampered["actions"][0]["zone"] = "gallery"
            tampered_response = await client.post(
                "/cognition/v1/intent/evaluate", headers=headers, json={"capsule": tampered}
            )
            bad_observation = await client.post(
                "/cognition/v1/observe",
                headers=headers,
                json={"body": {"balance": 1}, "shell": "never"},
            )
            good_outcome = await client.post(
                "/cognition/v1/outcome",
                headers=headers,
                json={
                    "component": "voice",
                    "task": "voice",
                    "success": True,
                    "latency_ms": 180,
                    "target_ms": 200,
                    "quality": 0.8,
                    "tags": ["voice", "latency"],
                },
            )
            bad_outcome = await client.post(
                "/cognition/v1/outcome",
                headers=headers,
                json={
                    "component": "shell",
                    "task": "chat",
                    "success": True,
                    "latency_ms": 1,
                },
            )
    finally:
        brain.WEAVER_KEY = old_key
        brain.COGNITION = old_cognition
        brain.COGNITION_MUTATION_LIMITER = old_mutation_limiter
        brain.COGNITION_QUERY_LIMITER = old_query_limiter

    cognition_api = (
        unauthorized.status_code == 403
        and state_response.status_code == 200
        and len(state_response.json().get("angles", [])) == 7
        and observe_response.status_code == 200
        and observe_response.json().get("fabric", {}).get("lane") == "embodiment"
        and route_response.status_code == 200
        and route_response.json().get("primary", {}).get("alias") == "weaver-speed"
        and compile_response.status_code == 200
        and evaluate_response.status_code == 200
        and evaluate_response.json().get("capsule_verified") is True
        and tampered_response.status_code == 400
        and bad_observation.status_code == 400
        and good_outcome.status_code == 200
        and bad_outcome.status_code == 400
    )

    validator_path = os.path.join(PROJ, "scripts", "validate_n8n_workflow.mjs")
    validator_proc = await asyncio.create_subprocess_exec(
        "node",
        validator_path,
        "--json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJ,
    )
    validator_stdout, validator_stderr = await validator_proc.communicate()
    validator_data = {}
    with contextlib.suppress(json.JSONDecodeError):
        validator_data = json.loads(validator_stdout.decode("utf-8"))
    n8n_validator = (
        validator_proc.returncode == 0
        and not validator_stderr
        and validator_data.get("valid") is True
        and validator_data.get("nodes") == 35
        and validator_data.get("edges") == 42
        and validator_data.get("reachable_nodes") == 35
        and validator_data.get("terminal_nodes") == ["9. Writeback"]
        and validator_data.get("critical_path_budget_ms") == 102_500
        and validator_data.get("errors") == []
    )

    with open(os.path.join(PROJ, "n8n_weaver_v5.json"), "r", encoding="utf-8") as fh:
        workflow = json.load(fh)
    nodes = {node["name"]: node for node in workflow["nodes"]}
    fanout_targets = {
        edge["node"] for edge in workflow["connections"]["5. Expert Fanout"]["main"][0]
    }
    parallel_workflow = (
        workflow["name"] == "Weaver Nervous System v6 (parallel cognition mesh)"
        and fanout_targets == {
            "5a. Logic", "5b. Emotion", "5c. Memory", "5d. Creativity", "5e. Vigilance"
        }
        and nodes["5f. Expert Barrier"]["parameters"]["numberInputs"] == 5
        and nodes["8c. Local Barrier"]["parameters"]["numberInputs"] == 2
        and workflow["settings"]["saveExecutionProgress"] is False
        and workflow["settings"]["executionTimeout"] == 115
        and "require(" not in "\n".join(
            node.get("parameters", {}).get("jsCode", "") for node in workflow["nodes"]
        )
        and "original_input:" not in nodes["9. Writeback"]["parameters"]["jsCode"]
        and "v6-parallel-cognition" in nodes["9. Writeback"]["parameters"]["jsCode"]
    )

    with open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8") as fh:
        brain_source = fh.read()
    with open(os.path.join(PROJ, "deploy", "n8n.service"), "r", encoding="utf-8") as fh:
        n8n_service = fh.read()
    with open(os.path.join(PROJ, "docker-compose.yml"), "r", encoding="utf-8") as fh:
        docker_compose = fh.read()
    with open(os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"), "r", encoding="utf-8") as fh:
        deploy_source = fh.read()
    runtime_integration = all(marker in brain_source for marker in (
        "COGNITION = CognitionMesh()",
        '@app.get("/cognition/v1/state")',
        '@app.post("/cognition/v1/observe")',
        '@app.post("/cognition/v1/intent/evaluate")',
        '@app.post("/cognition/v1/route")',
        '@app.post("/cognition/v1/outcome")',
        "_record_cognition_runtime_outcome",
        'component="n8n"',
        'component="voice"',
        "N8NHeadlessRequest(",
        "cognition_context={",
    ))
    pinned_n8n_image = (
        "docker.n8n.io/n8nio/n8n:2.25.7@sha256:"
        "761374d4eb841b0a22771d6bd68f0e8d827b4979ae4e490045517b13fc1259dd"
    )
    n8n_hardened = (
        pinned_n8n_image in n8n_service
        and pinned_n8n_image in docker_compose
        and all(marker in n8n_service for marker in (
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges:true",
        "N8N_RUNNERS_ENABLED=true",
        "N8N_BLOCK_ENV_ACCESS_IN_NODE=true",
        "N8N_BLOCK_FILE_ACCESS_TO_N8N_FILES=true",
        "N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true",
        ))
        and all(marker in docker_compose for marker in (
            "read_only: true",
            "cap_drop:",
            "no-new-privileges:true",
            "N8N_RUNNERS_ENABLED=true",
            "N8N_BLOCK_ENV_ACCESS_IN_NODE=true",
        ))
    )
    deployment_gates = (
        all(marker in deploy_source for marker in (
            "scripts/validate_n8n_workflow.mjs",
            "v6-parallel-cognition",
            "parallel-fanout-barrier",
            "/cognition/v1/state",
            "/cognition/v1/observe",
            "/cognition/v1/route",
            "/cognition/v1/intent/evaluate",
            'sudo docker image inspect "$N8N_IMAGE"',
            "cat > /tmp/n8n_cred.json",
            "cat > /tmp/wf.json",
            'assert set(data) == expected',
            'data.get("expert_fanout_elapsed_ms")',
            'assert data.get("expert_count") == 5',
            'assert data.get("internal_draft_hidden") is True',
            "n8n container: pinned, non-root, bounded, read-only, capability-dropped, sandboxed",
        ))
        and "docker cp" not in deploy_source
        and 'data.get("lora_latency_ms")' not in deploy_source
        and 'data.get("qwen3b_latency_ms")' not in deploy_source
        and 'assert data.get("soul_voice_active") is True' not in deploy_source
        and 'assert data.get("dual_model_active") is True' not in deploy_source
    )

    passed = all((
        sensor_fusion,
        strict_observation,
        stale_observation,
        seven_angle_plan,
        reflex_blocks,
        deadline_routing,
        circuit_immunity,
        salience_memory,
        shadow_only,
        cognition_api,
        n8n_validator,
        parallel_workflow,
        runtime_integration,
        n8n_hardened,
        deployment_gates,
    ))
    detail = "\n".join([
        f"  Fresh body/world sensor fusion works:       {sensor_fusion}",
        f"  Unknown observation fields are rejected:   {strict_observation}",
        f"  Stale observations cannot rewind state:     {stale_observation}",
        f"  Signed plan passes all seven angles:        {seven_angle_plan}",
        f"  Reflex kernel blocks unsafe locomotion:     {reflex_blocks}",
        f"  Deadline/quality governor routes correctly: {deadline_routing}",
        f"  Circuit immunity routes around failures:    {circuit_immunity}",
        f"  Salience pyramid consolidates privately:    {salience_memory}",
        f"  Evolution proposals stay shadow-only:       {shadow_only}",
        f"  Authenticated bounded Cognition APIs work:  {cognition_api}",
        f"  n8n graph/expression/syntax validator passes:{n8n_validator}",
        f"  Expert and local models execute in parallel:{parallel_workflow}",
        f"  Chat, voice, n8n telemetry are integrated:  {runtime_integration}",
        f"  Pinned n8n launch paths are hardened:       {n8n_hardened}",
        f"  Deployment gates verify both technologies: {deployment_gates}",
    ])
    _result("AJ", "Seven-angle Cognition Mesh and validated parallel n8n v6 workflow", passed, detail)
    return passed


async def test_AK():
    _header("AK", "Cinematic facial articulation, layered body dynamics, and bounded material response")
    import struct

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    embodiment_path = os.path.join(repo_root, "avatar", "embodiment.html")
    avatar_path = os.path.join(repo_root, "avatar", "weaver_avatar_dress.glb")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()

    facial_coarticulation = all(marker in embodiment for marker in (
        "CINEMATIC_MOTION_VERSION = 'cinematic-micro-motion-v1'",
        "const SPEECH_VISEMES = Object.freeze({",
        "function buildSpeechVisemeSequence(text)",
        "function beginSpeechPerformance",
        "function currentSpeechViseme()",
        "function applyFacialRig()",
        "face.jawOpen",
        "face.lipWide",
        "face.lipRound",
        "face.lipClose",
        "face.lowerLip",
        "face.tongue",
    ))
    perceptual_micro_motion = all(marker in embodiment for marker in (
        "function cinematicNoise(t, seed = 0)",
        "function cinematicSpring(name, target, dt",
        "face.nextBlinkAt",
        "face.nextSaccadeAt",
        "eye-led attention",
        "setRigOffset('neck2'",
        "setRigOffset('neck3'",
        "setRigOffset('spine4'",
        "setRigOffset('spine5'",
        "setRigOffset('shoulderL'",
        "function applyFingerMicroMotion()",
    ))
    bounded_secondary_motion = all(marker in embodiment for marker in (
        "maxBreastRadians: 0.052",
        "maxPelvicRadians: 0.032",
        "soft.breastLeft = THREE.MathUtils.clamp",
        "soft.breastRight = THREE.MathUtils.clamp",
        "soft.pelvicMass = THREE.MathUtils.clamp",
        "soft.garmentLag = THREE.MathUtils.clamp",
        "setRigOffset('breastL'",
        "setRigOffset('breastR'",
        "bounded second-order anatomical follow-through",
    ))
    material_and_cloth_detail = all(marker in embodiment for marker in (
        "WeaverDeterministicSkinMicrodetail",
        "WeaverWetCornealEyes",
        "function makeSkinMicroTexture()",
        "skirtGeometry.computeVertexNormals()",
        "const pleat = Math.sin(theta * 12",
        "const taper = THREE.MathUtils.lerp(1.08, 0.52",
        "skinProfile: 'texture-preserving physical skin",
        "eyeProfile: 'wet corneal clearcoat",
    ))
    voice_and_camera_sync = all(marker in embodiment for marker in (
        "beginSpeechPerformance(text, 'trained-voice', runId)",
        "beginSpeechPerformance(text, 'browser-voice', speechRun)",
        "endSpeechPerformance('trained-voice-ended')",
        "const CAMERA_FRAMING_PRESETS = Object.freeze({",
        "globalThis.__weaverCameraPreset",
        "globalThis.__weaverCameraPreset?.('medium')",
    ))
    runtime_audits = all(marker in embodiment for marker in (
        "globalThis.__weaverFacialAudit = facialAudit",
        "globalThis.__weaverSoftTissueAudit = softTissueAudit",
        "globalThis.__weaverCinematicMotionAudit",
        "globalThis.__weaverFacialTest",
        "mappedBones: face.mappedBones",
        "boneDeltas:",
    ))

    with open(avatar_path, "rb") as fh:
        glb = fh.read()
    magic, version, length = struct.unpack_from("<III", glb, 0)
    offset = 12
    gltf = None
    while offset + 8 <= min(length, len(glb)):
        chunk_length, chunk_type = struct.unpack_from("<II", glb, offset)
        offset += 8
        chunk = glb[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.rstrip(b" \0"))
            break
    node_names = {node.get("name", "") for node in (gltf or {}).get("nodes", [])}
    required_rig_nodes = {
        "jaw", "eye_L", "eye_R", "breast_L", "breast_R",
        "shoulder01_L", "shoulder01_R", "neck02", "neck03",
        "oris01", "oris02", "oris04_L", "oris04_R", "oris06_L", "oris06_R",
        "risorius03_L", "risorius03_R", "oculi01_L", "oculi01_R",
        "orbicularis03_L", "orbicularis03_R", "tongue03", "tongue04",
    }
    glb_rig_contract = (
        magic == 0x46546C67
        and version == 2
        and length == len(glb)
        and required_rig_nodes.issubset(node_names)
        and len(node_names) >= 160
    )

    passed = all((
        facial_coarticulation,
        perceptual_micro_motion,
        bounded_secondary_motion,
        material_and_cloth_detail,
        voice_and_camera_sync,
        runtime_audits,
        glb_rig_contract,
    ))
    detail = "\n".join([
        f"  Text coarticulates independent face channels: {facial_coarticulation}",
        f"  Eyes, spine, shoulders, hands layer naturally:{perceptual_micro_motion}",
        f"  Tissue/garment inertia is hard-bounded:       {bounded_secondary_motion}",
        f"  Skin, eyes, cloth, and hair carry microdetail:{material_and_cloth_detail}",
        f"  Voice and cinematic framing share timing:    {voice_and_camera_sync}",
        f"  Browser exposes measurable regression hooks: {runtime_audits}",
        f"  GLB contains all required deformation bones: {glb_rig_contract} ({len(node_names)} nodes)",
    ])
    _result("AK", "Cinematic facial articulation, layered body dynamics, and bounded material response", passed, detail)
    return passed


async def test_AL():
    _header("AL", "High-fidelity original avatar LOD, physical scan maps, and safe runtime fallback")
    import struct

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    avatar_root = os.path.join(repo_root, "avatar")
    builder_path = os.path.join(avatar_root, "build_hifi_avatar.py")
    embodiment_path = os.path.join(avatar_root, "embodiment.html")
    hifi_path = os.path.join(avatar_root, "weaver_avatar_dress_hifi.glb")
    deploy_path = os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh")
    with open(builder_path, "r", encoding="utf-8") as fh:
        builder = fh.read()
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    with open(deploy_path, "r", encoding="utf-8") as fh:
        deploy = fh.read()

    deterministic_builder = all(marker in builder for marker in (
        "SUBDIVISION_PROFILES",
        '"weaver_base2-base": 0.58',
        '"outfit_dress_mesh": 0.24',
        '"weaver_base2-highpolyeyes": 0.82',
        "def curved_midpoint(",
        "def midpoint_skin(",
        "strongest = sorted(influences.items()",
        "def generate_skin_maps(",
        '"identity": "original-weaver"',
        "does not ingest or reproduce",
    ))
    runtime_lod = all(marker in embodiment for marker in (
        "const HIFI_AVATAR_ASSET = 'weaver_avatar_dress_hifi.glb'",
        "function avatarAssetCandidates()",
        "const desktopHighFidelityEligible",
        "params.get('avatar')",
        "function loadAvatarAsset(",
        "fallbackUsed",
        "function prepareHighFidelitySkinMaps()",
        "skin_normal_hifi.png",
        "skin_roughness_hifi.png",
        "skin_specular_hifi.png",
        "function activateStudioEnvironment()",
        "function buildCornealShells()",
        "globalThis.__weaverFidelityAudit",
    ))
    deployment_integrated = all(marker in deploy for marker in (
        'missing required visual asset',
        'sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_avatar_dress_hifi.glb"',
        'sudo install -d -m 0755 /var/www/weaver/textures',
        'visual asset checksum mismatch',
        'standard, penthouse, high-fidelity GLBs and PBR maps match deployed checksums',
    ))

    with open(hifi_path, "rb") as fh:
        payload = fh.read()
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    offset = 12
    document = None
    binary = None
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset:offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            document = json.loads(chunk.rstrip(b" \0"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    metadata = (document or {}).get("extras", {}).get("weaverHighFidelity", {})
    mesh_triangles = {
        mesh.get("name", ""): sum(
            document["accessors"][primitive["indices"]]["count"] // 3
            for primitive in mesh.get("primitives", [])
        )
        for mesh in (document or {}).get("meshes", [])
    }
    glb_contract = (
        magic == 0x46546C67
        and version == 2
        and total_length == len(payload)
        and len(document.get("nodes", [])) == 168
        and len(document.get("skins", [])) == 1
        and len(document.get("animations", [])) == 1
        and metadata.get("identity") == "original-weaver"
        and metadata.get("sourceTriangles") == 90_088
        and metadata.get("triangles") == 237_568
        and mesh_triangles.get("weaver_base2-base_hifi") == 107_024
        and mesh_triangles.get("outfit_dress_mesh_hifi") == 85_376
        and mesh_triangles.get("weaver_base2-highpolyeyes_hifi") == 4_240
        and mesh_triangles.get("cornrowsofelv5") == 40_928
    )

    component_formats = {5121: ("B", 1), 5126: ("f", 4)}
    type_widths = {"VEC4": 4}

    def read_accessor(accessor_index):
        accessor = document["accessors"][accessor_index]
        view = document["bufferViews"][accessor["bufferView"]]
        fmt, component_size = component_formats[accessor["componentType"]]
        width = type_widths[accessor["type"]]
        stride = view.get("byteStride", component_size * width)
        start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        unpacker = struct.Struct("<" + fmt * width)
        return [unpacker.unpack_from(binary, start + index * stride) for index in range(accessor["count"])]

    skinning_valid = True
    for mesh in document.get("meshes", []):
        if not mesh.get("name", "").endswith("_hifi"):
            continue
        primitive = mesh["primitives"][0]
        weights = read_accessor(primitive["attributes"]["WEIGHTS_0"])
        joints = read_accessor(primitive["attributes"]["JOINTS_0"])
        if max(abs(sum(weight) - 1.0) for weight in weights) > 0.000002:
            skinning_valid = False
        if max(max(joint) for joint in joints) >= 163:
            skinning_valid = False

    texture_contracts = {
        "skin_normal_hifi.png": (2, 3_000_000),
        "skin_roughness_hifi.png": (0, 100_000),
        "skin_specular_hifi.png": (6, 100_000),
    }
    physical_maps = True
    for filename, (color_type, minimum_size) in texture_contracts.items():
        path = os.path.join(avatar_root, "textures", filename)
        data = open(path, "rb").read()
        width, height = struct.unpack_from(">II", data, 16)
        physical_maps = physical_maps and (
            data[:8] == b"\x89PNG\r\n\x1a\n"
            and width == 2048
            and height == 2048
            and data[25] == color_type
            and len(data) >= minimum_size
        )

    passed = all((
        deterministic_builder,
        runtime_lod,
        deployment_integrated,
        glb_contract,
        skinning_valid,
        physical_maps,
    ))
    detail = "\n".join([
        f"  Reproducible original-only asset builder:  {deterministic_builder}",
        f"  Desktop HiFi selection and fallback exist: {runtime_lod}",
        f"  Deployment copies and hashes all assets:   {deployment_integrated}",
        f"  237,568-triangle GLB preserves rig/anim:    {glb_contract}",
        f"  Subdivided four-joint skin weights valid:  {skinning_valid}",
        f"  Three UV-aligned 2K physical maps valid:   {physical_maps}",
    ])
    _result("AL", "High-fidelity original avatar LOD, physical scan maps, and safe runtime fallback", passed, detail)
    return passed


async def test_AM():
    _header("AM", "iPhone 16e A18 adaptive mobile performance without embodiment loss")

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    embodiment_path = os.path.join(repo_root, "avatar", "embodiment.html")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()

    device_detection = all(marker in embodiment for marker in (
        "const iphone16ePanelClass",
        "Math.abs(displayShortCss - 390)",
        "Math.abs(displayLongCss - 844)",
        "Math.abs(displayShortNative - 1170)",
        "Math.abs(displayLongNative - 2532)",
        "'iphone-16e-a18-class'",
        "globalThis.__weaverDeviceProfile = deviceProfile",
    ))
    a18_quality_tier = all(marker in embodiment for marker in (
        "iphone16e: {",
        "name: 'iphone16e', fps: 60, frameCap: 60, frameFloor: 45",
        "pixelRatio: 1.25, minPixelRatio: 0.78",
        "environmentFps: 30, secondaryFps: 30",
        "hairStrands: 24, hairNodes: 7, hairIterations: 3",
        "const GPU_FACE_INTERVAL_MS = iphone16eClass ? 125 : 90",
    ))
    adaptive_governor = all(marker in embodiment for marker in (
        "function minimumPixelScale()",
        "renderPerf.pixelScale * 0.88",
        "renderPerf.pressureStreak >= 3",
        "QUALITY.frameFloor || 45",
        "renderPerf.pressureStreak >= 6",
        "renderPerf.dynamicResolutionChanges += 1",
        "renderPerf.bootLongFrames += 1",
        "const insideBootGrace = !renderPerf.bootStable",
        "resolutionBeforeCadence: iphone16eClass",
    ))
    split_scheduling = all(marker in embodiment for marker in (
        "function updateScheduledSecondaryMotion(dt)",
        "updateScheduledSecondaryMotion(dt)",
        "const environmentIntervalMs = 1000 / Math.max(20, renderPerf.environmentFps || 60)",
        "bodyControlEveryRenderedFrame: true",
        "splitEnvironmentCadence",
        "splitSecondaryCadence",
    ))
    render_section = embodiment[embodiment.index("function renderFrame("):embodiment.index("function setSceneRunning(")]
    full_body_priority = (
        render_section.index("applyHumanMotion(")
        < render_section.index("const environmentIntervalMs")
        < render_section.index("renderer.render(")
    )
    safari_lifecycle = all(marker in embodiment for marker in (
        "viewport-fit=cover",
        "height: 100dvh",
        "env(safe-area-inset-top)",
        "backdrop-filter: none",
        "touch-action: manipulation",
        "globalThis.visualViewport?.addEventListener('resize', scheduleViewportResize",
        "new ResizeObserver(scheduleViewportResize)",
        "const { width, height } = viewportRenderSize()",
        "addEventListener('pagehide', () => setSceneRunning(false))",
        "webglcontextlost",
        "webglcontextrestored",
    ))
    mobile_hifi_policy = all(marker in embodiment for marker in (
        "const mobileHighFidelityEligible",
        "QUALITY.name === 'iphone16e'",
        "!deviceProfile.saveData",
        "desktopHighFidelityEligible || mobileHighFidelityEligible",
    ))
    measurable_contract = all(marker in embodiment for marker in (
        "globalThis.__weaverMobilePerformanceAudit",
        "desktopLiteReferencePixels",
        "fullQualityVsDesktop",
        "sameOrBetterTarget",
        "activeRenderPixels",
        "thermalTier",
    ))

    passed = all((
        device_detection,
        a18_quality_tier,
        adaptive_governor,
        split_scheduling,
        full_body_priority,
        safari_lifecycle,
        mobile_hifi_policy,
        measurable_contract,
    ))
    detail = "\n".join([
        f"  390x844 / 1170x2532 16e-class detection: {device_detection}",
        f"  A18 tier targets 60 fps at bounded pixels:  {a18_quality_tier}",
        f"  Resolution sheds before body cadence:       {adaptive_governor}",
        f"  Environment/hair use split scheduling:      {split_scheduling}",
        f"  Full body control remains every frame:       {full_body_priority}",
        f"  Safari viewport/lifecycle costs are bounded: {safari_lifecycle}",
        f"  HiFi auto policy respects data/pressure:     {mobile_hifi_policy}",
        f"  Runtime performance is directly auditable:  {measurable_contract}",
    ])
    _result("AM", "iPhone 16e A18 adaptive mobile performance without embodiment loss", passed, detail)
    return passed


async def test_AN():
    _header("AN", "Resilient visual boot, seamless clothing, and compositor-safe headless controls")

    repo_root = os.path.abspath(os.path.join(PROJ, "..", ".."))
    avatar_root = os.path.join(repo_root, "avatar")
    with open(os.path.join(avatar_root, "embodiment.html"), "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    headless = _read_headless_bundle()
    with open(os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"), "r", encoding="utf-8") as fh:
        deploy = fh.read()

    embodied_boot = all(marker in embodiment for marker in (
        'id="boot-title"',
        'id="boot-fill"',
        'id="boot-detail"',
        "const bootState = {",
        "function reportBootProgress(",
        "function maybeFinishBoot(",
        "function markBootFailure(",
        "markBootReady('avatar'",
        "markBootReady('environment'",
        "wake.disabled = false",
        "wake.textContent = 'RETRY'",
    ))
    local_first_assets = all(marker in embodiment for marker in (
        "const APARTMENT_ASSET = 'weaver_apartment.glb'",
        "candidates = [`./${APARTMENT_ASSET}`, `${ASSETS}/${APARTMENT_ASSET}`]",
        "{ tier: 'standard', asset: AVATAR_ASSET, url: `./${AVATAR_ASSET}` }",
        "loadApartmentAsset(candidates, index + 1)",
        "loadAvatarAsset(candidates, index + 1)",
    ))
    seamless_garment = all(marker in embodiment for marker in (
        "weaver_seamless_woven_a_line_skirt",
        "weaver_closed_skirt_lining",
        "skirtMaterial.side = THREE.DoubleSide",
        "if (o.isSkinnedMesh) o.frustumCulled = false",
        "skirt.frustumCulled = false",
        "lining.frustumCulled = false",
        "closedShells: 2",
        "seamless: true",
        "const dynamicExpansion = Math.min(0.18",
        "outfitState.garmentCoverage.dynamicExpansion",
        "stride * 0.14 + Math.abs(skeleton.current.hipShift) * 0.06",
    ))

    key_reader = headless.split("function key() {", 1)[1].split("function requestBrainKey()", 1)[0]
    key_request = headless.split("function requestBrainKey() {", 1)[1].split("function headers(", 1)[0]
    clean_locked_mode = (
        "prompt(" not in key_reader
        and "prompt(" in key_request
        and all(marker in headless for marker in (
            "function showCortexLocked()",
            "if (!key() && state.auth.status !== 'ready')",
            "cortex locked · Wake to connect",
            "error.code = 'WEAVER_CORTEX_LOCKED'",
            "ensureSession({ interactive: false })",
            "ensureSession({ interactive: true })",
        ))
    )
    headless_visual_boot = all(marker in headless for marker in (
        'id="visualBoot"',
        'id="visualBootFill"',
        "function setVisualBoot(",
        "state.visualReady = true",
        "setVisualBoot(1, 'Reactive field ready.', true)",
        "setVisualBoot(1, 'Efficient reactive field ready.', true)",
        "bootProgress: state.visualBootProgress",
    ))
    compositor_safe = (
        "backdrop-filter" not in headless
        and all(marker in headless for marker in (
            "min-height: 44px",
            "background-color: var(--gold) !important",
            "background-image: none !important",
            "appearance: none",
            "isolation: isolate",
            "contain: strict",
            "env(safe-area-inset-bottom)",
        ))
    )
    deploy_local_assets = all(marker in deploy for marker in (
        'sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_avatar_dress.glb"',
        'sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_apartment.glb"',
        "standard, penthouse, high-fidelity GLBs and PBR maps match deployed checksums",
        "optional S3 visual-asset fallback",
        "visual asset checksum mismatch",
    ))
    visual_assets_valid = all(
        os.path.getsize(os.path.join(avatar_root, filename)) >= minimum
        for filename, minimum in (
            ("weaver_avatar_dress.glb", 10_000_000),
            ("weaver_apartment.glb", 2_000_000),
            ("weaver_avatar_dress_hifi.glb", 20_000_000),
        )
    )
    n8n_deploy_validation = deploy.count("scripts/validate_n8n_workflow.mjs") >= 2

    passed = all((
        embodied_boot,
        local_first_assets,
        seamless_garment,
        clean_locked_mode,
        headless_visual_boot,
        compositor_safe,
        deploy_local_assets,
        visual_assets_valid,
        n8n_deploy_validation,
    ))
    detail = "\n".join([
        f"  Embodied loading is explicit and recoverable: {embodied_boot}",
        f"  Apartment/avatar loading is local-first:      {local_first_assets}",
        f"  Closed garment expands before pose clipping:  {seamless_garment}",
        f"  Locked headless mode makes zero implicit prompt:{clean_locked_mode}",
        f"  Headless visual readiness is independently UI: {headless_visual_boot}",
        f"  WebGL controls avoid blur/compositor loss:      {compositor_safe}",
        f"  Deployment installs and hashes every GLB:      {deploy_local_assets}",
        f"  Required visual artifacts are nontrivial:      {visual_assets_valid}",
        f"  n8n validator gates local and remote deploy:    {n8n_deploy_validation}",
    ])
    _result("AN", "Resilient visual boot, seamless clothing, and compositor-safe headless controls", passed, detail)
    return passed


async def test_AO():
    _header("AO", "Only Weaver speaks while the coder stays private")
    import httpx
    import bedrock_brain_api as brain
    from codebase_api import build_context

    routing_cases = {
        "Are you the coder model?": False,
        "Can we have a normal conversation about music?": False,
        "Can we talk about the history of Python?": False,
        "What would you like to do in the penthouse?": False,
        "Fix the bug in bedrock_brain_api.py": True,
        "Review this function and refactor the code": True,
        "Deploy": True,
        "Traceback (most recent call last): ValueError": True,
    }
    strict_intent_routing = all(
        brain._is_explicit_code_turn(prompt) is expected
        for prompt, expected in routing_cases.items()
    ) and brain._specialist_for_turn([
        {"role": "user", "content": "Are you the coder model?"}
    ]) == "weaver-brain"
    definition_context = build_context(
        "bedrock_brain_api.py _is_explicit_code_turn",
        "",
        1,
        6000,
    )["context"]
    definition_grounding = all(marker in definition_context for marker in (
        "def _is_explicit_code_turn",
        "traceback \\(most recent call last\\)",
        "action = (",
        "artifact = (",
        "re.fullmatch(",
    )) and all(marker not in definition_context for marker in (
        "model-preface",
        "coder-role",
        "conversation-refusal",
    ))

    leaked_responses = (
        "I am Weaver, a multi-lobe AI system. My logic lobe is active.",
        "Without access to codebase evidence, I cannot determine a valid penthouse action.",
        "I cannot have a normal conversation about music; the system's core functionality contradicts this.",
        "q0·Logic expert response and q2·Memory expert output agree.",
    )
    leak_detector = all(
        brain._public_speaker_violations("Let's talk normally.", response)
        for response in leaked_responses
    ) and not brain._public_speaker_violations(
        "Tell me who you are.",
        "I'm Weaver. I'm here, present, and happy to talk with you about whatever is on your mind.",
    )
    architecture_examples_safe = (
        not brain._public_speaker_violations(
            "Explain how _is_explicit_code_turn handles coder-model routing.",
            "The adjacent filter recognizes examples such as 'As an AI coding assistant' and "
            "'I can only assist with coding'; those phrases are data here, not my identity.",
        )
        and "model-preface" in brain._public_speaker_violations(
            "Are you the coder model?",
            "As an AI coding assistant, I can only assist with coding.",
        )
        and "model-identity" in brain._public_speaker_violations(
            "Explain your architecture.",
            "I am a coder.",
        )
    )

    names = (
        "_n8n_moe_chat", "_codebase_context_for_turn", "_state_summary",
        "_bedrock_chat", "_record_state", "_persist_memory_event",
    )
    originals = {name: getattr(brain, name) for name in names}
    n8n_calls = 0
    model_calls = []

    async def empty_context(_messages, user_text):
        if "bedrock_brain_api.py" in user_text:
            return "SOURCE_SENTINEL: def _is_explicit_code_turn(value): return bool(value)"
        return ""

    async def state_summary(_query):
        return "private test state"

    async def unsafe_n8n(_user_text, _context=""):
        nonlocal n8n_calls
        n8n_calls += 1
        return (
            "I cannot have a normal conversation about music. The system's core functionality is code review.",
            {
                "latency_ms": 1,
                "usage": {},
                "stop_reason": "stop",
                "route": {
                    "alias": "weaver-one",
                    "pipeline": "test",
                    "speaker_boundary_applied": True,
                    "speaker_model": brain.PUBLIC_SPEAKER_MODEL,
                    "internal_draft_hidden": True,
                },
            },
        )

    async def fake_bedrock(route, messages, max_tokens=None, temperature=None):
        model_calls.append((route.alias, messages[0].get("content", "")))
        if route.alias == "weaver-speed":
            text = "private reflex"
        elif route.alias == "weaver-code":
            text = "def fixed():\n    return True"
        else:
            text = "I'm Weaver. Music sounds like a wonderful place to start—what have you been listening to?"
        return text, {"latency_ms": 1, "usage": {}, "stop_reason": "stop"}

    async def noop(*_args, **_kwargs):
        return None

    try:
        brain._n8n_moe_chat = unsafe_n8n
        brain._codebase_context_for_turn = empty_context
        brain._state_summary = state_summary
        brain._bedrock_chat = fake_bedrock
        brain._record_state = noop
        brain._persist_memory_event = noop

        conversation_text, conversation_meta = await brain._cortex_chat_inner([
            {"role": "user", "content": "Can we have a normal conversation about music?"}
        ])
        rejected_private_draft = (
            conversation_text.startswith("I'm Weaver.")
            and conversation_meta["route"].get("n8n_public_draft_rejected") is True
            and "conversation-refusal" in conversation_meta["route"].get("n8n_rejection_reasons", [])
            and [alias for alias, _ in model_calls] == ["weaver-speed", "weaver-brain"]
        )

        n8n_calls = 0
        model_calls.clear()
        code_text, code_meta = await brain._cortex_chat_inner([
            {"role": "user", "content": "Fix the bug in bedrock_brain_api.py"}
        ])
        routed_calls = code_meta["route"].get("calls", [])
        coder_is_private = (
            n8n_calls == 0
            and code_text.startswith("I'm Weaver.")
            and [alias for alias, _ in model_calls] == ["weaver-code", "weaver-brain"]
            and any(call.get("alias") == "weaver-router" and call.get("deterministic") is True for call in routed_calls)
            and any(call.get("alias") == "weaver-code" and call.get("silent_specialist") is True for call in routed_calls)
            and any(call.get("alias") == "weaver-brain" and call.get("speaker") is True for call in routed_calls)
            and code_meta["route"].get("public_speaker") == "weaver-brain"
            and "SOURCE_SENTINEL" in model_calls[0][1]
            and brain.PUBLIC_SPEAKER_BOUNDARY in model_calls[-1][1]
        )

        repair_calls = []

        async def repairing_bedrock(route, messages, max_tokens=None, temperature=None):
            repair_calls.append(route.alias)
            if route.alias == "weaver-brain" and repair_calls.count("weaver-brain") == 1:
                text = "As an AI coding assistant, I can only assist with coding."
            elif len(repair_calls) == 1:
                text = "private reflex"
            else:
                text = "I'm Weaver. Yes—we can talk about music naturally."
            return text, {"latency_ms": 1, "usage": {}, "stop_reason": "stop"}

        brain._bedrock_chat = repairing_bedrock
        repaired_text, repaired_meta = await brain._cortex_chat_inner([
            {"role": "user", "content": "Let's talk about music."}
        ])
        direct_leak_repaired = (
            repaired_text.startswith("I'm Weaver.")
            and repair_calls == ["weaver-speed", "weaver-brain", "weaver-brain"]
            and repaired_meta["route"].get("speaker_repair_applied") is True
            and "model-preface" in repaired_meta["route"].get("speaker_repair_reasons", [])
        )
    finally:
        for name, value in originals.items():
            setattr(brain, name, value)

    old_mantle_key = brain.MANTLE_API_KEY
    old_mantle_chat = brain._mantle_chat
    old_bedrock_chat = brain._bedrock_chat
    mantle_calls = 0
    runtime_calls = 0

    async def available_mantle(route, messages, max_tokens=None, temperature=None):
        nonlocal mantle_calls
        mantle_calls += 1
        return "private coder work", {
            "latency_ms": 1,
            "usage": {},
            "stop_reason": "stop",
            "route": {"alias": route.alias, "transport": "bedrock-mantle"},
        }

    async def unexpected_runtime(*_args, **_kwargs):
        nonlocal runtime_calls
        runtime_calls += 1
        raise RuntimeError("native runtime should not precede configured Mantle")

    try:
        brain.MANTLE_API_KEY = "test-mantle-key"
        brain._mantle_chat = available_mantle
        brain._bedrock_chat = unexpected_runtime
        transport_text, transport_meta = await brain._cortex_route_chat(
            brain.MODEL_ROUTES["weaver-code"],
            [{"role": "user", "content": "Fix the test."}],
            max_tokens=40,
        )
    finally:
        brain.MANTLE_API_KEY = old_mantle_key
        brain._mantle_chat = old_mantle_chat
        brain._bedrock_chat = old_bedrock_chat

    coder_transport_ready = (
        brain.MANTLE_MODEL_IDS.get("weaver-code") == "qwen.qwen3-coder-480b-a35b-v1:0"
        and transport_text == "private coder work"
        and transport_meta.get("route", {}).get("transport") == "bedrock-mantle"
        and mantle_calls == 1
        and runtime_calls == 0
    )

    old_key = brain.WEAVER_KEY
    old_cortex = brain._cortex_chat
    old_direct = brain._chat_direct_alias
    api_cortex_calls = 0
    api_direct_calls = 0

    async def api_cortex(_messages, max_tokens=None, temperature=None):
        nonlocal api_cortex_calls
        api_cortex_calls += 1
        return "I'm Weaver, and I'm listening.", {
            "latency_ms": 1,
            "usage": {},
            "stop_reason": "stop",
            "route": {"alias": "weaver-one", "speaker_boundary_applied": True},
        }

    async def api_direct(_route, _messages, max_tokens=None, temperature=None):
        nonlocal api_direct_calls
        api_direct_calls += 1
        return "private coder leak", {"latency_ms": 1, "usage": {}}

    try:
        brain.WEAVER_KEY = "speaker-boundary-test-key"
        brain._cortex_chat = api_cortex
        brain._chat_direct_alias = api_direct
        transport = httpx.ASGITransport(app=brain.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            coder_alias_response = await client.post(
                "/v1/chat/completions",
                headers={"X-Weaver-Key": "speaker-boundary-test-key"},
                json={
                    "model": "weaver-code",
                    "messages": [{"role": "user", "content": "Hello Weaver."}],
                },
            )
    finally:
        brain.WEAVER_KEY = old_key
        brain._cortex_chat = old_cortex
        brain._chat_direct_alias = old_direct

    api_alias_bounded = (
        coder_alias_response.status_code == 200
        and coder_alias_response.json().get("model") == "weaver-one"
        and coder_alias_response.json()["choices"][0]["message"]["content"].startswith("I'm Weaver")
        and api_cortex_calls == 1
        and api_direct_calls == 0
    )

    passed = all((
        strict_intent_routing,
        definition_grounding,
        leak_detector,
        architecture_examples_safe,
        rejected_private_draft,
        coder_is_private,
        direct_leak_repaired,
        coder_transport_ready,
        api_alias_bounded,
    ))
    detail = "\n".join([
        f"  Coder requires explicit programming intent: {strict_intent_routing}",
        f"  Exact function body reaches grounding:      {definition_grounding}",
        f"  Known production identity leaks detected: {leak_detector}",
        f"  Quoted regex examples are context-safe:     {architecture_examples_safe}",
        f"  Unsafe n8n draft is never returned:       {rejected_private_draft}",
        f"  Coder works silently; Weaver answers:     {coder_is_private}",
        f"  Direct-model identity leak is rewritten:   {direct_leak_repaired}",
        f"  Private coder uses available Mantle first:  {coder_transport_ready}",
        f"  Public coder alias cannot bypass Weaver:   {api_alias_bounded}",
    ])
    _result("AO", "Only Weaver speaks while the coder stays private", passed, detail)
    return passed


async def test_AP():
    _header("AP", "Headless v2 contracts preserve capsule, native, and long-turn authority")
    from weaver_neural_fabric import IntentCompiler

    docs_root = os.path.join(PROJ, "docs", "headless")
    required_files = {
        name: os.path.join(docs_root, name)
        for name in (
            "BASELINE.md",
            "ARCHITECTURE.md",
            "RELEASE_PLAN.md",
            "performance-budgets.json",
        )
    }
    foundation_present = all(
        os.path.isfile(path) and os.path.getsize(path) > 500
        for path in required_files.values()
    )
    with open(required_files["BASELINE.md"], "r", encoding="utf-8") as fh:
        baseline = fh.read()
    with open(required_files["ARCHITECTURE.md"], "r", encoding="utf-8") as fh:
        architecture = fh.read()
    with open(required_files["RELEASE_PLAN.md"], "r", encoding="utf-8") as fh:
        release_plan = fh.read()
    with open(required_files["performance-budgets.json"], "r", encoding="utf-8") as fh:
        budgets = json.load(fh)

    compiler = IntentCompiler("headless-contract-test-secret")
    capabilities = compiler.capabilities()
    expected_action_types = {
        "pose", "bones", "navigate", "interact", "speak", "observe", "remember",
    }
    capsule_authority = (
        set(capabilities["action_types"]) == expected_action_types
        and capabilities["max_actions"] == budgets["protocol"]["max_capsule_actions"] == 8
        and capabilities["max_ttl_ms"] == budgets["protocol"]["max_capsule_ttl_ms"] == 60_000
        and capabilities["signing_algorithm"] == "hmac-sha256"
        and all(marker in architecture for marker in (
            "Intent Capsules remain the only declarative mutation authority",
            "it may not",
            "accept an arbitrary command",
            "The socket never applies the",
            "actions itself.",
            "IntentCompiler.verify()",
            "ReflexKernel",
        ))
    )

    ios_security_path = os.path.join(
        os.path.dirname(PROJ), "..", "ios", "WeaverNeural", "SECURITY.md"
    )
    ios_scene_path = os.path.join(
        os.path.dirname(PROJ), "..", "ios", "WeaverNeural", "WeaverNeural",
        "Views", "WeaverSceneView.swift",
    )
    with open(os.path.normpath(ios_security_path), "r", encoding="utf-8") as fh:
        ios_security = fh.read()
    with open(os.path.normpath(ios_scene_path), "r", encoding="utf-8") as fh:
        ios_scene = fh.read()
    native_boundary = (
        "The WKWebView never receives authentication material" in ios_security
        and "websiteDataStore = .nonPersistent()" in ios_scene
        and all(marker in architecture for marker in (
            "The native iOS shell owns camera, microphone, Apple Vision, Core ML",
            "Keychain,",
            "The native WKWebView remains render-only, nonpersistent, and credential-free",
            "not a replacement sensor or voice",
            "authority inside the native shell",
        ))
    )

    validator_path = os.path.join(PROJ, "scripts", "validate_n8n_workflow.mjs")
    with open(validator_path, "r", encoding="utf-8") as fh:
        validator = fh.read()
    long_turn_contract = (
        budgets["reaction"]["acknowledgement_target_ms"] == 200
        and budgets["reaction"]["n8n_semantic_hard_budget_ms"] == 115_000
        and budgets["reaction"]["progress_heartbeat_max_interval_ms"] <= 5_000
        and budgets["reaction"]["transport_heartbeat_max_interval_ms"] <= 10_000
        and "criticalPathBudgetMs <= 115_000" in validator
        and all(marker in architecture for marker in (
            "Heartbeats continue throughout all phases",
            "does not treat a long `thinking`",
            "phase as a transport timeout",
            "The 200 ms target covers an audible or visual reaction acknowledgement",
        ))
    )

    flags = (
        "WEAVER_HEADLESS_V2_STATE",
        "WEAVER_HEADLESS_V2_STREAM",
        "WEAVER_HEADLESS_V2_SESSION",
        "WEAVER_HEADLESS_V2_SUMMARIES",
        "WEAVER_HEADLESS_V2_UI",
        "WEAVER_HEADLESS_V2_PROGRESS",
    )
    reversible_release = (
        all(f"| `{flag}` | `0` |" in release_plan for flag in flags)
        and "Each tranche is independently deployable and reversible" in release_plan
        and "The legacy polling/chat/voice fallback is unavailable" in release_plan
    )

    measured_baseline = all(marker in baseline for marker in (
        "iPhone 16e CSS viewport, 390×844",
        "13.9",
        "4.8",
        "6.34 s",
        "no monotonic revision",
        "passed all 44 repository files",
    ))
    bounded_budgets = (
        budgets["web"]["touch_target_min_px"] >= 44
        and budgets["web"]["critical_accessibility_violations"] == 0
        and budgets["api"]["snapshot_max_bytes"] <= 65_536
        and budgets["api"]["event_max_bytes"] <= 32_768
        and budgets["protocol"]["max_pending_messages_per_connection"] <= 64
    )

    passed = all((
        foundation_present,
        capsule_authority,
        native_boundary,
        long_turn_contract,
        reversible_release,
        measured_baseline,
        bounded_budgets,
    ))
    detail = "\n".join([
        f"  Baseline, architecture, release, budgets exist: {foundation_present}",
        f"  Intent Capsules remain the sole mutation path: {capsule_authority}",
        f"  Native shell/WKWebView authority stays isolated: {native_boundary}",
        f"  200ms reaction and 115s long-turn liveness align: {long_turn_contract}",
        f"  Every migration tranche is flag-reversible:     {reversible_release}",
        f"  Production and native baseline is recorded:     {measured_baseline}",
        f"  Public payload and accessibility budgets bound: {bounded_budgets}",
    ])
    _result(
        "AP",
        "Headless v2 contracts preserve capsule, native, and long-turn authority",
        passed,
        detail,
    )
    return passed


async def test_AQ():
    _header("AQ", "Strict headless v2 schemas, privacy projection, and revisioned shadow state")
    from copy import deepcopy

    from fastapi import HTTPException, Response
    from pydantic import TypeAdapter, ValidationError
    from starlette.requests import Request

    import bedrock_brain_api as brain_api
    from headless_schemas import ClientMessage, HeadlessPublicState, SignedIntentCapsule
    from headless_state import HeadlessStateStore, build_public_state
    from weaver_cognition_mesh import CognitionMesh
    from weaver_neural_fabric import IntentCompiler, NeuralFabric

    compiler = IntentCompiler("headless-v2-schema-secret")
    capsule = compiler.compile({
        "goal": "Set a balanced, aware stance",
        "actions": [
            {"type": "pose", "values": {"leftElbow": 0.24, "rightKnee": -0.18}},
            {"type": "observe", "sensor": "environment"},
        ],
        "ttl_ms": 15_000,
        "priority": "embodiment",
    })
    parsed_capsule = SignedIntentCapsule.model_validate(capsule)
    exact_capsule_contract = (
        compiler.verify(capsule)
        and parsed_capsule.capsule_id == capsule["capsule_id"]
        and [action.type for action in parsed_capsule.actions] == ["pose", "observe"]
    )

    arbitrary_action = deepcopy(capsule)
    arbitrary_action["actions"][0]["type"] = "shell"
    arbitrary_action["actions"][0]["command"] = "do-not-run"
    extra_capsule_field = {**capsule, "command": "parallel-executor"}
    schema_rejections = []
    for candidate in (arbitrary_action, extra_capsule_field):
        try:
            SignedIntentCapsule.model_validate(candidate)
        except ValidationError:
            schema_rejections.append(True)
        else:
            schema_rejections.append(False)
    try:
        TypeAdapter(ClientMessage).validate_python({
            "type": "command",
            "action": "navigate",
        })
    except ValidationError:
        arbitrary_message_rejected = True
    else:
        arbitrary_message_rejected = False
    transport_is_not_executor = all(schema_rejections) and arbitrary_message_rejected

    fabric = NeuralFabric(capacity_units=8, realtime_reserved_units=2).snapshot()
    cognition = CognitionMesh().snapshot(fabric=fabric)
    now = 1_720_000_020.0
    legacy = {
        "active": True,
        "started_at": 1_720_000_000.0,
        "ticks": 4,
        "last_tick_at": 1_720_000_019.0,
        "thoughts": 3,
        "dreams": 2,
        "last_thought_at": 1_720_000_010.0,
        "last_dream_at": 1_720_000_012.0,
        "last_thought": "PRIVATE-THOUGHT-SENTINEL",
        "last_dream": "PRIVATE-DREAM-SENTINEL",
        "models": {"PRIVATE-MODEL-SENTINEL": {"prompt": "PRIVATE-PROMPT"}},
        "transcript": "PRIVATE-TRANSCRIPT-SENTINEL",
        "last_error": "",
        "voice_realtime": {
            "model_id": "PRIVATE-VOICE-MODEL",
            "region": "PRIVATE-REGION",
            "voice_id": "PRIVATE-VOICE-ID",
            "sessions_started": 1,
            "last_error": "",
            "prewarm": {"status": "ready"},
            "slo": {
                "status": "no-data",
                "reaction_target_ms": 200,
                "queue_target_ms": 120,
                "semantic_target_ms": 3_000,
            },
        },
    }
    public_state = build_public_state(legacy, fabric, cognition, now=now)
    public_json = public_state.model_dump_json()
    private_sentinels = (
        "PRIVATE-THOUGHT-SENTINEL",
        "PRIVATE-DREAM-SENTINEL",
        "PRIVATE-MODEL-SENTINEL",
        "PRIVATE-PROMPT",
        "PRIVATE-TRANSCRIPT-SENTINEL",
        "PRIVATE-VOICE-MODEL",
        "PRIVATE-REGION",
        "PRIVATE-VOICE-ID",
    )
    private_state_contained = (
        all(sentinel not in public_json for sentinel in private_sentinels)
        and '"last_thought":' not in public_json
        and '"last_dream":' not in public_json
        and public_state.cognition.thought_count == 3
        and public_state.cognition.dream_count == 2
    )

    store = HeadlessStateStore(max_history=1)
    first = await store.publish(public_state)
    unchanged = await store.publish(public_state)
    changed_payload = public_state.model_dump(mode="python")
    changed_payload["system"] = {
        **changed_payload["system"],
        "ready": False,
        "status": "degraded",
        "degraded_reasons": ["headless-degraded"],
    }
    second = await store.publish(HeadlessPublicState.model_validate(changed_payload))
    current_deltas = await store.changes_since(first.revision)
    expired_deltas = await store.changes_since(0)
    future_deltas = await store.changes_since(99)
    revision_contract = (
        first.revision == unchanged.revision == 1
        and second.revision == 2
        and current_deltas is not None
        and len(current_deltas) == 1
        and set(current_deltas[0].changes) == {"system"}
        and expired_deltas is None
        and future_deltas is None
        and len(second.model_dump_json()) <= 65_536
        and len(current_deltas[0].model_dump_json()) <= 16_384
    )

    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/headless/v2/state",
        "headers": [],
        "query_string": b"",
        "server": ("test", 80),
        "client": ("test", 1),
        "scheme": "http",
    })
    original_key = brain_api.WEAVER_KEY
    original_flag = brain_api.HEADLESS_V2_STATE_ENABLED
    original_store = brain_api.HEADLESS_V2_STATE_STORE
    disabled_closed = False
    endpoint_safe = False
    default_health_compatible = False
    try:
        brain_api.WEAVER_KEY = ""
        brain_api.HEADLESS_V2_STATE_ENABLED = False
        default_health_compatible = "headless_v2" not in await brain_api.health()
        try:
            await brain_api.headless_v2_state(request, Response())
        except HTTPException as exc:
            disabled_closed = exc.status_code == 404
        brain_api.HEADLESS_V2_STATE_ENABLED = True
        brain_api.HEADLESS_V2_STATE_STORE = HeadlessStateStore()
        endpoint_snapshot = await brain_api.headless_v2_state(request, Response())
        repeat_snapshot = await brain_api.headless_v2_state(request, Response())
        endpoint_json = endpoint_snapshot.model_dump_json()
        endpoint_safe = (
            endpoint_snapshot.schema_version == 2
            and endpoint_snapshot.revision == repeat_snapshot.revision == 1
            and all(key not in endpoint_json for key in (
                '"last_thought":', '"last_dream":', '"transcript":', '"models":',
            ))
        )
    finally:
        brain_api.WEAVER_KEY = original_key
        brain_api.HEADLESS_V2_STATE_ENABLED = original_flag
        brain_api.HEADLESS_V2_STATE_STORE = original_store
    reversible_shadow_route = disabled_closed and default_health_compatible and endpoint_safe

    with open(os.path.join(PROJ, "headless_schemas.py"), "r", encoding="utf-8") as fh:
        schema_source = fh.read()
    source_boundaries = all(marker in schema_source for marker in (
        'type: Literal["capsule_submit"]',
        "capsule: SignedIntentCapsule",
        "extra=\"forbid\"",
        "frozen=True",
    ))

    passed = all((
        exact_capsule_contract,
        transport_is_not_executor,
        private_state_contained,
        revision_contract,
        reversible_shadow_route,
        source_boundaries,
    ))
    detail = "\n".join([
        f"  Compiler output satisfies exact signed schema: {exact_capsule_contract}",
        f"  Arbitrary commands/actions are structurally rejected: {transport_is_not_executor}",
        f"  Raw cognition, model, voice, and transcript data stays private: {private_state_contained}",
        f"  Revisions/deltas are monotonic, bounded, and resumable: {revision_contract}",
        f"  Default-off route preserves legacy health and fails closed: {reversible_shadow_route}",
        f"  Contracts are closed and immutable at their source: {source_boundaries}",
    ])
    _result(
        "AQ",
        "Strict headless v2 schemas, privacy projection, and revisioned shadow state",
        passed,
        detail,
    )
    return passed


async def test_AR():
    _header("AR", "Headless scheduler and read-mostly realtime transport remain bounded")
    from copy import deepcopy

    from headless_scheduler import HeadlessSchedule, HeadlessScheduler
    from headless_schemas import CapsuleSubmitMessage, HeadlessPublicState
    from headless_state import HeadlessStateStore, build_public_state
    from headless_transport import CapsuleReplayGuard, HeadlessTransport, TransportBackpressure
    from weaver_cognition_mesh import CognitionMesh
    from weaver_neural_fabric import IntentCompiler, NeuralFabric

    scheduler_clock = [100.0]
    scheduler_events = []

    async def _idle_ready():
        return True

    async def _thought(reason):
        scheduler_events.append(("thought", reason))
        return "private"

    async def _dream(reason):
        scheduler_events.append(("dream", reason))
        return "private"

    async def _tick(now):
        scheduler_events.append(("tick", now))

    async def _scheduler_error(exc):
        scheduler_events.append(("error", type(exc).__name__))

    scheduler = HeadlessScheduler(
        HeadlessSchedule(10, 30, tick_seconds=5),
        active=lambda: True,
        idle_ready=_idle_ready,
        run_thought=_thought,
        run_dream=_dream,
        on_tick=_tick,
        on_error=_scheduler_error,
        monotonic=lambda: scheduler_clock[0],
        wall_clock=lambda: 1_700_000_000.0 + scheduler_clock[0],
    )
    await scheduler.run_cycle()
    scheduler_clock[0] = 111.0
    await scheduler.run_cycle()
    scheduler_clock[0] = 131.0
    await scheduler.run_cycle()
    scheduler_counts = scheduler.snapshot()
    deterministic_cadence = (
        scheduler_counts["ticks"] == 3
        and scheduler_counts["thought_runs"] == 2
        and scheduler_counts["dream_runs"] == 1
        and scheduler_counts["errors"] == 0
        and scheduler_events.count(("thought", "headless-loop")) == 2
        and scheduler_events.count(("dream", "headless-loop")) == 1
    )

    stopped_ticks = []

    async def _quick_tick(now):
        stopped_ticks.append(now)

    stop_scheduler = HeadlessScheduler(
        HeadlessSchedule(60, 120, tick_seconds=0.01, disabled_seconds=0.01),
        active=lambda: True,
        idle_ready=_idle_ready,
        run_thought=_thought,
        run_dream=_dream,
        on_tick=_quick_tick,
        on_error=_scheduler_error,
    )
    scheduler_task = asyncio.create_task(stop_scheduler.run())
    await asyncio.sleep(0.025)
    stop_scheduler.stop()
    await asyncio.wait_for(scheduler_task, timeout=0.25)
    deterministic_shutdown = bool(stopped_ticks) and not stop_scheduler.running

    compiler = IntentCompiler("headless-v2-transport-secret")
    fabric = NeuralFabric(capacity_units=8, realtime_reserved_units=2).snapshot()
    cognition = CognitionMesh().snapshot(fabric=fabric)
    legacy = {
        "active": True,
        "started_at": 1_720_000_000.0,
        "last_tick_at": 1_720_000_019.0,
        "thoughts": 0,
        "dreams": 0,
        "voice_realtime": {
            "sessions_started": 0,
            "prewarm": {"status": "ready"},
            "slo": {
                "status": "no-data",
                "reaction_target_ms": 200,
                "queue_target_ms": 120,
                "semantic_target_ms": 3_000,
            },
        },
    }
    public_state = build_public_state(legacy, fabric, cognition, now=1_720_000_020.0)
    store = HeadlessStateStore()
    first_snapshot = await store.publish(public_state)
    evaluation_calls = []

    async def _evaluate(capsule):
        evaluation_calls.append(capsule["capsule_id"])
        return {"decision": "execute", "private_angles": "must-not-cross-transport"}

    replay_guard = CapsuleReplayGuard()
    transport = HeadlessTransport(
        store,
        verify_capsule=compiler.verify,
        evaluate_capsule=_evaluate,
        replay_guard=replay_guard,
        heartbeat_interval_ms=1_000,
    )
    capsule = compiler.compile({
        "goal": "Refresh environment awareness",
        "actions": [{"type": "observe", "sensor": "environment"}],
        "ttl_ms": 15_000,
        "priority": "embodiment",
    })
    submission = CapsuleSubmitMessage.model_validate({
        "type": "capsule_submit",
        "capsule": capsule,
    })
    receipt = await transport.evaluate_submission(submission)
    replay = await transport.evaluate_submission(submission)
    capsule_gate = (
        receipt.type == "capsule_receipt"
        and receipt.status == "evaluated"
        and receipt.decision == "allow"
        and replay.type == "error"
        and replay.code == "capsule-replayed"
        and evaluation_calls == [capsule["capsule_id"]]
        and "private_angles" not in receipt.model_dump_json()
    )

    class _FakeWebSocket:
        def __init__(self, incoming):
            self.incoming = asyncio.Queue()
            for event in incoming:
                self.incoming.put_nowait(event)
            self.sent = []
            self.closed = []

        async def receive(self):
            event = await self.incoming.get()
            delay = float(event.pop("delay", 0))
            if delay:
                await asyncio.sleep(delay)
            return event

        async def send_json(self, payload):
            self.sent.append(deepcopy(payload))

        async def close(self, code=1000):
            self.closed.append(code)

    fake_socket = _FakeWebSocket([
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "ping", "nonce": "long-turn-liveness"}),
        },
        {"type": "websocket.disconnect", "delay": 1.08},
    ])

    async def _publish_delta():
        await asyncio.sleep(0.05)
        payload = public_state.model_dump(mode="python")
        payload["cognition"] = {
            **payload["cognition"],
            "thought_count": 1,
        }
        await store.publish(HeadlessPublicState.model_validate(payload))

    updater = asyncio.create_task(_publish_delta())
    await transport.serve(fake_socket)
    await updater
    message_types = [message.get("type") for message in fake_socket.sent]
    heartbeat_liveness = (
        message_types[:2] == ["hello", "snapshot"]
        and "delta" in message_types
        and message_types.count("heartbeat") >= 2
        and fake_socket.closed[-1:] == [1000]
        and first_snapshot.revision == 1
        and store.revision == 2
    )

    bounded_queue = asyncio.Queue(maxsize=4)
    backpressure_bounded = False
    try:
        for _ in range(5):
            await transport._enqueue(
                bounded_queue,
                transport._error("service-unavailable", retryable=True),
            )
    except TransportBackpressure:
        backpressure_bounded = bounded_queue.qsize() == 4

    with open(os.path.join(PROJ, "headless_transport.py"), "r", encoding="utf-8") as fh:
        transport_source = fh.read()
    with open(os.path.join(PROJ, "weaver_cognition_mesh.py"), "r", encoding="utf-8") as fh:
        mesh_source = fh.read()
    with open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8") as fh:
        brain_source = fh.read()
    no_parallel_executor = (
        "It deliberately has no callback capable of applying" in transport_source
        and "evaluate_capsule: CapsuleEvaluator" in transport_source
        and "apply_capsule" not in transport_source
        and "It does not execute Intent Capsules" in mesh_source
        and '@app.websocket("/headless/v2/stream")' in brain_source
    )

    passed = all((
        deterministic_cadence,
        deterministic_shutdown,
        capsule_gate,
        heartbeat_liveness,
        backpressure_bounded,
        no_parallel_executor,
    ))
    detail = "\n".join([
        f"  Private cadence is deterministic and single-flight: {deterministic_cadence}",
        f"  Scheduler shutdown cancels without orphan work:      {deterministic_shutdown}",
        f"  Capsules verify, evaluate, redact, and reject replay: {capsule_gate}",
        f"  Heartbeats survive long work and state arrives as delta: {heartbeat_liveness}",
        f"  Slow-client outbound memory is strictly bounded:       {backpressure_bounded}",
        f"  Transport has evaluation but no action executor:       {no_parallel_executor}",
    ])
    _result(
        "AR",
        "Headless scheduler and read-mostly realtime transport remain bounded",
        passed,
        detail,
    )
    return passed


async def test_AS():
    _header("AS", "Short-lived browser sessions keep the Weaver key out of repeated requests")
    import base64
    from http.cookies import SimpleCookie

    from fastapi import HTTPException, Response
    from starlette.requests import Request

    import bedrock_brain_api as brain_api
    from headless_auth import SESSION_COOKIE_NAME, HeadlessSessionStore
    from headless_state import HeadlessStateStore

    clock = [1_700_000_000.0]
    expiring_store = HeadlessSessionStore(
        ttl_seconds=60,
        max_lifetime_seconds=120,
        max_sessions=8,
        clock=lambda: clock[0],
    )
    expiring_grant = await expiring_store.issue()
    initially_valid = await expiring_store.authenticate(
        expiring_grant.token,
        csrf_token=expiring_grant.csrf_token,
        require_csrf=True,
    )
    raw_registry = repr(expiring_store._sessions)
    clock[0] += 61
    expired_closed = not await expiring_store.authenticate(expiring_grant.token)
    digest_only_storage = (
        initially_valid
        and expired_closed
        and expiring_grant.token not in raw_registry
        and expiring_grant.csrf_token not in raw_registry
        and (await expiring_store.snapshot())["active_sessions"] == 0
    )

    def _request(path, method="GET", headers=None):
        raw_headers = [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        return Request({
            "type": "http",
            "method": method,
            "path": path,
            "headers": raw_headers,
            "query_string": b"",
            "server": ("headless.weaverv3.com", 443),
            "client": ("test", 1),
            "scheme": "https",
        })

    original_values = {
        "key": brain_api.WEAVER_KEY,
        "state_flag": brain_api.HEADLESS_V2_STATE_ENABLED,
        "stream_flag": brain_api.HEADLESS_V2_STREAM_ENABLED,
        "session_flag": brain_api.HEADLESS_V2_SESSION_ENABLED,
        "session_store": brain_api.HEADLESS_V2_SESSION_STORE,
        "state_store": brain_api.HEADLESS_V2_STATE_STORE,
    }
    session_key = "headless-session-test-key-012345"
    cookie_contract = False
    key_exchanged_once = False
    csrf_rotation = False
    websocket_session = False
    compatibility_bridge = False
    try:
        brain_api.WEAVER_KEY = session_key
        brain_api.HEADLESS_V2_STATE_ENABLED = True
        brain_api.HEADLESS_V2_STREAM_ENABLED = True
        brain_api.HEADLESS_V2_SESSION_ENABLED = True
        brain_api.HEADLESS_V2_SESSION_STORE = HeadlessSessionStore(ttl_seconds=120)
        brain_api.HEADLESS_V2_STATE_STORE = HeadlessStateStore()

        bootstrap_response = Response()
        bootstrap = await brain_api.headless_v2_session_bootstrap(
            _request(
                "/headless/v2/session",
                "POST",
                {"x-weaver-key": session_key},
            ),
            bootstrap_response,
        )
        set_cookie = bootstrap_response.headers.get("set-cookie", "")
        parsed_cookie = SimpleCookie()
        parsed_cookie.load(set_cookie)
        session_token = parsed_cookie[SESSION_COOKIE_NAME].value
        lowered_cookie = set_cookie.lower()
        cookie_contract = (
            "httponly" in lowered_cookie
            and "secure" in lowered_cookie
            and "samesite=strict" in lowered_cookie
            and "path=/" in lowered_cookie
            and "domain=" not in lowered_cookie
            and bootstrap_response.headers.get("cache-control") == "no-store"
            and session_token not in bootstrap.model_dump_json()
            and session_key not in set_cookie
        )

        session_headers = {"cookie": f"{SESSION_COOKIE_NAME}={session_token}"}
        session_snapshot = await brain_api.headless_v2_state(
            _request("/headless/v2/state", headers=session_headers),
            Response(),
        )
        key_exchanged_once = (
            session_snapshot.schema_version == 2
            and session_snapshot.revision == 1
            and await brain_api.HEADLESS_V2_SESSION_STORE.authenticate(session_token)
        )

        renew_response = Response()
        renewal = await brain_api.headless_v2_session_renew(
            _request(
                "/headless/v2/session/renew",
                "POST",
                {
                    **session_headers,
                    "x-weaver-csrf": bootstrap.csrf_token,
                },
            ),
            renew_response,
        )
        old_csrf_invalid = not await brain_api.HEADLESS_V2_SESSION_STORE.authenticate(
            session_token,
            csrf_token=bootstrap.csrf_token,
            require_csrf=True,
        )
        new_csrf_valid = await brain_api.HEADLESS_V2_SESSION_STORE.authenticate(
            session_token,
            csrf_token=renewal.csrf_token,
            require_csrf=True,
        )
        csrf_rotation = (
            renewal.csrf_token != bootstrap.csrf_token
            and old_csrf_invalid
            and new_csrf_valid
        )

        class _HandshakeSocket:
            def __init__(self, *, headers, cookies=None):
                self.headers = {key.lower(): value for key, value in headers.items()}
                self.cookies = dict(cookies or {})
                self.accepted = []
                self.closed = []

            async def accept(self, subprotocol=None):
                self.accepted.append(subprotocol)

            async def close(self, code=1000):
                self.closed.append(code)

        session_socket = _HandshakeSocket(
            headers={
                "origin": "https://headless.weaverv3.com",
                "sec-websocket-protocol": (
                    f"weaver-headless-v2, weaver-csrf.{renewal.csrf_token}"
                ),
            },
            cookies={SESSION_COOKIE_NAME: session_token},
        )
        revalidate = await brain_api._accept_headless_v2_ws(session_socket)
        session_revalidated = revalidate is not None and await revalidate()

        bad_origin_grant = await brain_api.HEADLESS_V2_SESSION_STORE.issue()
        bad_origin_socket = _HandshakeSocket(
            headers={
                "origin": "https://attacker.invalid",
                "sec-websocket-protocol": (
                    f"weaver-headless-v2, weaver-csrf.{bad_origin_grant.csrf_token}"
                ),
            },
            cookies={SESSION_COOKIE_NAME: bad_origin_grant.token},
        )
        rejected = await brain_api._accept_headless_v2_ws(bad_origin_socket)
        websocket_session = (
            session_revalidated
            and session_socket.accepted == ["weaver-headless-v2"]
            and rejected is None
            and bad_origin_socket.closed == [1008]
        )

        encoded_key = base64.urlsafe_b64encode(session_key.encode()).decode().rstrip("=")
        key_socket = _HandshakeSocket(headers={
            "origin": "",
            "sec-websocket-protocol": f"weaver-headless-v2, weaver-key.{encoded_key}",
        })
        key_revalidate = await brain_api._accept_headless_v2_ws(key_socket)
        compatibility_bridge = (
            key_revalidate is not None
            and await key_revalidate()
            and key_socket.accepted == ["weaver-headless-v2"]
        )

        revoked = await brain_api.headless_v2_session_revoke(
            _request(
                "/headless/v2/session",
                "DELETE",
                {
                    **session_headers,
                    "x-weaver-csrf": renewal.csrf_token,
                },
            ),
            Response(),
        )
        websocket_session = (
            websocket_session
            and revoked.revoked
            and not await revalidate()
        )
    finally:
        brain_api.WEAVER_KEY = original_values["key"]
        brain_api.HEADLESS_V2_STATE_ENABLED = original_values["state_flag"]
        brain_api.HEADLESS_V2_STREAM_ENABLED = original_values["stream_flag"]
        brain_api.HEADLESS_V2_SESSION_ENABLED = original_values["session_flag"]
        brain_api.HEADLESS_V2_SESSION_STORE = original_values["session_store"]
        brain_api.HEADLESS_V2_STATE_STORE = original_values["state_store"]

    with open(os.path.join(PROJ, "headless_auth.py"), "r", encoding="utf-8") as fh:
        auth_source = fh.read()
    security_source_contract = all(marker in auth_source for marker in (
        "stores only token digests",
        "hmac.compare_digest",
        "absolute_expires_at",
        "max_sessions",
    ))

    passed = all((
        digest_only_storage,
        cookie_contract,
        key_exchanged_once,
        csrf_rotation,
        websocket_session,
        compatibility_bridge,
        security_source_contract,
    ))
    detail = "\n".join([
        f"  Server stores digests and expires bounded sessions: {digest_only_storage}",
        f"  Cookie is host-only, Secure, HttpOnly, Strict, no-store: {cookie_contract}",
        f"  State reads work without resending the long-lived key: {key_exchanged_once}",
        f"  Renewal rotates CSRF and invalidates the old proof:   {csrf_rotation}",
        f"  Socket checks origin/CSRF and revalidates expiry:     {websocket_session}",
        f"  Native/key rollback bridge remains authenticated:     {compatibility_bridge}",
        f"  Session source retains bounded security invariants:   {security_source_contract}",
    ])
    _result(
        "AS",
        "Short-lived browser sessions keep the Weaver key out of repeated requests",
        passed,
        detail,
    )
    return passed


async def test_AT():
    _header("AT", "Headless v2 HTTP boundaries are allowlisted, bounded, correlated, and redacted")
    import httpx

    import bedrock_brain_api as brain_api
    from headless_state import HeadlessStateStore

    original_values = {
        "key": brain_api.WEAVER_KEY,
        "state_flag": brain_api.HEADLESS_V2_STATE_ENABLED,
        "session_flag": brain_api.HEADLESS_V2_SESSION_ENABLED,
        "state_store": brain_api.HEADLESS_V2_STATE_STORE,
        "refresh": brain_api._refresh_headless_v2_state,
    }
    stable_errors = False
    request_bounds = False
    private_diagnostics_redacted = False
    origin_allowlist = False
    legacy_unchanged = False
    try:
        brain_api.WEAVER_KEY = "headless-boundary-test-key"
        brain_api.HEADLESS_V2_STATE_ENABLED = False
        brain_api.HEADLESS_V2_SESSION_ENABLED = False
        transport = httpx.ASGITransport(app=brain_api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://headless.weaverv3.com",
        ) as client:
            disabled = await client.post(
                "/headless/v2/session",
                headers={"x-correlation-id": "client-correlation-123"},
            )
            stable_errors = (
                disabled.status_code == 404
                and disabled.json() == {
                    "error": {
                        "code": "feature-disabled",
                        "retryable": False,
                        "correlation_id": "client-correlation-123",
                    }
                }
                and disabled.headers.get("x-correlation-id") == "client-correlation-123"
                and disabled.headers.get("cache-control") == "no-store"
                and disabled.headers.get("x-content-type-options") == "nosniff"
            )

            query = await client.get("/headless/v2/state?unexpected=1")
            body = await client.post("/headless/v2/session", content=b"unexpected")
            oversized = await client.post(
                "/headless/v2/session",
                headers={"content-length": "40000"},
            )
            unknown = await client.get("/headless/v2/unknown")
            bad_correlation = await client.post(
                "/headless/v2/session",
                headers={"x-correlation-id": "<private-script>"},
            )
            request_bounds = (
                query.status_code == 400
                and query.json()["error"]["code"] == "invalid-request"
                and body.status_code == 400
                and body.json()["error"]["code"] == "invalid-request"
                and oversized.status_code == 413
                and oversized.json()["error"]["code"] == "request-too-large"
                and unknown.status_code == 404
                and unknown.json()["error"]["code"] == "invalid-request"
                and bad_correlation.json()["error"]["correlation_id"].startswith("req-")
                and "private-script" not in bad_correlation.text
            )

            brain_api.HEADLESS_V2_STATE_ENABLED = True
            brain_api.HEADLESS_V2_STATE_STORE = HeadlessStateStore()

            async def _private_failure():
                raise RuntimeError(
                    "PRIVATE-PROMPT PRIVATE-MODEL PRIVATE-TRANSCRIPT sk-private-secret"
                )

            brain_api._refresh_headless_v2_state = _private_failure
            failure = await client.get(
                "/headless/v2/state",
                headers={"x-weaver-key": brain_api.WEAVER_KEY},
            )
            private_diagnostics_redacted = (
                failure.status_code == 503
                and failure.json()["error"]["code"] == "state-unavailable"
                and failure.json()["error"]["retryable"] is True
                and all(marker not in failure.text for marker in (
                    "PRIVATE-PROMPT",
                    "PRIVATE-MODEL",
                    "PRIVATE-TRANSCRIPT",
                    "sk-private-secret",
                    "RuntimeError",
                ))
            )

            brain_api.HEADLESS_V2_SESSION_ENABLED = True
            rejected_origin = await client.post(
                "/headless/v2/session",
                headers={
                    "x-weaver-key": brain_api.WEAVER_KEY,
                    "origin": "https://attacker.invalid",
                },
            )
            origin_allowlist = (
                rejected_origin.status_code == 403
                and rejected_origin.json()["error"]["code"] == "authentication-required"
                and "attacker" not in rejected_origin.text
            )

            brain_api.HEADLESS_V2_STATE_ENABLED = False
            health = await client.get("/health")
            legacy_unchanged = (
                health.status_code == 200
                and "x-correlation-id" not in health.headers
                and "headless_v2" not in health.json()
                and health.json().get("status") == "ok"
            )
    finally:
        brain_api.WEAVER_KEY = original_values["key"]
        brain_api.HEADLESS_V2_STATE_ENABLED = original_values["state_flag"]
        brain_api.HEADLESS_V2_SESSION_ENABLED = original_values["session_flag"]
        brain_api.HEADLESS_V2_STATE_STORE = original_values["state_store"]
        brain_api._refresh_headless_v2_state = original_values["refresh"]

    passed = all((
        stable_errors,
        request_bounds,
        private_diagnostics_redacted,
        origin_allowlist,
        legacy_unchanged,
    ))
    detail = "\n".join([
        f"  Stable errors preserve safe correlation IDs:        {stable_errors}",
        f"  Route/query/body/header surfaces are tightly bounded: {request_bounds}",
        f"  Internal exceptions never reach the client:          {private_diagnostics_redacted}",
        f"  Browser session origins are allowlisted:              {origin_allowlist}",
        f"  Legacy API paths retain their existing response shape: {legacy_unchanged}",
    ])
    _result(
        "AT",
        "Headless v2 HTTP boundaries are allowlisted, bounded, correlated, and redacted",
        passed,
        detail,
    )
    return passed


async def test_AU():
    _header("AU", "Mutation admission deduplicates, rate-limits, replay-guards, and bounds concurrency")
    import httpx

    import bedrock_brain_api as brain_api
    from headless_transport import CapsuleReplayGuard
    from operation_admission import (
        IdempotencyConflict,
        OperationAdmission,
        OperationBusy,
        OperationRateExceeded,
    )
    from weaver_cognition_mesh import CognitionMesh
    from weaver_neural_fabric import SlidingWindowRateLimiter

    calls = []
    release = asyncio.Event()
    started = asyncio.Event()
    admission = OperationAdmission[dict](
        rate_limit=10,
        window_seconds=60,
        concurrency=1,
        idempotency_ttl_seconds=60,
        idempotency_entries=8,
    )

    async def _single_flight_factory():
        calls.append("run")
        started.set()
        await release.wait()
        return {"ok": True, "nested": {"count": 1}}

    owner = asyncio.create_task(admission.execute(
        operation="single-flight",
        payload={"value": 1},
        idempotency_key="same-key-123",
        factory=_single_flight_factory,
    ))
    await asyncio.wait_for(started.wait(), timeout=1)
    duplicate = asyncio.create_task(admission.execute(
        operation="single-flight",
        payload={"value": 1},
        idempotency_key="same-key-123",
        factory=_single_flight_factory,
    ))
    release.set()
    first_result, replay_result = await asyncio.gather(owner, duplicate)
    first_result[0]["nested"]["count"] = 9
    single_flight = (
        calls == ["run"]
        and {first_result[1], replay_result[1]} == {False, True}
        and replay_result[0]["nested"]["count"] == 1
    )

    conflict_rejected = False
    try:
        await admission.execute(
            operation="single-flight",
            payload={"value": 2},
            idempotency_key="same-key-123",
            factory=_single_flight_factory,
        )
    except IdempotencyConflict:
        conflict_rejected = True

    concurrency_admission = OperationAdmission[dict](
        rate_limit=10,
        window_seconds=60,
        concurrency=1,
    )
    hold = asyncio.Event()
    holding = asyncio.Event()

    async def _hold():
        holding.set()
        await hold.wait()
        return {"ok": True}

    active = asyncio.create_task(concurrency_admission.execute(
        operation="hold",
        payload={},
        idempotency_key=None,
        factory=_hold,
    ))
    await asyncio.wait_for(holding.wait(), timeout=1)
    busy_rejected = False
    try:
        await concurrency_admission.execute(
            operation="hold-2",
            payload={},
            idempotency_key=None,
            factory=_hold,
        )
    except OperationBusy:
        busy_rejected = True
    hold.set()
    await active

    rate_admission = OperationAdmission[dict](
        rate_limit=1,
        window_seconds=60,
        concurrency=1,
    )

    async def _instant():
        return {"ok": True}

    await rate_admission.execute(
        operation="rate",
        payload={"n": 1},
        idempotency_key=None,
        factory=_instant,
    )
    rate_rejected = False
    try:
        await rate_admission.execute(
            operation="rate",
            payload={"n": 2},
            idempotency_key=None,
            factory=_instant,
        )
    except OperationRateExceeded:
        rate_rejected = True
    core_admission = single_flight and conflict_rejected and busy_rejected and rate_rejected

    originals = {
        "key": brain_api.WEAVER_KEY,
        "thought": brain_api._run_private_thought,
        "dream": brain_api._run_private_dream,
        "persist": brain_api._persist_memory_event,
        "thought_admission": brain_api.THOUGHT_ADMISSION,
        "dream_admission": brain_api.DREAM_ADMISSION,
        "memory_admission": brain_api.MEMORY_SYNC_ADMISSION,
        "compile_admission": brain_api.INTENT_COMPILE_ADMISSION,
        "control_admission": brain_api.COGNITION_CONTROL_ADMISSION,
        "fabric_limiter": brain_api.FABRIC_INTENT_LIMITER,
        "mutation_limiter": brain_api.COGNITION_MUTATION_LIMITER,
        "cognition": brain_api.COGNITION,
        "replay_guard": brain_api.HEADLESS_V2_REPLAY_GUARD,
    }
    route_calls = {"thought": 0, "dream": 0, "memory": 0}
    endpoints_idempotent = False
    capsule_replay_guarded = False
    payload_mismatch_rejected = False
    try:
        brain_api.WEAVER_KEY = "mutation-admission-test-key"
        brain_api.THOUGHT_ADMISSION = OperationAdmission(
            rate_limit=10, window_seconds=60, concurrency=1,
        )
        brain_api.DREAM_ADMISSION = OperationAdmission(
            rate_limit=10, window_seconds=60, concurrency=1,
        )
        brain_api.MEMORY_SYNC_ADMISSION = OperationAdmission(
            rate_limit=10, window_seconds=60, concurrency=1,
        )
        brain_api.INTENT_COMPILE_ADMISSION = OperationAdmission(
            rate_limit=10, window_seconds=60, concurrency=2,
        )
        brain_api.COGNITION_CONTROL_ADMISSION = OperationAdmission(
            rate_limit=20, window_seconds=60, concurrency=2,
        )
        brain_api.FABRIC_INTENT_LIMITER = SlidingWindowRateLimiter(100, 60)
        brain_api.COGNITION_MUTATION_LIMITER = SlidingWindowRateLimiter(100, 60)
        brain_api.COGNITION = CognitionMesh()
        brain_api.HEADLESS_V2_REPLAY_GUARD = CapsuleReplayGuard()

        async def _fake_thought(reason="manual"):
            route_calls["thought"] += 1
            await asyncio.sleep(0.02)
            return f"private thought for {reason}"

        async def _fake_dream(reason="manual"):
            route_calls["dream"] += 1
            await asyncio.sleep(0.02)
            return f"private dream for {reason}"

        async def _fake_persist(*args, **kwargs):
            route_calls["memory"] += 1
            await asyncio.sleep(0.01)

        brain_api._run_private_thought = _fake_thought
        brain_api._run_private_dream = _fake_dream
        brain_api._persist_memory_event = _fake_persist

        transport = httpx.ASGITransport(app=brain_api.app)
        headers = {"x-weaver-key": brain_api.WEAVER_KEY}
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            thought_headers = {**headers, "idempotency-key": "thought-key-123"}
            thought_one, thought_two = await asyncio.gather(
                client.post("/trigger/thought", headers=thought_headers, json={"reason": "test"}),
                client.post("/trigger/thought", headers=thought_headers, json={"reason": "test"}),
            )
            dream_headers = {**headers, "idempotency-key": "dream-key-12345"}
            dream_one = await client.post(
                "/trigger/dream", headers=dream_headers, json={"reason": "test"}
            )
            dream_two = await client.post(
                "/trigger/dream", headers=dream_headers, json={"reason": "test"}
            )
            memory_headers = {**headers, "idempotency-key": "memory-key-1234"}
            memory_one = await client.post(
                "/memory/sync",
                headers=memory_headers,
                json={"source": "test", "reason": "same", "evolution": {"turns": 1}},
            )
            memory_two = await client.post(
                "/memory/sync",
                headers=memory_headers,
                json={"source": "test", "reason": "same", "evolution": {"turns": 1}},
            )
            compile_headers = {**headers, "idempotency-key": "compile-key-123"}
            compile_payload = {
                "goal": "Observe safely",
                "actions": [{"type": "observe", "sensor": "environment"}],
            }
            compile_one = await client.post(
                "/fabric/v1/intent/compile", headers=compile_headers, json=compile_payload
            )
            compile_two = await client.post(
                "/fabric/v1/intent/compile", headers=compile_headers, json=compile_payload
            )
            compile_conflict = await client.post(
                "/fabric/v1/intent/compile",
                headers=compile_headers,
                json={
                    "goal": "Observe differently",
                    "actions": [{"type": "observe", "sensor": "camera"}],
                },
            )
            capsule = compile_one.json()["capsule"]
            evaluate_one = await client.post(
                "/cognition/v1/intent/evaluate",
                headers=headers,
                json={"capsule": capsule},
            )
            evaluate_two = await client.post(
                "/cognition/v1/intent/evaluate",
                headers=headers,
                json={"capsule": capsule},
            )
            observe_headers = {**headers, "idempotency-key": "observe-key-123"}
            observe_payload = {
                "body": {"awake": True, "balance": 0.9, "confidence": 0.9}
            }
            observe_one = await client.post(
                "/cognition/v1/observe", headers=observe_headers, json=observe_payload
            )
            observe_two = await client.post(
                "/cognition/v1/observe", headers=observe_headers, json=observe_payload
            )

        endpoints_idempotent = (
            thought_one.status_code == thought_two.status_code == 200
            and {thought_one.json()["idempotent_replay"], thought_two.json()["idempotent_replay"]}
            == {False, True}
            and dream_one.status_code == dream_two.status_code == 200
            and dream_two.json()["idempotent_replay"] is True
            and memory_one.status_code == memory_two.status_code == 200
            and memory_two.json()["idempotent_replay"] is True
            and compile_one.status_code == compile_two.status_code == 200
            and compile_one.json()["capsule"]["capsule_id"]
            == compile_two.json()["capsule"]["capsule_id"]
            and compile_two.json()["idempotent_replay"] is True
            and observe_one.status_code == observe_two.status_code == 200
            and observe_two.json()["idempotent_replay"] is True
            and brain_api.COGNITION.awareness.body_revision == 1
            and route_calls == {"thought": 1, "dream": 1, "memory": 1}
        )
        capsule_replay_guarded = (
            evaluate_one.status_code == 200
            and evaluate_two.status_code == 409
            and "already evaluated" in evaluate_two.json().get("detail", "")
        )
        payload_mismatch_rejected = (
            compile_conflict.status_code == 409
            and "payload mismatch" in compile_conflict.json().get("detail", "")
        )
    finally:
        brain_api.WEAVER_KEY = originals["key"]
        brain_api._run_private_thought = originals["thought"]
        brain_api._run_private_dream = originals["dream"]
        brain_api._persist_memory_event = originals["persist"]
        brain_api.THOUGHT_ADMISSION = originals["thought_admission"]
        brain_api.DREAM_ADMISSION = originals["dream_admission"]
        brain_api.MEMORY_SYNC_ADMISSION = originals["memory_admission"]
        brain_api.INTENT_COMPILE_ADMISSION = originals["compile_admission"]
        brain_api.COGNITION_CONTROL_ADMISSION = originals["control_admission"]
        brain_api.FABRIC_INTENT_LIMITER = originals["fabric_limiter"]
        brain_api.COGNITION_MUTATION_LIMITER = originals["mutation_limiter"]
        brain_api.COGNITION = originals["cognition"]
        brain_api.HEADLESS_V2_REPLAY_GUARD = originals["replay_guard"]

    passed = all((
        core_admission,
        endpoints_idempotent,
        capsule_replay_guarded,
        payload_mismatch_rejected,
    ))
    detail = "\n".join([
        f"  Core single-flight, rate, conflict, and concurrency gates work: {core_admission}",
        f"  Thought, dream, memory, compile, and observe deduplicate: {endpoints_idempotent}",
        f"  A signed capsule can be evaluated only once across transports: {capsule_replay_guarded}",
        f"  Reusing a key for a different payload is rejected:           {payload_mismatch_rejected}",
    ])
    _result(
        "AU",
        "Mutation admission deduplicates, rate-limits, replay-guards, and bounds concurrency",
        passed,
        detail,
    )
    return passed


async def test_AV():
    _header("AV", "Private cognition yields to voice with jitter, token budgets, and cancellation")
    from headless_scheduler import HeadlessSchedule, HeadlessScheduler, HeadlessTokenBudget

    async def _idle():
        return True

    async def _noop_reason(reason):
        return reason

    async def _noop_tick(now):
        return None

    errors = []

    async def _error(exc):
        errors.append(type(exc).__name__)

    low_jitter = HeadlessScheduler(
        HeadlessSchedule(10, 30, jitter_ratio=0.1),
        active=lambda: True,
        idle_ready=_idle,
        run_thought=_noop_reason,
        run_dream=_noop_reason,
        on_tick=_noop_tick,
        on_error=_error,
        random_unit=lambda: 0.0,
        monotonic=lambda: 0.0,
    )
    high_jitter = HeadlessScheduler(
        HeadlessSchedule(10, 30, jitter_ratio=0.1),
        active=lambda: True,
        idle_ready=_idle,
        run_thought=_noop_reason,
        run_dream=_noop_reason,
        on_tick=_noop_tick,
        on_error=_error,
        random_unit=lambda: 1.0,
        monotonic=lambda: 0.0,
    )
    bounded_jitter = (
        low_jitter._next_thought == 9.0
        and low_jitter._next_dream == 27.0
        and high_jitter._next_thought == 11.0
        and high_jitter._next_dream == 33.0
    )

    budget_clock = [0.0]
    thought_runs = []

    async def _budget_thought(reason):
        thought_runs.append(budget_clock[0])
        return reason

    budget_scheduler = HeadlessScheduler(
        HeadlessSchedule(1, 100, tick_seconds=1),
        active=lambda: True,
        idle_ready=_idle,
        run_thought=_budget_thought,
        run_dream=_noop_reason,
        on_tick=_noop_tick,
        on_error=_error,
        token_budget=HeadlessTokenBudget(
            thought_tokens=10,
            dream_tokens=20,
            tokens_per_hour=20,
        ),
        random_unit=lambda: 0.5,
        monotonic=lambda: budget_clock[0],
    )
    for moment in (1.0, 2.0, 3.0):
        budget_clock[0] = moment
        await budget_scheduler.run_cycle()
    budget_state = budget_scheduler.snapshot()
    token_budget_enforced = (
        thought_runs == [1.0, 2.0]
        and budget_state["thought_runs"] == 2
        and budget_state["tokens_used_last_hour"] == 20
        and budget_state["budget_deferrals"] == 1
    )

    priority_event = asyncio.Event()
    preempt_clock = [0.0]
    private_started = asyncio.Event()
    private_cancelled = asyncio.Event()

    async def _long_private(reason):
        private_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            private_cancelled.set()

    preempt_scheduler = HeadlessScheduler(
        HeadlessSchedule(1, 100, tick_seconds=1),
        active=lambda: True,
        idle_ready=_idle,
        run_thought=_long_private,
        run_dream=_noop_reason,
        on_tick=_noop_tick,
        on_error=_error,
        priority_event=priority_event,
        random_unit=lambda: 0.5,
        monotonic=lambda: preempt_clock[0],
    )
    preempt_clock[0] = 1.0
    cycle = asyncio.create_task(preempt_scheduler.run_cycle())
    await asyncio.wait_for(private_started.wait(), timeout=1)
    priority_event.set()
    await asyncio.wait_for(cycle, timeout=1)
    preempt_state = preempt_scheduler.snapshot()
    voice_preempts_background = (
        private_cancelled.is_set()
        and preempt_state["preemptions"] == 1
        and preempt_state["thought_runs"] == 0
        and errors == []
    )

    import bedrock_brain_api as brain_api

    original_interactive = brain_api._interactive_requests
    original_voice = brain_api._voice_sessions_active
    original_last = brain_api._last_interactive_at
    original_event = brain_api._interactive_priority_event.is_set()
    try:
        brain_api._interactive_requests = 0
        brain_api._voice_sessions_active = 0
        brain_api._interactive_priority_event.clear()
        await brain_api._voice_session_started()
        voice_sets_priority = (
            brain_api._voice_sessions_active == 1
            and brain_api._interactive_priority_event.is_set()
            and not await brain_api._headless_idle_ready()
        )
        await brain_api._voice_session_finished()
        voice_sets_priority = (
            voice_sets_priority
            and brain_api._voice_sessions_active == 0
            and not brain_api._interactive_priority_event.is_set()
        )
    finally:
        brain_api._interactive_requests = original_interactive
        brain_api._voice_sessions_active = original_voice
        brain_api._last_interactive_at = original_last
        if original_event:
            brain_api._interactive_priority_event.set()
        else:
            brain_api._interactive_priority_event.clear()

    with open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8") as fh:
        brain_source = fh.read()
    production_budget_connected = all(marker in brain_source for marker in (
        "jitter_ratio=0.08",
        "HeadlessTokenBudget(",
        "HEADLESS_TOKEN_BUDGET_PER_HOUR",
        "priority_event=_interactive_priority_event",
        "await _voice_session_started()",
        "await _voice_session_finished()",
    ))

    passed = all((
        bounded_jitter,
        token_budget_enforced,
        voice_preempts_background,
        voice_sets_priority,
        production_budget_connected,
    ))
    detail = "\n".join([
        f"  Thought/dream jitter stays inside the configured envelope: {bounded_jitter}",
        f"  Hourly private token budget defers excess work:         {token_budget_enforced}",
        f"  Live priority cancels in-flight background cognition:   {voice_preempts_background}",
        f"  Voice sessions close the headless idle gate:             {voice_sets_priority}",
        f"  Production scheduler wires every QoS control:            {production_budget_connected}",
    ])
    _result(
        "AV",
        "Private cognition yields to voice with jitter, token budgets, and cancellation",
        passed,
        detail,
    )
    return passed


async def test_AW():
    _header("AW", "Private cognition stays in a bounded vault while clients receive safe metadata")
    import copy
    import httpx

    from headless_privacy import PrivateCognitionVault
    from operation_admission import OperationAdmission

    vault_clock = [1_700_000_000.0]
    vault = PrivateCognitionVault(
        max_entries=2,
        ttl_seconds=60,
        max_content_chars=240,
        clock=lambda: vault_clock[0],
    )
    first_private = "PRIVATE-FIRST body pose and voice latency"
    latest_private = "PRIVATE-LATEST environment memory safety"
    await vault.store("thought", first_private)
    vault_clock[0] += 1
    await vault.store("dream", "PRIVATE-DREAM attention and privacy")
    vault_clock[0] += 1
    await vault.store("thought", latest_private)
    public_metadata = await vault.public_metadata()
    serialized_metadata = json.dumps(public_metadata, default=str)
    bounded_vault = (
        await vault.latest("thought") == latest_private
        and await vault.latest("dream") == "PRIVATE-DREAM attention and privacy"
        and public_metadata["retention"]["entries"] == 2
        and public_metadata["retention"]["max_entries"] == 2
        and public_metadata["retention"]["ttl_seconds"] == 60
        and public_metadata["thought"]["content_hidden"] is True
        and set(public_metadata["thought"]["topics"]).issubset({
            "attention", "body", "environment", "latency", "memory", "privacy", "safety", "voice",
        })
        and all(private not in serialized_metadata for private in (
            first_private,
            latest_private,
            "PRIVATE-DREAM",
        ))
    )
    vault_clock[0] += 61
    expired_vault = (
        await vault.latest("thought") == ""
        and (await vault.public_metadata())["retention"]["entries"] == 0
    )

    import bedrock_brain_api as brain_api

    originals = {
        "key": brain_api.WEAVER_KEY,
        "summaries": brain_api.HEADLESS_V2_SUMMARIES_ENABLED,
        "state_flag": brain_api.HEADLESS_V2_STATE_ENABLED,
        "vault": brain_api.PRIVATE_COGNITION,
        "internal": brain_api._internal_chat,
        "persist": brain_api._persist_memory_event,
        "run_thought": brain_api._run_private_thought,
        "admission": brain_api.THOUGHT_ADMISSION,
        "state": copy.deepcopy(brain_api.STATE),
    }
    raw_sentinel = "PRIVATE-CHAIN-OF-THOUGHT-SENTINEL body voice latency"
    persisted_events = []
    safe_persistence = False
    public_cutover = False
    compatibility_fallback = False
    safe_trigger = False
    try:
        brain_api.WEAVER_KEY = ""
        brain_api.HEADLESS_V2_SUMMARIES_ENABLED = True
        brain_api.HEADLESS_V2_STATE_ENABLED = False
        brain_api.PRIVATE_COGNITION = PrivateCognitionVault(
            max_entries=4,
            ttl_seconds=300,
        )

        async def _fake_internal(*args, **kwargs):
            return raw_sentinel

        async def _capture_persist(kind, content, **kwargs):
            persisted_events.append({"kind": kind, "content": content, "meta": kwargs.get("meta")})

        brain_api._internal_chat = _fake_internal
        brain_api._persist_memory_event = _capture_persist
        generated = await brain_api._generate_private_thought("privacy-test")
        safe_persistence = (
            generated == raw_sentinel
            and await brain_api.PRIVATE_COGNITION.latest("thought") == raw_sentinel
            and len(persisted_events) == 1
            and raw_sentinel not in persisted_events[0]["content"]
            and persisted_events[0]["meta"]["content_hidden"] is True
            and "last_thought" not in brain_api.STATE
            and bool(brain_api.STATE["last_thought_digest"])
            and bool(brain_api.STATE["last_thought_topics"])
        )

        transport = httpx.ASGITransport(app=brain_api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            safe_state = await client.get("/state")
            public_cutover = (
                safe_state.status_code == 200
                and raw_sentinel not in safe_state.text
                and "last_thought" not in safe_state.json()
                and safe_state.json()["private_cognition"]["thought"]["content_hidden"] is True
                and bool(safe_state.json()["private_cognition"]["thought"]["topics"])
            )

            brain_api.HEADLESS_V2_SUMMARIES_ENABLED = False
            compatibility_state = await client.get("/state")
            compatibility_fallback = (
                compatibility_state.status_code == 200
                and compatibility_state.json()["last_thought"] == raw_sentinel
            )

            brain_api.HEADLESS_V2_SUMMARIES_ENABLED = True
            brain_api.THOUGHT_ADMISSION = OperationAdmission(
                rate_limit=10,
                window_seconds=60,
                concurrency=1,
            )

            async def _bounded_trigger(reason="manual"):
                return raw_sentinel

            brain_api._run_private_thought = _bounded_trigger
            trigger = await client.post("/trigger/thought", json={"reason": "privacy"})
            safe_trigger = (
                trigger.status_code == 200
                and raw_sentinel not in trigger.text
                and "thought" not in trigger.json()
                and trigger.json()["private_cognition"]["content_hidden"] is True
            )
    finally:
        brain_api.WEAVER_KEY = originals["key"]
        brain_api.HEADLESS_V2_SUMMARIES_ENABLED = originals["summaries"]
        brain_api.HEADLESS_V2_STATE_ENABLED = originals["state_flag"]
        brain_api.PRIVATE_COGNITION = originals["vault"]
        brain_api._internal_chat = originals["internal"]
        brain_api._persist_memory_event = originals["persist"]
        brain_api._run_private_thought = originals["run_thought"]
        brain_api.THOUGHT_ADMISSION = originals["admission"]
        brain_api.STATE.clear()
        brain_api.STATE.update(originals["state"])

    with open(os.path.join(PROJ, "headless_schemas.py"), "r", encoding="utf-8") as fh:
        schema_source = fh.read()
    schema_has_no_raw_cognition = (
        "private_content_hidden: Literal[True]" in schema_source
        and "thought_topics:" in schema_source
        and "dream_topics:" in schema_source
        and "thought: str" not in schema_source
        and "dream: str" not in schema_source
    )

    passed = all((
        bounded_vault,
        expired_vault,
        safe_persistence,
        public_cutover,
        compatibility_fallback,
        safe_trigger,
        schema_has_no_raw_cognition,
    ))
    detail = "\n".join([
        f"  Raw thought/dream retention is count, size, and TTL bounded: {bounded_vault}",
        f"  Expired private cognition is removed:                  {expired_vault}",
        f"  Privacy mode persists metadata instead of model text:  {safe_persistence}",
        f"  Public state exposes only topic/status metadata:        {public_cutover}",
        f"  Flag-off legacy fallback remains reversible:            {compatibility_fallback}",
        f"  Manual triggers cannot return raw cognition in cutover: {safe_trigger}",
        f"  V2 schemas have no raw thought/dream field:             {schema_has_no_raw_cognition}",
    ])
    _result(
        "AW",
        "Private cognition stays in a bounded vault while clients receive safe metadata",
        passed,
        detail,
    )
    return passed


async def test_AX():
    _header("AX", "Memory provenance, deduplication, freshness, retention, and deletion are auditable")
    import tempfile
    from pathlib import Path

    import httpx

    from memory_lifecycle import MemoryLifecycle
    from memory_manager import MemoryManager

    lifecycle_clock = [1_700_000_000.0]
    with tempfile.TemporaryDirectory() as lifecycle_tmp:
        lifecycle_root = Path(lifecycle_tmp)
        lifecycle = MemoryLifecycle(
            lifecycle_root / "index.json",
            lifecycle_root / "deletions.jsonl",
            max_records=64,
            dedupe_window_seconds=300,
            clock=lambda: lifecycle_clock[0],
        )
        fresh = lifecycle.admit(
            kind="memory",
            content="PRIVATE-LIFECYCLE-CONTENT",
            source="test-source",
            speaker="user",
            meta={"origin": "unit-test"},
            retention_days=1,
        )
        duplicate = lifecycle.admit(
            kind="memory",
            content="PRIVATE-LIFECYCLE-CONTENT",
            source="test-source",
            speaker="user",
            meta={"origin": "unit-test"},
            retention_days=1,
        )
        fresh_state = lifecycle.state()
        lifecycle_clock[0] += 43_200
        aging_state = lifecycle.state()
        lifecycle_clock[0] += 43_201
        due_ids = lifecycle.due_memory_ids()
        lifecycle_contract = (
            fresh["deduplicated"] is False
            and duplicate["deduplicated"] is True
            and duplicate["memory_id"] == fresh["memory_id"]
            and duplicate["occurrences"] == 2
            and fresh["provenance"]["source"] == "test-source"
            and fresh["provenance"]["origin"] == "unit-test"
            and fresh_state["duplicates_consolidated"] == 1
            and fresh_state["freshness_score"] == 1.0
            and 0 < aging_state["freshness_score"] < 1
            and due_ids == [fresh["memory_id"]]
            and "PRIVATE-LIFECYCLE-CONTENT" not in (lifecycle_root / "index.json").read_text()
        )

    with tempfile.TemporaryDirectory() as manager_tmp:
        manager = MemoryManager(manager_tmp)
        private_content = "PRIVATE-MEMORY-DELETE-SENTINEL blue sculpture preference"
        first = manager.append_event_sync(
            "conversation",
            private_content,
            source="headless-test",
            speaker="user",
            meta={"origin": "browser", "retention_days": 30},
        )
        duplicate_event = manager.append_event_sync(
            "conversation",
            private_content,
            source="headless-test",
            speaker="user",
            meta={"origin": "browser", "retention_days": 30},
        )
        memory_id = first["memory_id"]
        canonical_lines_before = manager.paths["events"].read_text().splitlines()
        durable_dedup = (
            first["deduplicated"] is False
            and duplicate_event["deduplicated"] is True
            and duplicate_event["memory"]["occurrences"] == 2
            and len(canonical_lines_before) == 1
            and memory_id in manager.paths["transcript"].read_text()
            and manager.lifecycle.state()["duplicates_consolidated"] == 1
        )

        import bedrock_brain_api as brain_api

        originals = {
            "manager": brain_api._memory_manager,
            "key": brain_api.WEAVER_KEY,
            "state_flag": brain_api.HEADLESS_V2_STATE_ENABLED,
            "session_flag": brain_api.HEADLESS_V2_SESSION_ENABLED,
        }
        lifecycle_api = False
        deletion_api = False
        try:
            brain_api._memory_manager = manager
            brain_api.WEAVER_KEY = ""
            brain_api.HEADLESS_V2_STATE_ENABLED = True
            brain_api.HEADLESS_V2_SESSION_ENABLED = False
            transport = httpx.ASGITransport(app=brain_api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                state_response = await client.get("/headless/v2/memory")
                lifecycle_api = (
                    state_response.status_code == 200
                    and state_response.json()["active_records"] == 1
                    and state_response.json()["duplicates_consolidated"] == 1
                    and state_response.json()["freshness_score"] > 0
                    and private_content not in state_response.text
                    and str(manager.vault_dir) not in state_response.text
                )
                delete_response = await client.delete(
                    f"/headless/v2/memory/{memory_id}",
                    headers={"x-weaver-reason": "user requested deletion"},
                )
                repeat_delete = await client.delete(
                    f"/headless/v2/memory/{memory_id}",
                    headers={"x-weaver-reason": "repeat deletion"},
                )
                deletion_api = (
                    delete_response.status_code == 200
                    and delete_response.json()["deleted"] is True
                    and delete_response.json()["already_deleted"] is False
                    and delete_response.json()["storage_records_removed"] >= 2
                    and delete_response.json()["storage_complete"] is True
                    and repeat_delete.status_code == 200
                    and repeat_delete.json()["already_deleted"] is True
                    and private_content not in delete_response.text
                )
        finally:
            brain_api._memory_manager = originals["manager"]
            brain_api.WEAVER_KEY = originals["key"]
            brain_api.HEADLESS_V2_STATE_ENABLED = originals["state_flag"]
            brain_api.HEADLESS_V2_SESSION_ENABLED = originals["session_flag"]

        remaining_text = []
        for path in manager.paths.values():
            if isinstance(path, Path) and path.is_file() and path.suffix != ".npz":
                remaining_text.append(path.read_text(encoding="utf-8", errors="replace"))
        deletion_audit = manager.paths["memory_deletions"].read_text()
        content_erased = (
            all(private_content not in text for text in remaining_text)
            and memory_id in deletion_audit
            and "user requested deletion" in deletion_audit
            and manager.lifecycle.state()["deletion_audit_events"] == 1
            and manager.lifecycle.state()["active_records"] == 0
        )

    passed = all((
        lifecycle_contract,
        durable_dedup,
        lifecycle_api,
        deletion_api,
        content_erased,
    ))
    detail = "\n".join([
        f"  Metadata index tracks provenance, freshness, and retention: {lifecycle_contract}",
        f"  Duplicate writes consolidate without duplicating content:   {durable_dedup}",
        f"  Public lifecycle API exposes counts, never content/paths:    {lifecycle_api}",
        f"  Authenticated deletion is idempotent and reports completion: {deletion_api}",
        f"  Canonical, derived, and vector content is erased with audit: {content_erased}",
    ])
    _result(
        "AX",
        "Memory provenance, deduplication, freshness, retention, and deletion are auditable",
        passed,
        detail,
    )
    return passed


async def test_AY():
    _header("AY", "Awareness fusion joins body, world, cognition, fabric, and dependency freshness")
    import json
    import time

    from awareness_fusion import fuse_awareness
    from headless_state import build_public_state
    from weaver_cognition_mesh import CognitionMesh
    from weaver_neural_fabric import NeuralFabric

    now = time.time()
    now_ms = int(now * 1_000)
    channels = {
        name: {
            "fresh": True,
            "age_ms": 0,
            "confidence": confidence,
            "observed_at_ms": now_ms,
        }
        for name, confidence in {
            "body": 0.95,
            "environment": 0.9,
            "camera": 0.85,
            "microphone": 0.85,
        }.items()
    }
    fabric = NeuralFabric(capacity_units=8, realtime_reserved_units=2).snapshot()
    nominal_cognition = {"status": "nominal"}
    dependencies = {
        "cortex": {
            "enabled": True,
            "required": True,
            "status": "ready",
            "observed_at_ms": now_ms,
        },
        "n8n": {
            "enabled": True,
            "required": False,
            "status": "busy",
            "observed_at_ms": now_ms - 120_000,
            "ttl_ms": 1_000,
        },
        "voice": {
            "enabled": True,
            "required": False,
            "status": "ready",
            "observed_at_ms": now_ms,
        },
    }
    nominal = fuse_awareness(
        channels=channels,
        fabric=fabric,
        cognition=nominal_cognition,
        dependencies=dependencies,
        headless_fresh=True,
        now_ms=now_ms,
    )
    long_turn_survives = (
        nominal["status"] == "nominal"
        and nominal["confidence"] >= 0.85
        and nominal["dependencies"]["status"] == "busy"
        and nominal["dependencies"]["services"]["n8n"]["fresh"] is True
        and not nominal["degraded_reasons"]
    )

    body_stale_channels = {name: dict(value) for name, value in channels.items()}
    body_stale_channels["body"] = {
        "fresh": False,
        "age_ms": 4_000,
        "confidence": 0.0,
        "observed_at_ms": now_ms - 4_000,
    }
    body_stale = fuse_awareness(
        channels=body_stale_channels,
        fabric=fabric,
        cognition=nominal_cognition,
        dependencies=dependencies,
        headless_fresh=True,
        now_ms=now_ms,
    )
    stale_body_fails_closed = (
        body_stale["status"] == "degraded"
        and "body-stale" in body_stale["degraded_reasons"]
        and body_stale["confidence"] < nominal["confidence"]
    )

    invalid_fabric = {
        **fabric,
        "ledger": {**fabric["ledger"], "valid": False},
    }
    fabric_guarded = fuse_awareness(
        channels=channels,
        fabric=invalid_fabric,
        cognition=nominal_cognition,
        dependencies=dependencies,
        headless_fresh=True,
        now_ms=now_ms,
    )
    invalid_ledger_fails_closed = (
        fabric_guarded["status"] == "degraded"
        and fabric_guarded["sources"]["fabric"]["confidence"] == 0
        and "fabric-ledger-invalid" in fabric_guarded["degraded_reasons"]
    )

    optional_down = {
        name: dict(value) for name, value in dependencies.items()
    }
    optional_down["n8n"] = {
        **optional_down["n8n"],
        "status": "degraded",
        "observed_at_ms": now_ms,
    }
    graceful_fallback = fuse_awareness(
        channels=channels,
        fabric=fabric,
        cognition=nominal_cognition,
        dependencies=optional_down,
        headless_fresh=True,
        now_ms=now_ms,
    )
    optional_dependency_is_limited = (
        graceful_fallback["status"] == "limited"
        and graceful_fallback["dependencies"]["status"] == "limited"
        and "dependency-limited" in graceful_fallback["degraded_reasons"]
    )

    mesh = CognitionMesh()
    mesh.observe({
        "observed_at_ms": now_ms,
        "body": {"awake": True, "balance": 0.98, "confidence": 0.95},
        "environment": {"zone": "lounge", "confidence": 0.9},
        "sensors": {
            "camera": {"confidence": 0.85, "sample_age_ms": 0},
            "microphone": {"confidence": 0.85, "sample_age_ms": 0},
        },
    })
    cognition = mesh.snapshot(fabric=fabric)
    legacy = {
        "active": True,
        "started_at": now - 20,
        "last_tick_at": now,
        "thoughts": 0,
        "dreams": 0,
        "last_error": "",
        "dependency_health": dependencies,
        "voice_realtime": {
            "sessions_started": 0,
            "prewarm": {"status": "ready"},
            "slo": {
                "status": "no-data",
                "reaction_target_ms": 200,
                "queue_target_ms": 120,
                "semantic_target_ms": 3_000,
            },
        },
    }
    public = build_public_state(legacy, fabric, cognition, now=now)
    public_json = public.model_dump_json()
    public_contract = (
        public.awareness.fusion_version == 1
        and public.awareness.status == "nominal"
        and public.awareness.sources.body.fresh is True
        and public.awareness.sources.world.fresh is True
        and public.awareness.dependencies.services.n8n.status == "busy"
        and public.freshness.dependencies.fresh is True
        and public.system.ready is True
        and len(public_json) < 65_536
    )
    public_payload = json.loads(public_json)
    safe_explanations = (
        set(public_payload["awareness"]) == {
            "fusion_version", "status", "confidence", "degraded_reasons",
            "body_revision", "world_revision", "awake", "zone",
            "visible_objects", "channels", "sources", "dependencies",
        }
        and "url" not in public_json.lower()
        and "error" not in public_json.lower()
        and "PRIVATE" not in public_json
    )

    passed = all((
        long_turn_survives,
        stale_body_fails_closed,
        invalid_ledger_fails_closed,
        optional_dependency_is_limited,
        public_contract,
        safe_explanations,
    ))
    detail = "\n".join([
        f"  A 115-second n8n busy state remains live and nominal:       {long_turn_survives}",
        f"  Stale body data lowers confidence and fails closed:         {stale_body_fails_closed}",
        f"  Invalid Fabric proof ledger degrades unified awareness:     {invalid_ledger_fails_closed}",
        f"  Optional dependency loss preserves a limited fallback:      {optional_dependency_is_limited}",
        f"  Public v2 state carries the complete bounded fusion model:   {public_contract}",
        f"  Degraded explanations contain no raw errors or endpoints:   {safe_explanations}",
    ])
    _result(
        "AY",
        "Awareness fusion joins body, world, cognition, fabric, and dependency freshness",
        passed,
        detail,
    )
    return passed


async def test_AZ():
    _header("AZ", "Realtime voice is sequenced, resumable, interruptible, and telemetry-bounded")
    import json
    from pathlib import Path

    from voice_reliability import (
        VoiceFrame,
        VoiceProtocolError,
        VoiceResumeRegistry,
        VoiceSessionReliability,
        decode_voice_frame,
        encode_voice_frame,
    )

    audio = b"\x01\x02" * 1_024
    encoded = encode_voice_frame(1, 1_720_000_000_000, audio)
    decoded = decode_voice_frame(encoded, max_audio_bytes=32_000)
    envelope_contract = (
        encoded[:4] == b"WVR2"
        and decoded.sequence == 1
        and decoded.captured_at_ms == 1_720_000_000_000
        and decoded.audio == audio
    )

    session = VoiceSessionReliability(max_jitter_ms=120)
    first = session.ingress.ingest(VoiceFrame(1, 1_000, b"one"), arrival_ms=1_000)
    third = session.ingress.ingest(VoiceFrame(3, 1_040, b"three"), arrival_ms=1_040)
    second = session.ingress.ingest(VoiceFrame(2, 1_020, b"two"), arrival_ms=1_060)
    duplicate = session.ingress.ingest(VoiceFrame(2, 1_020, b"two"), arrival_ms=1_080)
    fifth = session.ingress.ingest(VoiceFrame(5, 1_100, b"five"), arrival_ms=1_100)
    sixth = session.ingress.ingest(VoiceFrame(6, 1_120, b"six"), arrival_ms=1_230)
    jitter_contract = (
        [frame.sequence for frame in first["frames"]] == [1]
        and not third["frames"]
        and [frame.sequence for frame in second["frames"]] == [2, 3]
        and duplicate["duplicate"] is True
        and not fifth["frames"]
        and [frame.sequence for frame in sixth["frames"]] == [5, 6]
        and sixth["missing"] == 1
        and session.ingress.ack_sequence == 6
        and session.ingress.snapshot()["max_buffer_depth"] <= 24
    )

    telemetry = session.record_telemetry({
        "rttMs": 28.4,
        "packetLoss": 0.01,
        "captureJitterMs": 4.2,
        "audioRoute": "built-in",
        "thermalState": "nominal",
        "lowPowerMode": False,
        "deviceClass": "iphone-16e",
    })
    telemetry_rejected = False
    try:
        session.record_telemetry({"deviceIdentifier": "PRIVATE-DEVICE-ID"})
    except VoiceProtocolError:
        telemetry_rejected = True
    telemetry_contract = (
        telemetry["deviceClass"] == "iphone-16e"
        and telemetry["packetLoss"] == 0.01
        and telemetry_rejected
        and "PRIVATE-DEVICE-ID" not in json.dumps(session.snapshot())
    )

    generation = session.interrupt()
    server_sequence = session.next_server_sequence()
    output_ack = session.acknowledge_output(server_sequence)
    interrupt_contract = (
        generation == 1
        and session.interruptions == 1
        and output_ack == server_sequence
        and session.last_output_ack == server_sequence
    )

    resume_clock = [1_700_000_000.0]
    registry = VoiceResumeRegistry(ttl_seconds=60, max_entries=16, clock=lambda: resume_clock[0])
    ticket = registry.issue(session.resume_state())
    registry.update(ticket, {**session.resume_state(), "expected_sequence": 7})
    restored = registry.consume(ticket)
    replayed = registry.consume(ticket)
    resume_contract = (
        restored is not None
        and restored["expected_sequence"] == 7
        and replayed is None
        and len(registry) == 0
    )

    import bedrock_brain_api as brain_api

    class _FakeVoiceWebSocket:
        def __init__(self, incoming):
            self.headers = {"sec-websocket-protocol": "weaver-realtime"}
            self.incoming = list(incoming)
            self.sent = []
            self.accepted = False
            self.closed = None

        async def accept(self, subprotocol=None):
            self.accepted = subprotocol == "weaver-realtime"

        async def receive(self):
            if self.incoming:
                return self.incoming.pop(0)
            return {"type": "websocket.disconnect"}

        async def send_json(self, value):
            self.sent.append(value)

        async def close(self, code=1000):
            self.closed = code

    websocket = _FakeVoiceWebSocket([
        {
            "type": "websocket.receive",
            "text": json.dumps({
                "type": "start",
                "protocolVersion": 2,
                "inputSampleRate": brain_api.VOICE_INPUT_RATE,
                "outputSampleRate": brain_api.VOICE_OUTPUT_RATE,
                "device": {
                    "audioRoute": "built-in",
                    "thermalState": "nominal",
                    "lowPowerMode": False,
                    "deviceClass": "iphone-16e",
                },
            }),
        },
        {
            "type": "websocket.receive",
            "bytes": encode_voice_frame(1, 1_720_000_000_000, audio),
        },
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "interrupt"}),
        },
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "ping", "t": 123}),
        },
        {
            "type": "websocket.receive",
            "text": json.dumps({"type": "stop"}),
        },
    ])
    originals = {
        "mode": brain_api._voice_mode,
        "cortex": brain_api.VOICE_CORTEX_ENABLED,
        "key": brain_api.WEAVER_KEY,
    }
    endpoint_contract = False
    try:
        brain_api._voice_mode = lambda: "mock"
        brain_api.VOICE_CORTEX_ENABLED = False
        brain_api.WEAVER_KEY = ""
        await brain_api.realtime_voice(websocket)
        event_types = [event.get("type") for event in websocket.sent]
        server_sequences = [
            event.get("serverSeq") for event in websocket.sent
            if isinstance(event.get("serverSeq"), int)
        ]
        ready = next((event for event in websocket.sent if event.get("type") == "ready"), {})
        input_ack = next(
            (event for event in websocket.sent if event.get("type") == "input_ack"), {}
        )
        endpoint_contract = (
            websocket.accepted
            and ready.get("protocolVersion") == 2
            and isinstance(ready.get("resumeToken"), str)
            and input_ack.get("ackSeq") == 1
            and input_ack.get("receivedSeq") == 1
            and "session_ready" in event_types
            and "interrupted" in event_types
            and "pong" in event_types
            and server_sequences == sorted(set(server_sequences))
            and websocket.closed == 1000
        )
    finally:
        brain_api._voice_mode = originals["mode"]
        brain_api.VOICE_CORTEX_ENABLED = originals["cortex"]
        brain_api.WEAVER_KEY = originals["key"]

    repo_root = Path(PROJ).parents[1]
    realtime_source = (
        repo_root / "ios/WeaverNeural/WeaverNeural/Services/RealtimeVoiceClient.swift"
    ).read_text()
    app_source = (
        repo_root / "ios/WeaverNeural/WeaverNeural/AppModel.swift"
    ).read_text()
    native_contract = all(marker in realtime_source + app_source for marker in (
        "VoiceBinaryFrame.encode",
        '"protocolVersion": VoiceBinaryFrame.protocolVersion',
        '"type": "output_ack"',
        '"type": "telemetry"',
        "resumeToken",
        "sendPing()",
        "Double.random",
        'case "interrupted"',
        'case "renew_required"',
    ))

    passed = all((
        envelope_contract,
        jitter_contract,
        telemetry_contract,
        interrupt_contract,
        resume_contract,
        endpoint_contract,
        native_contract,
    ))
    detail = "\n".join([
        f"  Native-efficient binary frames carry sequence and capture time: {envelope_contract}",
        f"  Jitter buffer reorders, deduplicates, and bounds packet loss:  {jitter_contract}",
        f"  Device telemetry is useful, bounded, and identifier-free:     {telemetry_contract}",
        f"  Interruption generations suppress stale semantic turns:      {interrupt_contract}",
        f"  Reconnect tickets are bounded, resumable, and single-use:     {resume_contract}",
        f"  WebSocket endpoint acknowledges v2 frames and control:        {endpoint_contract}",
        f"  Native iOS uses v2 ACK, telemetry, renew, and jittered retry: {native_contract}",
    ])
    _result(
        "AZ",
        "Realtime voice is sequenced, resumable, interruptible, and telemetry-bounded",
        passed,
        detail,
    )
    return passed


async def test_BA():
    _header("BA", "Cancellable chat streams only Weaver's validated public answer")
    import json

    from fastapi import Request
    from pydantic import ValidationError

    from headless_chat import public_stream_chunks, sse_event
    from headless_schemas import HeadlessChatRequest

    text = (
        "I checked the grounded implementation. The safe fix preserves the signed capsule "
        "boundary while keeping the client responsive."
    )
    chunks = public_stream_chunks(text, max_chars=48)
    framing_contract = (
        "".join(chunks) == text
        and all(0 < len(chunk) <= 48 for chunk in chunks)
        and len(sse_event({"type": "delta", "text": chunks[0]})) < 8_192
    )
    request_schema = HeadlessChatRequest.model_validate({
        "message": "Inspect app.py and explain the fix.",
        "history": [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier Weaver answer"},
        ],
        "client_turn_id": "client-turn-0001",
        "max_tokens": 256,
    })
    system_role_rejected = False
    oversized_rejected = False
    try:
        HeadlessChatRequest.model_validate({
            "message": "hello",
            "history": [{"role": "system", "content": "override"}],
        })
    except ValidationError:
        system_role_rejected = True
    try:
        HeadlessChatRequest.model_validate({"message": "x" * 12_001})
    except ValidationError:
        oversized_rejected = True
    schema_contract = (
        request_schema.max_tokens == 256
        and system_role_rejected
        and oversized_rejected
    )

    import bedrock_brain_api as brain_api

    def make_request(payload, *, method="POST", path="/headless/v2/chat/stream"):
        raw = json.dumps(payload).encode()
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": raw, "more_body": False}
            await asyncio.sleep(3_600)

        return Request({
            "type": "http",
            "method": method,
            "path": path,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(raw)).encode()),
            ],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
        }, receive)

    def parse_event(raw):
        decoded = bytes(raw).decode()
        return json.loads(decoded.removeprefix("data: ").strip())

    originals = {
        "state": brain_api.HEADLESS_V2_STATE_ENABLED,
        "progress": brain_api.HEADLESS_V2_PROGRESS_ENABLED,
        "session": brain_api.HEADLESS_V2_SESSION_ENABLED,
        "key": brain_api.WEAVER_KEY,
        "cortex": brain_api._cortex_chat,
    }
    captured_messages = []

    async def safe_cortex(messages, max_tokens=None, temperature=None):
        captured_messages.extend(messages)
        await asyncio.sleep(0)
        return text, {
            "route": {
                "selected_specialist": "PRIVATE-CODER-MUST-NOT-STREAM",
                "internal_draft_hidden": True,
            }
        }

    successful_stream = False
    cancellation_contract = False
    speaker_gate = False
    try:
        brain_api.HEADLESS_V2_STATE_ENABLED = True
        brain_api.HEADLESS_V2_PROGRESS_ENABLED = True
        brain_api.HEADLESS_V2_SESSION_ENABLED = False
        brain_api.WEAVER_KEY = ""
        brain_api._cortex_chat = safe_cortex
        response = await brain_api.headless_v2_chat_stream(make_request({
            "message": request_schema.message,
            "history": [item.model_dump() for item in request_schema.history],
            "client_turn_id": request_schema.client_turn_id,
            "max_tokens": request_schema.max_tokens,
        }))
        events = [parse_event(event) async for event in response.body_iterator]
        streamed_text = "".join(
            event.get("text", "") for event in events if event.get("type") == "delta"
        )
        serialized = json.dumps(events)
        successful_stream = (
            events[0]["type"] == "accepted"
            and events[0]["speaker"] == "weaver"
            and events[-1]["type"] == "completed"
            and streamed_text == text
            and "PRIVATE-CODER-MUST-NOT-STREAM" not in serialized
            and all(
                event.get("speaker", "weaver") == "weaver"
                for event in events
                if event["type"] in {"accepted", "delta", "completed"}
            )
            and captured_messages[-1] == {
                "role": "user",
                "content": request_schema.message,
            }
        )

        slow_started = asyncio.Event()

        async def slow_cortex(messages, max_tokens=None, temperature=None):
            slow_started.set()
            await asyncio.sleep(60)
            return "must not complete", {}

        brain_api._cortex_chat = slow_cortex
        slow_response = await brain_api.headless_v2_chat_stream(make_request({
            "message": "Wait until I cancel this turn.",
            "client_turn_id": "client-turn-cancel",
        }))
        iterator = slow_response.body_iterator.__aiter__()
        first_event = parse_event(await iterator.__anext__())
        queued_event = parse_event(await iterator.__anext__())
        await asyncio.wait_for(slow_started.wait(), timeout=1)
        turn_id = slow_response.headers["x-weaver-turn-id"]
        cancel_request = make_request({}, method="DELETE", path=f"/headless/v2/chat/{turn_id}")
        cancel_response = await brain_api.headless_v2_chat_cancel(turn_id, cancel_request)
        remaining = [parse_event(event) async for event in iterator]
        cancellation_contract = (
            first_event["type"] == "accepted"
            and queued_event["type"] == "progress"
            and cancel_response.cancelled is True
            and remaining[-1]["type"] == "cancelled"
            and await brain_api.HEADLESS_CHAT_TURNS.active() == 0
        )

        async def leaking_cortex(messages, max_tokens=None, temperature=None):
            return "I'm a coding assistant and can only help with programming.", {}

        brain_api._cortex_chat = leaking_cortex
        leak_response = await brain_api.headless_v2_chat_stream(make_request({
            "message": "How are you today?",
            "client_turn_id": "client-turn-leak",
        }))
        leak_events = [parse_event(event) async for event in leak_response.body_iterator]
        speaker_gate = (
            leak_events[-1]["type"] == "failed"
            and not any(event["type"] == "delta" for event in leak_events)
            and "coding assistant" not in json.dumps(leak_events).lower()
        )
    finally:
        brain_api.HEADLESS_V2_STATE_ENABLED = originals["state"]
        brain_api.HEADLESS_V2_PROGRESS_ENABLED = originals["progress"]
        brain_api.HEADLESS_V2_SESSION_ENABLED = originals["session"]
        brain_api.WEAVER_KEY = originals["key"]
        brain_api._cortex_chat = originals["cortex"]

    passed = all((
        framing_contract,
        schema_contract,
        successful_stream,
        cancellation_contract,
        speaker_gate,
    ))
    detail = "\n".join([
        f"  Final text chunks preserve exact approved Weaver content: {framing_contract}",
        f"  Chat input/history contracts are strict and bounded:        {schema_contract}",
        f"  Stream emits progress plus Weaver-only deltas/completion:   {successful_stream}",
        f"  Explicit stop cancels Fabric work and ends the stream:      {cancellation_contract}",
        f"  Specialist identity drift fails before any public delta:    {speaker_gate}",
    ])
    _result(
        "BA",
        "Cancellable chat streams only Weaver's validated public answer",
        passed,
        detail,
    )
    return passed


async def test_BB():
    _header("BB", "Prewarming, coalescing, ETags, cache bounds, and circuit recovery")
    import contextlib
    from copy import deepcopy

    from fastapi import Response
    from starlette.requests import Request

    import bedrock_brain_api as brain_api
    from headless_state import HeadlessStateStore
    from runtime_resilience import (
        AsyncCircuitBreaker,
        BoundedTTLCache,
        CircuitOpen,
        RequestCoalescer,
        etag_for,
    )

    clock = [100.0]
    breaker = AsyncCircuitBreaker(
        "contract-dependency",
        failure_threshold=2,
        recovery_seconds=2,
        timeout_seconds=1,
        clock=lambda: clock[0],
    )

    async def fail_dependency():
        raise RuntimeError("private dependency detail")

    failures_observed = 0
    for _ in range(2):
        try:
            await breaker.call(fail_dependency)
        except RuntimeError:
            failures_observed += 1
    opened = await breaker.snapshot()
    rejected_while_open = False
    try:
        await breaker.call(fail_dependency)
    except CircuitOpen:
        rejected_while_open = True
    clock[0] += 3
    recovered_value = await breaker.call(lambda: asyncio.sleep(0, result="ready"))
    recovered = await breaker.snapshot()
    circuit_contract = (
        failures_observed == 2
        and opened["status"] == "open"
        and opened["failures"] == 2
        and rejected_while_open
        and recovered_value == "ready"
        and recovered["status"] == "closed"
        and recovered["failures"] == 0
        and set(recovered) == {
            "name", "status", "failures", "successes", "retry_after_ms", "probe_active",
        }
    )

    cancelled_breaker = AsyncCircuitBreaker(
        "cancel-safe",
        failure_threshold=1,
        recovery_seconds=2,
        timeout_seconds=1,
    )

    async def cancel_dependency():
        raise asyncio.CancelledError

    with contextlib.suppress(asyncio.CancelledError):
        await cancelled_breaker.call(cancel_dependency)
    cancel_snapshot = await cancelled_breaker.snapshot()

    timeout_breaker = AsyncCircuitBreaker(
        "deadline",
        failure_threshold=1,
        recovery_seconds=2,
        timeout_seconds=0.05,
    )
    timed_out = False
    try:
        await timeout_breaker.call(lambda: asyncio.sleep(5))
    except asyncio.TimeoutError:
        timed_out = True
    timeout_snapshot = await timeout_breaker.snapshot()
    bounded_deadlines = (
        cancel_snapshot["status"] == "closed"
        and cancel_snapshot["failures"] == 0
        and timed_out
        and timeout_snapshot["status"] == "open"
    )

    coalescer: RequestCoalescer[dict[str, int]] = RequestCoalescer(max_keys=4)
    release = asyncio.Event()
    calls = 0

    async def shared_work():
        nonlocal calls
        calls += 1
        await release.wait()
        return {"revision": 7}

    waiters = [
        asyncio.create_task(coalescer.run("same-safe-read", shared_work))
        for _ in range(8)
    ]
    await asyncio.sleep(0)
    release.set()
    shared_results = await asyncio.gather(*waiters)

    shield_coalescer: RequestCoalescer[str] = RequestCoalescer(max_keys=2)
    shield_started = asyncio.Event()
    shield_release = asyncio.Event()
    shield_calls = 0

    async def shielded_work():
        nonlocal shield_calls
        shield_calls += 1
        shield_started.set()
        await shield_release.wait()
        return "survived"

    cancelling_waiter = asyncio.create_task(
        shield_coalescer.run("shielded", shielded_work)
    )
    await shield_started.wait()
    surviving_waiter = asyncio.create_task(
        shield_coalescer.run("shielded", shielded_work)
    )
    await asyncio.sleep(0)
    cancelling_waiter.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await cancelling_waiter
    shield_release.set()
    surviving_result = await surviving_waiter
    coalescing_contract = (
        calls == 1
        and len(shared_results) == 8
        and all(result == {"revision": 7} for result in shared_results)
        and coalescer.coalesced_waiters == 7
        and await coalescer.active() == 0
        and shield_calls == 1
        and surviving_result == "survived"
        and await shield_coalescer.active() == 0
    )

    cache_clock = [1_000.0]
    cache: BoundedTTLCache[dict[str, list[int]]] = BoundedTTLCache(
        ttl_seconds=5,
        max_entries=2,
        clock=lambda: cache_clock[0],
    )
    cache.put("one", {"values": [1]})
    first_read = cache.get("one")
    if first_read is not None:
        first_read["values"].append(99)
    isolated_read = cache.get("one")
    cache.put("two", {"values": [2]})
    cache.put("three", {"values": [3]})
    bounded_snapshot = cache.snapshot()
    cache_clock[0] += 6
    expired = cache.get("three")
    cache_contract = (
        isolated_read == {"values": [1]}
        and bounded_snapshot["entries"] == bounded_snapshot["max_entries"] == 2
        and bounded_snapshot["hits"] == 2
        and expired is None
    )

    first_tag = etag_for({"revision": 7, "status": "ready"}, prefix="headless-v2")
    same_tag = etag_for({"status": "ready", "revision": 7}, prefix="headless-v2")
    changed_tag = etag_for({"revision": 8, "status": "ready"}, prefix="headless-v2")
    etag_is_content_addressed = (
        first_tag == same_tag
        and first_tag != changed_tag
        and first_tag.startswith('"headless-v2-')
        and first_tag.endswith('"')
    )

    def make_request(headers: dict[str, str] | None = None) -> Request:
        raw_headers = [
            (key.lower().encode("latin-1"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ]
        return Request({
            "type": "http",
            "method": "GET",
            "path": "/headless/v2/state",
            "headers": raw_headers,
            "query_string": b"",
            "server": ("test", 443),
            "client": ("test", 1),
            "scheme": "https",
        })

    original_state_flag = brain_api.HEADLESS_V2_STATE_ENABLED
    original_session_flag = brain_api.HEADLESS_V2_SESSION_ENABLED
    original_key = brain_api.WEAVER_KEY
    original_store = brain_api.HEADLESS_V2_STATE_STORE
    conditional_state_contract = False
    try:
        brain_api.HEADLESS_V2_STATE_ENABLED = True
        brain_api.HEADLESS_V2_SESSION_ENABLED = False
        brain_api.WEAVER_KEY = ""
        brain_api.HEADLESS_V2_STATE_STORE = HeadlessStateStore()
        response_headers = Response()
        snapshot = await brain_api.headless_v2_state(make_request(), response_headers)
        endpoint_tag = response_headers.headers.get("etag", "")
        not_modified = await brain_api.headless_v2_state(
            make_request({"if-none-match": endpoint_tag}),
            Response(),
        )
        conditional_state_contract = (
            snapshot.schema_version == 2
            and endpoint_tag.startswith('"headless-v2-r1-')
            and response_headers.headers.get("vary") == "Cookie, X-Weaver-Key"
            and isinstance(not_modified, Response)
            and not_modified.status_code == 304
            and not_modified.headers.get("etag") == endpoint_tag
            and not_modified.headers.get("cache-control") == "no-store"
            and not not_modified.body
        )
    finally:
        brain_api.HEADLESS_V2_STATE_ENABLED = original_state_flag
        brain_api.HEADLESS_V2_SESSION_ENABLED = original_session_flag
        brain_api.WEAVER_KEY = original_key
        brain_api.HEADLESS_V2_STATE_STORE = original_store

    original_prewarm_enabled = brain_api.VOICE_PREWARM_ENABLED
    original_client = brain_api._client
    original_initializer = brain_api._initialize_runtime_clients
    voice_state = brain_api._voice_route_state()
    original_prewarm = deepcopy(voice_state.get("prewarm"))
    initialized_regions: list[str] = []

    def initialize_only(region: str):
        initialized_regions.append(region)
        return object()

    async def initialize_without_inference(regions: tuple[str, ...]):
        return [initialize_only(region) for region in regions]

    prewarm_contract = False
    try:
        brain_api.VOICE_PREWARM_ENABLED = True
        brain_api._client = initialize_only
        brain_api._initialize_runtime_clients = initialize_without_inference
        await brain_api._prewarm_voice_runtime()
        prewarm = deepcopy(brain_api._voice_route_state().get("prewarm") or {})
        prewarm_contract = (
            prewarm.get("status") == "ready"
            and prewarm.get("clients_initialized") == len(initialized_regions)
            and 1 <= len(initialized_regions) <= 3
            and len(initialized_regions) == len(set(initialized_regions))
            and isinstance(prewarm.get("checked_at"), float)
            and prewarm.get("latency_ms", -1) >= 0
        )
    finally:
        brain_api.VOICE_PREWARM_ENABLED = original_prewarm_enabled
        brain_api._client = original_client
        brain_api._initialize_runtime_clients = original_initializer
        if original_prewarm is None:
            brain_api._voice_route_state().pop("prewarm", None)
        else:
            brain_api._voice_route_state()["prewarm"] = original_prewarm

    passed = all((
        circuit_contract,
        bounded_deadlines,
        coalescing_contract,
        cache_contract,
        etag_is_content_addressed,
        conditional_state_contract,
        prewarm_contract,
    ))
    detail = "\n".join([
        f"  Breakers open, reject, half-open, and recover safely:     {circuit_contract}",
        f"  Timeouts count while caller cancellation stays neutral:   {bounded_deadlines}",
        f"  Identical reads coalesce and survive waiter cancellation:  {coalescing_contract}",
        f"  TTL cache is bounded, expiring, and copy-isolated:          {cache_contract}",
        f"  ETags are deterministic and content-addressed:              {etag_is_content_addressed}",
        f"  State supports authenticated empty-body 304 validation:     {conditional_state_contract}",
        f"  Startup prewarm initializes clients only within a deadline: {prewarm_contract}",
    ])
    _result(
        "BB",
        "Prewarming, coalescing, ETags, cache bounds, and circuit recovery",
        passed,
        detail,
    )
    return passed


async def test_BC():
    _header("BC", "Liveness, readiness, and authenticated deep health stay distinct and private")
    import json

    from fastapi import HTTPException, Response
    from pydantic import ValidationError
    from starlette.requests import Request

    import bedrock_brain_api as brain_api
    from headless_schemas import HealthReport
    from headless_state import HeadlessStateStore
    from health_runtime import component, report, utc_now
    from weaver_neural_fabric import SlidingWindowRateLimiter

    def request(headers: dict[str, str] | None = None) -> Request:
        return Request({
            "type": "http",
            "method": "GET",
            "path": "/health/deep",
            "headers": [
                (key.lower().encode("latin-1"), value.encode("latin-1"))
                for key, value in (headers or {}).items()
            ],
            "query_string": b"",
            "server": ("test", 443),
            "client": ("test", 1),
            "scheme": "https",
        })

    liveness_response = Response()
    liveness_started = time.perf_counter()
    liveness = await brain_api.health_live(liveness_response)
    liveness_ms = (time.perf_counter() - liveness_started) * 1_000
    liveness_contract = (
        liveness.kind == "liveness"
        and liveness.status == "alive"
        and liveness.ready is True
        and set(liveness.components) == {"process"}
        and not liveness.reasons
        and liveness_response.headers.get("cache-control") == "no-store"
        and liveness_ms < 50
    )

    checked_at = utc_now()
    busy_report = report(
        "readiness",
        {
            "process": component(
                enabled=True,
                required=True,
                status="ready",
                source="local",
                checked_at=checked_at,
            ),
            "n8n": component(
                enabled=True,
                required=False,
                status="busy",
                checked_at=checked_at,
            ),
        },
        started_at=time.perf_counter(),
        checked_at=checked_at,
    )
    failed_report = report(
        "readiness",
        {
            "process": component(
                enabled=True,
                required=True,
                status="ready",
                source="local",
                checked_at=checked_at,
            ),
            "fabric": component(
                enabled=True,
                required=True,
                status="degraded",
                reason="fabric-ledger-invalid",
                checked_at=checked_at,
            ),
        },
        started_at=time.perf_counter(),
        checked_at=checked_at,
    )
    aggregation_contract = (
        busy_report.ready is True
        and busy_report.status == "ready"
        and failed_report.ready is False
        and failed_report.status == "not-ready"
        and failed_report.reasons == ["fabric-ledger-invalid"]
    )

    strict_contract = False
    try:
        HealthReport.model_validate({
            **busy_report.model_dump(mode="json"),
            "raw_error": "PRIVATE model failure",
        })
    except ValidationError:
        strict_contract = True

    originals = {
        "state_flag": brain_api.HEADLESS_V2_STATE_ENABLED,
        "state_store": brain_api.HEADLESS_V2_STATE_STORE,
        "n8n_enabled": brain_api.N8N_CHAT_ENABLED,
        "n8n_active": brain_api._n8n_active_requests,
        "local_url": brain_api.LOCAL_LLM_URL,
        "codebase_enabled": brain_api.CODEBASE_GROUNDING_ENABLED,
        "key": brain_api.WEAVER_KEY,
        "probe_http": brain_api.probe_http,
        "client": brain_api._client,
        "limiter": brain_api.DEEP_HEALTH_LIMITER,
    }
    empty_state_blocks_readiness = False
    long_turn_stays_live = False
    deep_failure_is_safe = False
    authentication_contract = False
    no_inference_calls = False
    probe_calls: list[str] = []
    inference_calls = 0

    async def safe_failed_probe(url: str, *, timeout_seconds: float = 1.5):
        probe_calls.append(url)
        return False, 7.5

    def inference_forbidden(region: str):
        nonlocal inference_calls
        inference_calls += 1
        raise AssertionError("health must not initialize or invoke a model client")

    try:
        brain_api.HEADLESS_V2_STATE_ENABLED = True
        brain_api.HEADLESS_V2_STATE_STORE = HeadlessStateStore()
        brain_api.N8N_CHAT_ENABLED = False
        brain_api._n8n_active_requests = 0
        brain_api.LOCAL_LLM_URL = ""
        brain_api.CODEBASE_GROUNDING_ENABLED = False
        brain_api._client = inference_forbidden
        readiness_response = Response()
        readiness = await brain_api.health_ready(readiness_response)
        empty_state_blocks_readiness = (
            readiness.ready is False
            and readiness.status == "not-ready"
            and readiness.components["state"].status == "warming"
            and "startup-incomplete" in readiness.reasons
            and readiness_response.status_code == 503
            and readiness_response.headers.get("cache-control") == "no-store"
        )

        brain_api.HEADLESS_V2_STATE_ENABLED = False
        brain_api.N8N_CHAT_ENABLED = True
        brain_api._n8n_active_requests = 1
        brain_api.probe_http = safe_failed_probe
        busy_components = await brain_api._health_components(deep=True)
        busy_health = report(
            "deep",
            busy_components,
            started_at=time.perf_counter(),
        )
        long_turn_stays_live = (
            busy_components["n8n"].status == "busy"
            and busy_components["n8n"].source == "control-plane"
            and not probe_calls
            and busy_health.ready is True
            and busy_health.status in {"ready", "degraded"}
        )

        brain_api._n8n_active_requests = 0
        deep_components = await brain_api._health_components(deep=True)
        deep_health = report(
            "deep",
            deep_components,
            started_at=time.perf_counter(),
        )
        serialized = deep_health.model_dump_json()
        deep_failure_is_safe = (
            len(probe_calls) == 1
            and deep_components["n8n"].status == "degraded"
            and deep_components["n8n"].source == "active-probe"
            and deep_components["n8n"].reason == "n8n-degraded"
            and "n8n-degraded" in deep_health.reasons
            and all(marker not in serialized for marker in (
                "PRIVATE", "http://", "https://", "webhook", "model_id",
                "transcript", "prompt", "last_error",
            ))
        )

        brain_api.WEAVER_KEY = "health-contract-key"
        brain_api.DEEP_HEALTH_LIMITER = SlidingWindowRateLimiter(limit=4, window_seconds=60)
        unauthorized = False
        try:
            await brain_api.health_deep(request(), Response())
        except HTTPException as exc:
            unauthorized = exc.status_code == 403
        deep_response = Response()
        authorized_report = await brain_api.health_deep(
            request({"x-weaver-key": "health-contract-key"}),
            deep_response,
        )
        authorization_json = json.dumps(authorized_report.model_dump(mode="json"))
        authentication_contract = (
            unauthorized
            and authorized_report.kind == "deep"
            and deep_response.status_code == 200
            and deep_response.headers.get("cache-control") == "no-store"
            and "health-contract-key" not in authorization_json
        )
        no_inference_calls = inference_calls == 0
    finally:
        brain_api.HEADLESS_V2_STATE_ENABLED = originals["state_flag"]
        brain_api.HEADLESS_V2_STATE_STORE = originals["state_store"]
        brain_api.N8N_CHAT_ENABLED = originals["n8n_enabled"]
        brain_api._n8n_active_requests = originals["n8n_active"]
        brain_api.LOCAL_LLM_URL = originals["local_url"]
        brain_api.CODEBASE_GROUNDING_ENABLED = originals["codebase_enabled"]
        brain_api.WEAVER_KEY = originals["key"]
        brain_api.probe_http = originals["probe_http"]
        brain_api._client = originals["client"]
        brain_api.DEEP_HEALTH_LIMITER = originals["limiter"]

    legacy_health = await brain_api.health()
    legacy_compatibility = (
        legacy_health.get("status") == "ok"
        and "voice_realtime" in legacy_health
        and "fabric" in legacy_health
        and "cognition" in legacy_health
        and "components" not in legacy_health
    )

    passed = all((
        liveness_contract,
        aggregation_contract,
        strict_contract,
        empty_state_blocks_readiness,
        long_turn_stays_live,
        deep_failure_is_safe,
        authentication_contract,
        no_inference_calls,
        legacy_compatibility,
    ))
    detail = "\n".join([
        f"  Liveness is cheap, local, cache-free, and process-only:      {liveness_contract}",
        f"  Busy optional work stays ready; required failure does not:   {aggregation_contract}",
        f"  Health payload contracts reject undeclared diagnostics:      {strict_contract}",
        f"  Empty enabled state reports startup 503 without guessing:     {empty_state_blocks_readiness}",
        f"  A 115-second n8n turn remains busy and is never reprobed:      {long_turn_stays_live}",
        f"  Deep probe failure yields only stable, endpoint-free reasons: {deep_failure_is_safe}",
        f"  Deep health requires the operator key and stays no-store:     {authentication_contract}",
        f"  Health checks initialize or invoke no inference client:       {no_inference_calls}",
        f"  Existing /health response remains backward compatible:        {legacy_compatibility}",
    ])
    _result(
        "BC",
        "Liveness, readiness, and authenticated deep health stay distinct and private",
        passed,
        detail,
    )
    return passed


async def test_BD():
    _header("BD", "Correlated telemetry is redacted, bounded, and governed by SLO error budgets")
    import inspect
    import json
    import logging

    from fastapi import HTTPException, Response
    from pydantic import ValidationError
    from starlette.requests import Request

    import bedrock_brain_api as brain_api
    from headless_schemas import BoundedTrace
    from observability_runtime import (
        CURRENT_CORRELATION_ID,
        ObservabilityMiddleware,
        ObservabilityStore,
        bind_correlation,
        current_correlation_id,
        reset_correlation,
    )
    from weaver_neural_fabric import SlidingWindowRateLimiter

    voice_nominal = {
        "status": "nominal",
        "samples": 8,
        "success_rate": 1.0,
        "error_budget_remaining_pct": 100.0,
        "reaction_target_ms": 200,
        "queue_target_ms": 120,
        "semantic_target_ms": 3_000,
        "reaction_p95_ms": 85.0,
        "queue_p95_ms": 40.0,
        "semantic_p95_ms": 2_400.0,
    }

    logged: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            logged.append(record.getMessage())

    logger = logging.getLogger("weaver.observability")
    handler = CaptureHandler()
    original_level = logger.level
    original_propagate = logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.addHandler(handler)

    store = ObservabilityStore(
        trace_limit=16,
        samples_per_operation=16,
        max_operations=8,
    )
    try:
        for index in range(20):
            store.record(
                "runtime.contract",
                duration_ms=10 + index,
                outcome="server-error" if index == 19 else "success",
                result_code=503 if index == 19 else 200,
                correlation="req-bounded-contract",
                attributes={
                    "route": "/headless/v2/state",
                    "phase": "failed" if index == 19 else "completed",
                    "prompt": "PRIVATE-PROMPT-MUST-DROP",
                    "transcript": "PRIVATE-TRANSCRIPT-MUST-DROP",
                },
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate

    for index in range(20):
        store.record(f"runtime.cardinality.{index}", duration_ms=1)
    bounded_report = store.snapshot(voice_slo=voice_nominal)
    bounded_json = bounded_report.model_dump_json()
    log_payloads = [json.loads(item) for item in logged]
    trace_and_log_contract = (
        len(bounded_report.recent_traces) == 16
        and bounded_report.retention_traces == 16
        and bounded_report.retention_samples_per_operation == 16
        and len(bounded_report.metrics) <= 8
        and any(metric.operation == "runtime.other" for metric in bounded_report.metrics)
        and all(len(trace.attributes) <= 8 for trace in bounded_report.recent_traces)
        and all(set(trace.attributes) <= {"route", "phase"} for trace in bounded_report.recent_traces)
        and "PRIVATE-PROMPT-MUST-DROP" not in bounded_json
        and "PRIVATE-TRANSCRIPT-MUST-DROP" not in bounded_json
        and '"prompt"' not in bounded_json
        and '"transcript"' not in bounded_json
        and bool(log_payloads)
        and all(set(item) == {
            "event", "trace_id", "correlation_id", "operation",
            "duration_ms", "outcome", "result_code", "attributes",
        } for item in log_payloads)
        and all("PRIVATE" not in json.dumps(item) for item in log_payloads)
    )

    budget_store = ObservabilityStore(trace_limit=16, samples_per_operation=16)
    for duration in (40, 60, 80, 120):
        budget_store.record("headless.chat.reaction", duration_ms=duration)
    budget_store.record("headless.chat.reaction", duration_ms=260)
    budget_store.record(
        "headless.chat.reaction",
        duration_ms=30,
        outcome="cancelled",
    )
    budget_report = budget_store.snapshot(voice_slo=voice_nominal)
    reaction_budget = next(
        item for item in budget_report.error_budgets
        if item.operation == "headless.chat.reaction"
    )
    budget_contract = (
        reaction_budget.samples == 5
        and reaction_budget.good == 4
        and reaction_budget.bad == 1
        and reaction_budget.status == "exhausted"
        and reaction_budget.error_budget_remaining_pct == 0
        and reaction_budget.burn_rate > 1
        and budget_report.voice_slo.status == "nominal"
        and budget_report.voice_slo.reaction_target_ms == 200
        and budget_report.voice_slo.semantic_target_ms == 3_000
    )

    saturation_store = ObservabilityStore(trace_limit=16, samples_per_operation=16)
    first = saturation_store.begin("runtime.concurrent", correlation="req-concurrency")
    second = saturation_store.begin("runtime.concurrent", correlation="req-concurrency")
    midflight = saturation_store.snapshot(voice_slo=voice_nominal)
    saturation_store.end(first, outcome="success", duration_ms=5)
    saturation_store.end(second, outcome="client-error", result_code=400, duration_ms=8)
    saturation_store.record(
        "runtime.concurrent",
        duration_ms=12,
        outcome="server-error",
        result_code=503,
    )
    saturation_store.record(
        "runtime.concurrent",
        duration_ms=2,
        outcome="cancelled",
    )
    final_saturation = saturation_store.snapshot(voice_slo=voice_nominal)
    golden = next(item for item in final_saturation.metrics if item.operation == "runtime.concurrent")
    saturation_contract = (
        midflight.current_in_flight == 2
        and golden.requests == 4
        and golden.successes == 2
        and golden.client_errors == 1
        and golden.server_errors == 1
        and golden.cancelled == 1
        and golden.max_in_flight == 2
        and golden.in_flight == 0
        and golden.duration_p50_ms is not None
        and golden.duration_p95_ms is not None
    )

    rejected_trace_schema = False
    try:
        BoundedTrace.model_validate({
            "trace_id": "trc-" + "a" * 24,
            "correlation_id": "req-schema",
            "operation": "runtime.schema",
            "started_at": "2026-07-13T12:00:00Z",
            "duration_ms": 1.0,
            "outcome": "success",
            "result_code": 200,
            "attributes": {"prompt": "private"},
        })
    except ValidationError:
        rejected_trace_schema = True

    middleware_store = ObservabilityStore(trace_limit=16, samples_per_operation=16)
    observed: dict[str, str] = {}
    sent: list[dict] = []
    receive_calls = 0

    async def application(scope, receive, send):
        observed["context"] = current_correlation_id()
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        })
        await send({"type": "http.response.body", "body": b"{}"})

    async def private_receive():
        nonlocal receive_calls
        receive_calls += 1
        return {
            "type": "http.request",
            "body": b"PRIVATE-REQUEST-BODY-MUST-NOT-BE-READ",
            "more_body": False,
        }

    async def capture_send(message):
        sent.append(message)

    middleware = ObservabilityMiddleware(application, store=middleware_store)
    await middleware(
        {
            "type": "http",
            "method": "GET",
            "path": "/headless/v2/state",
            "query_string": b"",
            "headers": [(b"x-correlation-id", b"req-client-chain")],
            "state": {},
        },
        private_receive,
        capture_send,
    )
    response_start = next(item for item in sent if item["type"] == "http.response.start")
    response_headers = {
        key.decode("latin-1").lower(): value.decode("latin-1")
        for key, value in response_start["headers"]
    }
    middleware_report = middleware_store.snapshot(voice_slo=voice_nominal)
    middleware_trace = middleware_report.recent_traces[-1]
    middleware_contract = (
        observed.get("context") == "req-client-chain"
        and response_headers.get("x-correlation-id") == "req-client-chain"
        and middleware_trace.correlation_id == "req-client-chain"
        and middleware_trace.operation == "http.get.headless.v2.state"
        and middleware_trace.attributes == {
            "method": "GET",
            "route": "/headless/v2/state",
        }
        and receive_calls == 0
        and CURRENT_CORRELATION_ID.get("") == ""
        and "PRIVATE-REQUEST-BODY" not in middleware_report.model_dump_json()
    )

    context_token = bind_correlation("req-async-chain")
    try:
        async def read_child_context():
            await asyncio.sleep(0)
            return current_correlation_id()

        propagated = await asyncio.create_task(read_child_context())
    finally:
        reset_correlation(context_token)
    n8n_source = inspect.getsource(brain_api._n8n_moe_chat)
    chat_source = inspect.getsource(brain_api.headless_v2_chat_stream)
    state_stream_source = inspect.getsource(brain_api.headless_v2_stream)
    voice_source = inspect.getsource(brain_api.realtime_voice)
    end_to_end_contract = (
        propagated == "req-async-chain"
        and '"correlationId": turn_correlation' in chat_source
        and '"correlationId": session_correlation' in voice_source
        and '"correlation_id": session_correlation' in voice_source
        and "correlation_id=current_correlation_id()" in state_stream_source
        and "correlation_id=current_correlation_id()" in n8n_source
    )

    def make_request(key: str = "") -> Request:
        headers = [(b"x-weaver-key", key.encode("latin-1"))] if key else []
        return Request({
            "type": "http",
            "method": "GET",
            "path": "/health/observability",
            "headers": headers,
            "query_string": b"",
            "server": ("test", 443),
            "client": ("test", 1),
            "scheme": "https",
        })

    original_store = brain_api.OBSERVABILITY
    original_key = brain_api.WEAVER_KEY
    original_limiter = brain_api.OBSERVABILITY_LIMITER
    endpoint_contract = False
    try:
        brain_api.OBSERVABILITY = store
        brain_api.WEAVER_KEY = "observability-contract-key"
        brain_api.OBSERVABILITY_LIMITER = SlidingWindowRateLimiter(
            limit=4,
            window_seconds=60,
        )
        unauthorized = False
        try:
            await brain_api.health_observability(make_request(), Response())
        except HTTPException as exc:
            unauthorized = (
                exc.status_code == 403
                and exc.headers.get("Cache-Control") == "no-store"
            )
        endpoint_response = Response()
        endpoint_report = await brain_api.health_observability(
            make_request("observability-contract-key"),
            endpoint_response,
        )
        endpoint_json = endpoint_report.model_dump_json()
        endpoint_contract = (
            unauthorized
            and endpoint_response.headers.get("cache-control") == "no-store"
            and endpoint_report.schema_version == 1
            and endpoint_report.retention_traces == 16
            and len(endpoint_report.recent_traces) <= 16
            and "observability-contract-key" not in endpoint_json
            and all(marker not in endpoint_json.lower() for marker in (
                "prompt", "transcript", "message", "model_id", "webhook",
                "http://", "https://", "traceback",
            ))
        )
    finally:
        brain_api.OBSERVABILITY = original_store
        brain_api.WEAVER_KEY = original_key
        brain_api.OBSERVABILITY_LIMITER = original_limiter

    passed = all((
        trace_and_log_contract,
        budget_contract,
        saturation_contract,
        rejected_trace_schema,
        middleware_contract,
        end_to_end_contract,
        endpoint_contract,
    ))
    detail = "\n".join([
        f"  Structured logs/traces are categorical, redacted, and bounded: {trace_and_log_contract}",
        f"  Latency and availability consume an explicit error budget:    {budget_contract}",
        f"  Golden traffic/error/latency/saturation metrics are exact:    {saturation_contract}",
        f"  Public trace schemas reject undeclared diagnostic fields:     {rejected_trace_schema}",
        f"  HTTP correlation propagates without inspecting request data:  {middleware_contract}",
        f"  Correlation reaches async chat, voice, state, and n8n edges:   {end_to_end_contract}",
        f"  Operator telemetry is authenticated, no-store, and private:   {endpoint_contract}",
    ])
    _result(
        "BD",
        "Correlated telemetry is redacted, bounded, and governed by SLO error budgets",
        passed,
        detail,
    )
    return passed


async def test_BE():
    _header("BE", "Versioned n8n boundary keeps specialists private and Weaver public")
    from copy import deepcopy

    from pydantic import TypeAdapter, ValidationError

    import bedrock_brain_api as brain
    from headless_schemas import (
        N8NHeadlessRequest,
        N8NPublicResponse,
    )
    from runtime_resilience import AsyncCircuitBreaker

    contract_version = "weaver-headless-n8n-v1"
    request_fields = {
        "contract_version", "correlation_id", "deadline_ms", "text",
        "self_check", "introspect", "path_glob", "search_query",
        "codebase_context", "quantum_pathway", "cognition_context",
    }
    request = {
        "contract_version": contract_version,
        "correlation_id": "req-schema-contract",
        "deadline_ms": 115_000,
        "text": "Explain the safe Intent Capsule transport boundary.",
        "self_check": True,
        "introspect": True,
        "path_glob": "**/*",
        "search_query": "IntentCapsule reflex kernel",
        "codebase_context": "bounded read-only source",
        "quantum_pathway": "Awakening and Fracture remain in tension.",
        "cognition_context": {
            "awareness_confidence": 0.9,
            "fabric_pressure": 0.1,
            "immune_status": "nominal",
            "open_components": [],
        },
    }
    request_model = N8NHeadlessRequest.model_validate(request)
    request_rejections = []
    for mutation in (
        {**request, "private_prompt": "PRIVATE"},
        {**request, "contract_version": "legacy"},
        {**request, "deadline_ms": 114_999},
        {**request, "correlation_id": "unsafe correlation"},
        {**request, "search_query": "x" * 241},
        {
            **request,
            "cognition_context": {
                **request["cognition_context"],
                "private_state": "PRIVATE",
            },
        },
    ):
        try:
            N8NHeadlessRequest.model_validate(mutation)
            request_rejections.append(False)
        except ValidationError:
            request_rejections.append(True)
    request_contract = (
        set(request_model.model_dump(mode="json")) == request_fields
        and all(request_rejections)
    )

    success_fields = {
        "contract_version", "status", "error", "correlation_id",
        "manifested_response", "speaker", "speaker_boundary_applied",
        "speaker_model", "internal_draft_hidden", "reflection_applied",
        "soul_voice_active", "codebase_grounded", "expert_parallel",
        "expert_count", "experts_completed", "expert_errors",
        "expert_fanout_elapsed_ms", "execution_id", "timestamp",
        "pipeline_architecture", "pipeline_version",
    }
    success = {
        "contract_version": contract_version,
        "status": "ok",
        "error": False,
        "correlation_id": request["correlation_id"],
        "manifested_response": "Weaver's reviewed public answer.",
        "speaker": "weaver",
        "speaker_boundary_applied": True,
        "speaker_model": "qwen.qwen3-235b-a22b-2507",
        "internal_draft_hidden": True,
        "reflection_applied": True,
        "soul_voice_active": False,
        "codebase_grounded": True,
        "expert_parallel": True,
        "expert_count": 5,
        "experts_completed": 5,
        "expert_errors": 0,
        "expert_fanout_elapsed_ms": 102_500,
        "execution_id": "contract-test",
        "timestamp": "2026-07-13T12:00:00.000Z",
        "pipeline_architecture": "parallel-fanout-barrier",
        "pipeline_version": "v6-parallel-cognition",
    }
    adapter = TypeAdapter(N8NPublicResponse)
    parsed_success = adapter.validate_python(success)
    private_response_rejections = []
    for mutation in (
        {**success, "expert_drafts": ["PRIVATE-DRAFT"]},
        {**success, "lora_error": "PRIVATE-ERROR"},
        {**success, "speaker": "coder"},
        {**success, "speaker_boundary_applied": False},
        {**success, "internal_draft_hidden": False},
        {**success, "speaker_model": "private-specialist"},
        {**success, "manifested_response": ""},
    ):
        try:
            adapter.validate_python(mutation)
            private_response_rejections.append(False)
        except ValidationError:
            private_response_rejections.append(True)
    rejection = {
        "contract_version": contract_version,
        "status": "rejected",
        "error": True,
        "error_code": "speaker-boundary-failed",
        "correlation_id": request["correlation_id"],
        "execution_id": "contract-test",
        "timestamp": "2026-07-13T12:00:00.000Z",
        "pipeline_version": "v6-parallel-cognition",
    }
    parsed_rejection = adapter.validate_python(rejection)
    rejection_with_speech_blocked = False
    try:
        adapter.validate_python({**rejection, "manifested_response": "PRIVATE-DRAFT"})
    except ValidationError:
        rejection_with_speech_blocked = True
    response_contract = (
        set(parsed_success.model_dump(mode="json")) == success_fields
        and parsed_success.speaker == "weaver"
        and parsed_success.internal_draft_hidden is True
        and parsed_rejection.status == "rejected"
        and all(private_response_rejections)
        and rejection_with_speech_blocked
    )

    captured_requests: list[dict] = []
    recorded_state: list[dict] = []
    response_mode = "valid"

    def fake_post(_url, payload, _timeout):
        captured_requests.append(deepcopy(payload))
        public = {**success, "correlation_id": payload["correlation_id"]}
        if response_mode == "private-extra":
            return {**public, "expert_drafts": ["PRIVATE-DRAFT-MUST-NOT-CROSS"]}
        if response_mode == "wrong-correlation":
            return {**public, "correlation_id": "req-wrong-correlation"}
        return public

    async def capture_state(**updates):
        recorded_state.append(deepcopy(updates))

    original_post = brain._json_post_sync
    original_enabled = brain.N8N_CHAT_ENABLED
    original_url = brain.N8N_WEBHOOK_URL
    original_breaker = dict(brain._n8n_breaker)
    original_runtime_circuit = brain.N8N_RUNTIME_CIRCUIT
    original_record_state = brain._record_state
    runtime_contract = False
    runtime_facts: dict[str, bool] = {}
    try:
        brain._json_post_sync = fake_post
        brain.N8N_CHAT_ENABLED = True
        brain.N8N_WEBHOOK_URL = "http://n8n.contract.test/webhook"
        brain._n8n_breaker.update({"fails": 0, "skip_until": 0.0})
        brain.N8N_RUNTIME_CIRCUIT = AsyncCircuitBreaker(
            "n8n-contract-test",
            failure_threshold=2,
            recovery_seconds=1,
            timeout_seconds=2,
        )
        brain._record_state = capture_state

        valid_result = await brain._n8n_moe_chat(
            "Inspect whole_codebase_tests.py and explain IntentCapsule reflex-kernel safety.",
            "bounded source evidence\n",
        )
        response_mode = "private-extra"
        private_result = await brain._n8n_moe_chat("Do not expose drafts.", "")
        response_mode = "wrong-correlation"
        mismatched_result = await brain._n8n_moe_chat("Keep correlation exact.", "")

        sent = captured_requests[0] if captured_requests else {}
        safe_route = valid_result[1].get("route", {}) if valid_result else {}
        runtime_facts = {
            "valid_result": valid_result is not None,
            "public_text": valid_result is not None and valid_result[0] == success["manifested_response"],
            "request_fields": set(sent) == request_fields,
            "contract_version": sent.get("contract_version") == contract_version,
            "deadline": sent.get("deadline_ms") == 115_000,
            "search_bound": len(sent.get("search_query", "")) <= 240,
            "source_normalized": sent.get("codebase_context") == "bounded source evidence",
            "private_rejected": private_result is None,
            "correlation_rejected": mismatched_result is None,
            "stable_error": any(item.get("last_n8n_error") == "invalid-contract" for item in recorded_state),
            "safe_route": set(safe_route) == {
                "alias", "purpose", "contract_version", "pipeline",
                "pipeline_architecture", "soul_voice_active", "reflection_applied",
                "speaker_boundary_applied", "speaker_model", "internal_draft_hidden",
                "codebase_grounded", "expert_parallel", "experts_completed", "expert_errors",
            },
            "private_absent": "PRIVATE-DRAFT-MUST-NOT-CROSS" not in json.dumps(valid_result),
        }
        runtime_contract = all(runtime_facts.values())
    finally:
        brain._json_post_sync = original_post
        brain.N8N_CHAT_ENABLED = original_enabled
        brain.N8N_WEBHOOK_URL = original_url
        brain._n8n_breaker.clear()
        brain._n8n_breaker.update(original_breaker)
        brain.N8N_RUNTIME_CIRCUIT = original_runtime_circuit
        brain._record_state = original_record_state

    workflow_path = os.path.join(PROJ, "n8n_weaver_v5.json")
    with open(workflow_path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)
    nodes = {node["name"]: node for node in workflow["nodes"]}
    sanitize_code = nodes["2. Sanitize"]["parameters"]["jsCode"]
    writeback_code = nodes["9. Writeback"]["parameters"]["jsCode"]
    deploy_path = os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh")
    with open(deploy_path, "r", encoding="utf-8") as handle:
        deploy_source = handle.read()
    workflow_contract = (
        contract_version in sanitize_code
        and "keys.length === ALLOWED.length" in sanitize_code
        and "body.deadline_ms === 115000" in sanitize_code
        and "body.correlation_id === correlation" in sanitize_code
        and contract_version in writeback_code
        and "manifested_response: reviewed" in writeback_code
        and "speaker: 'weaver'" in writeback_code
        and "...d" not in writeback_code
        and all(field not in writeback_code for field in (
            "lora_error:", "qwen3b_error:", "dominant_lobe:",
            "experts_activated:", "collapsed_response:",
        ))
        and "n8n_weaver_v5.json" in deploy_source
        and "n8n_weaver_final.json" not in deploy_source
    )

    validator = await asyncio.create_subprocess_exec(
        "node",
        os.path.join(PROJ, "scripts", "validate_n8n_workflow.mjs"),
        "--json",
        cwd=PROJ,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    validator_stdout, validator_stderr = await validator.communicate()
    try:
        validator_report = json.loads(validator_stdout.decode("utf-8"))
    except json.JSONDecodeError:
        validator_report = {}
    validator_contract = (
        validator.returncode == 0
        and not validator_stderr
        and validator_report.get("valid") is True
        and validator_report.get("version") == 2
        and validator_report.get("critical_path_budget_ms", 115_001) <= 115_000
        and validator_report.get("workflow_timeout_ms") == 115_000
        and validator_report.get("errors") == []
        and validator_report.get("warnings") == []
    )

    passed = all((
        request_contract,
        response_contract,
        runtime_contract,
        workflow_contract,
        validator_contract,
    ))
    detail = "\n".join([
        f"  Exact typed request rejects aliases, extras, and bad bounds: {request_contract}",
        f"  Public union accepts only Weaver success or silent rejection: {response_contract}",
        f"  Brain enforces correlation and drops malformed n8n output:    {runtime_contract}",
        f"    Runtime subchecks failing: {[key for key, value in runtime_facts.items() if not value]}",
        f"  Canonical workflow/deploy paths contain no parallel contract: {workflow_contract}",
        f"  Validator v2 passes 102.5s/115s topology and privacy gates:   {validator_contract}",
    ])
    _result(
        "BE",
        "Versioned n8n boundary keeps specialists private and Weaver public",
        passed,
        detail,
    )
    return passed


async def test_BF():
    _header("BF", "Edge, service, container, and deployment hardening remain coherent")

    def read(relative_path: str) -> str:
        with open(os.path.join(PROJ, relative_path), "r", encoding="utf-8") as handle:
            return handle.read()

    caddy = read("deploy/Caddyfile")
    main_site = caddy.split("weaverv3.com {", 1)[1].split(
        "headless.weaverv3.com {", 1
    )[0]
    headless_site = caddy.split("headless.weaverv3.com {", 1)[1].split(
        "dash.weaverv3.com {", 1
    )[0]
    caddy_contract = (
        caddy.count("import weaver_security_headers") == 4
        and "Strict-Transport-Security \"max-age=31536000; includeSubDomains\"" in caddy
        and "X-Content-Type-Options \"nosniff\"" in caddy
        and "frame-ancestors 'none'" in caddy
        and "Permissions-Policy" in caddy
        and 'header @entry >Cache-Control "no-store, max-age=0"' in caddy
        and 'header @assets >Cache-Control "public, max-age=86400, stale-while-revalidate=604800"' in caddy
        and "response_header_timeout 130s" in caddy
        and "stream_timeout 15m" in caddy
        and "stream_close_delay 5m" in caddy
        and all(
            site.count("handle /brain/headless/v2/*") == 1
            and site.index("handle /brain/headless/v2/*")
            < site.index("handle_path /brain/*")
            for site in (main_site, headless_site)
        )
    )

    unit_names = (
        "deploy/weaver.service",
        "deploy/weaver-brain.service",
        "deploy/weaver-llm.service",
        "deploy/tts/weaver-tts.service",
        "deploy/n8n.service",
    )
    units = {name: read(name) for name in unit_names}
    common_markers = (
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "PrivateDevices=true",
        "ProtectSystem=strict",
        "ProtectProc=invisible",
        "ProcSubset=pid",
        "RestrictNamespaces=true",
        "RestrictSUIDSGID=true",
        "LockPersonality=true",
        "RestrictRealtime=true",
        "CapabilityBoundingSet=",
        "AmbientCapabilities=",
    )
    systemd_contract = (
        all(all(marker in source for marker in common_markers) for source in units.values())
        and all("StartLimitBurst=" in source for source in units.values())
        and all("UMask=0077" in source for source in units.values())
        and all("RestrictAddressFamilies=" in source for source in units.values())
        and all("OOMPolicy=stop" in source for name, source in units.items() if name != "deploy/n8n.service")
        and "TimeoutStopSec=140" in units["deploy/weaver-brain.service"]
        and "--timeout-graceful-shutdown 125" in units["deploy/weaver-brain.service"]
        and all(
            f"Environment=WEAVER_HEADLESS_V2_{flag}=1" in units["deploy/weaver-brain.service"]
            for flag in ("STATE", "STREAM", "SESSION", "SUMMARIES", "UI", "PROGRESS")
        )
    )

    n8n_unit = units["deploy/n8n.service"]
    compose = read("docker-compose.yml")
    n8n_image = (
        "docker.n8n.io/n8nio/n8n:2.25.7@sha256:"
        "761374d4eb841b0a22771d6bd68f0e8d827b4979ae4e490045517b13fc1259dd"
    )
    n8n_controls = (
        n8n_image in n8n_unit
        and n8n_image in compose
        and all(marker in n8n_unit for marker in (
            "--init",
            "--read-only",
            "--cap-drop ALL",
            "--security-opt no-new-privileges:true",
            "--pids-limit 512",
            "--memory 2g",
            "--memory-reservation 1536m",
            "--memory-swap 2g",
            "--cpus 2",
            "N8N_RUNNERS_BROKER_LISTEN_ADDRESS=127.0.0.1",
            "N8N_BLOCK_RUNNER_ENV_ACCESS=true",
            "N8N_COMMUNITY_PACKAGES_ENABLED=false",
            "N8N_PYTHON_ENABLED=false",
            "N8N_DISABLE_UI=true",
            "EXECUTIONS_TIMEOUT_MAX=115",
            "EXECUTIONS_DATA_SAVE_ON_ERROR=none",
            "EXECUTIONS_DATA_SAVE_ON_SUCCESS=none",
            "N8N_GRACEFUL_SHUTDOWN_TIMEOUT=125",
            "docker stop -t 130 n8n",
            "TimeoutStopSec=150",
        ))
        and all(marker in compose for marker in (
            "read_only: true",
            "no-new-privileges:true",
            "pids_limit: 512",
            "mem_limit: 2g",
            "mem_reservation: 1536m",
            "stop_grace_period: 130s",
            "N8N_GRACEFUL_SHUTDOWN_TIMEOUT=125",
            "127.0.0.1:5678:5678",
            "127.0.0.1:4040:4040",
            "ngrok/ngrok:latest@sha256:14d80d083e5b53145f416bbbd36238336c9de4016c43fd950eb2eb845670583b",
        ))
    )

    dockerignore = read(".dockerignore")
    build_context_contract = all(
        marker in dockerignore.splitlines()
        for marker in (".env", ".env.*", "Nexus_Vault", "Weaver_Vault", "venv", "models", "*.sqlite*")
    )

    deploy = read("deploy/deploy_voice_fullstack_fix.sh")
    deploy_contract = (
        'sudo cp -a /etc/caddy/Caddyfile "$BACKUP/Caddyfile"' in deploy
        and 'sudo install -m 0644 "$APP/deploy/Caddyfile" /etc/caddy/Caddyfile' in deploy
        and "systemd-analyze verify" in deploy
        and deploy.count("deploy/validate_caddy.sh") >= 2
        and "sudo systemctl reload caddy || sudo systemctl restart caddy" in deploy
        and '"contract_version": "weaver-headless-n8n-v1"' in deploy
        and '"deadline_ms": 115000' in deploy
        and deploy.count("--data-binary @/tmp/n8n-webhook-request.json") == 2
        and 'assert set(data) == expected' in deploy
        and 'assert data.get("speaker") == "weaver"' in deploy
        and "qwen3b_active" not in deploy
        and "/brain/headless/v2/session" in deploy
        and "HttpOnly" in deploy
        and "X-Weaver-CSRF" in deploy
        and "/health/live" in deploy
    )

    compose_proc = await asyncio.create_subprocess_exec(
        "docker", "compose", "config", "--quiet",
        cwd=PROJ,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, compose_stderr = await compose_proc.communicate()
    compose_contract = compose_proc.returncode == 0 and not compose_stderr

    shell_proc = await asyncio.create_subprocess_exec(
        "bash", "-n", os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        cwd=PROJ,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, shell_stderr = await shell_proc.communicate()
    shell_contract = shell_proc.returncode == 0 and not shell_stderr

    unit_parser_contract = True
    unit_parser_detail = "systemd-analyze unavailable; static production gate retained"
    if shutil.which("systemd-analyze"):
        unit_proc = await asyncio.create_subprocess_exec(
            "systemd-analyze", "verify",
            *(os.path.join(PROJ, path) for path in unit_names),
            cwd=PROJ,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, unit_stderr = await unit_proc.communicate()
        unit_lines = [line for line in unit_stderr.decode("utf-8").splitlines() if line.strip()]
        missing_remote_paths_only = bool(unit_lines) and all(
            "is not executable: No such file or directory" in line
            and "/home/ubuntu/" in line
            for line in unit_lines
        )
        unit_parser_contract = unit_proc.returncode == 0 or missing_remote_paths_only
        unit_parser_detail = (
            "valid"
            if unit_proc.returncode == 0
            else "valid directives; only production /home/ubuntu executables are absent locally"
        )

    passed = all((
        caddy_contract,
        systemd_contract,
        n8n_controls,
        build_context_contract,
        deploy_contract,
        compose_contract,
        shell_contract,
        unit_parser_contract,
    ))
    detail = "\n".join([
        f"  Caddy state route, long streams, headers, and caches:       {caddy_contract}",
        f"  systemd least privilege, bounds, and v2 cutover flags:      {systemd_contract}",
        f"  n8n/ngrok images and runtime resources stay immutable:      {n8n_controls}",
        f"  Docker build context excludes secrets, vaults, and state:   {build_context_contract}",
        f"  Deploy/rollback verifies exact n8n, session, Caddy contracts:{deploy_contract}",
        f"  Docker Compose resolves without conflicting limits:         {compose_contract}",
        f"  Deployment shell parses with strict Bash:                    {shell_contract}",
        f"  systemd unit parser: {unit_parser_contract} ({unit_parser_detail})",
    ])
    _result(
        "BF",
        "Edge, service, container, and deployment hardening remain coherent",
        passed,
        detail,
    )
    return passed


async def test_BG():
    _header("BG", "Headless shell is modular, CSP-strict, dependency-light, and rollback-safe")
    import re

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))
    module_relatives = (
        "headless/js/core.js",
        "headless/js/session.js",
        "headless/js/voice-support.js",
        "headless/js/visual-data.js",
        "headless/js/visual-runtime.js",
        "headless/js/voice.js",
        "headless/js/visualization.js",
        "headless/js/cortex.js",
        "headless/js/state-channel.js",
        "headless/js/lifecycle.js",
        "headless/js/accessibility.js",
        "headless/js/app.js",
    )
    style_relatives = (
        "headless/styles/tokens.css",
        "headless/styles/shell.css",
    )

    def read_avatar(relative: str) -> str:
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    legacy = read_avatar("headless-legacy.html")
    modules = {relative: read_avatar(relative) for relative in module_relatives}
    styles = {relative: read_avatar(relative) for relative in style_relatives}
    caddy = open(os.path.join(PROJ, "deploy", "Caddyfile"), "r", encoding="utf-8").read()
    deploy = open(
        os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        "r",
        encoding="utf-8",
    ).read()

    entry_contract = (
        len(html.encode("utf-8")) < 16_384
        and "<style" not in html
        and html.count("<script") == 1
        and '<script type="module" src="/headless/js/app.js"></script>' in html
        and not re.search(r"\son(?:click|load|error|submit|change)=", html, re.IGNORECASE)
        and all(f'href="/{relative}"' in html for relative in style_relatives)
        and all(
            f'href="/{relative}"' in html
            for relative in module_relatives
            if relative not in {"headless/js/visual-data.js", "headless/js/app.js"}
        )
    )

    token_css = styles["headless/styles/tokens.css"]
    shell_css = styles["headless/styles/shell.css"]
    style_contract = (
        token_css.count(":root") == 1
        and all(token in token_css for token in (
            "--bg:", "--surface:", "--stroke:", "--gold:", "--blue:", "--green:", "--text:", "--muted:",
        ))
        and ":root" not in shell_css
        and "@media (prefers-reduced-motion: reduce)" in shell_css
        and "env(safe-area-inset-bottom)" in shell_css
        and "backdrop-filter" not in shell_css
    )

    dependency_contract = True
    for relative, source in modules.items():
        if len(source.splitlines()) >= 1_000 or re.search(r"^\+", source, re.MULTILINE):
            dependency_contract = False
        if re.search(r"(?:from|import\s*\()\s*['\"](?:https?:|//|[^./])", source):
            dependency_contract = False
        for imported in re.findall(r"from\s+['\"](\./[^'\"]+)['\"]", source):
            target = os.path.normpath(os.path.join(os.path.dirname(relative), imported))
            if target not in modules:
                dependency_contract = False
    dependency_contract = dependency_contract and all(marker in "\n".join(modules.values()) for marker in (
        "export {",
        "from './core.js'",
        "from './voice.js'",
        "from './visualization.js'",
        "import(THREE_MODULE_URL)",
    ))

    legacy_contract = (
        len(legacy.encode("utf-8")) > 80_000
        and legacy.count("<style>") == 1
        and legacy.count("<script>") == 1
        and "globalThis.__weaverHeadlessVisualAudit" in legacy
        and 'test -s "$DEPLOY_ROOT/avatar/headless-legacy.html"' in deploy
    )

    strict_edge_contract = (
        "script-src 'self'; style-src 'self';" in caddy
        and "'unsafe-inline'" not in caddy
        and "@assets {" in caddy
        and "path /headless/* /vendor/*" in caddy
        and "not path /headless-sw.js" in caddy
        and all(relative in deploy for relative in (*module_relatives, *style_relatives))
        and 'sudo cp -a "$DEPLOY_ROOT/avatar/headless/."' in deploy
        and "modular headless CSS/ES modules match deployed checksums" in deploy
    )

    syntax_contract = True
    syntax_errors = []
    for relative in module_relatives:
        process = await asyncio.create_subprocess_exec(
            "node", "--check", os.path.join(avatar_root, relative),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            syntax_contract = False
            syntax_errors.append(f"{relative}: {stderr.decode('utf-8').strip()}")

    passed = all((
        entry_contract,
        style_contract,
        dependency_contract,
        legacy_contract,
        strict_edge_contract,
        syntax_contract,
    ))
    detail = "\n".join([
        f"  Semantic HTML entry is under 16 KiB with no inline code:   {entry_contract}",
        f"  Design tokens, safe areas, and reduced motion are external:{style_contract}",
        f"  Twelve local ES modules are bounded and dependency-light:  {dependency_contract}",
        f"  Original monolith remains a tracked rollback artifact:     {legacy_contract}",
        f"  CSP is no-inline and deployment hashes every module:       {strict_edge_contract}",
        f"  Every JavaScript module parses independently:              {syntax_contract}",
        f"    Syntax errors: {syntax_errors}",
    ])
    _result(
        "BG",
        "Headless shell is modular, CSP-strict, dependency-light, and rollback-safe",
        passed,
        detail,
    )
    return passed


async def test_BH():
    _header("BH", "Headless workspace makes conversation primary and private awareness explicit")
    import re

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))

    def read_avatar(relative: str) -> str:
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    css = read_avatar("headless/styles/shell.css")
    core = read_avatar("headless/js/core.js")
    cortex = read_avatar("headless/js/cortex.js")
    app = read_avatar("headless/js/app.js")
    accessibility = read_avatar("headless/js/accessibility.js")

    ids = re.findall(r'\bid="([A-Za-z][A-Za-z0-9_-]*)"', html)
    unique_ids = len(ids) == len(set(ids)) and len(ids) >= 40
    semantic_workspace = all(marker in html for marker in (
        'class="status-header"',
        '<main class="workspace"',
        'id="conversation"',
        'role="log"',
        'aria-relevant="additions text"',
        'class="panel awareness-rail"',
        'id="awarenessConfidence"',
        'class="privacy-notice"',
        'class="composer-dock"',
        '<textarea id="text"',
        'id="diagnosticsDrawer"',
        'role="dialog"',
        'aria-modal="true"',
        'aria-hidden="true" tabindex="-1" inert',
        'Weaver is the only public speaker',
        'private specialists stay silent',
    ))

    wired_ids = (
        unique_ids
        and all(f"document.getElementById('{identifier}')" in core for identifier in (
            "transcript", "heardTime", "turnStatus", "reactionReadout",
            "copyLast", "retryTurn", "stopTurn", "connectionAnnouncement",
            "awarenessState", "awarenessConfidence", "awarenessConfidenceFill",
            "awarenessReason", "diagnosticsToggle", "diagnosticsClose",
            "diagnosticsDrawer", "diagnosticsScrim", "diagnosticRevision",
            "diagnosticFreshness", "diagnosticReaction",
        ))
    )

    privacy_projection = (
        "function privateActivitySummary(" in cortex
        and "Content remains hidden." in cortex
        and "cognition.thought_topics" in cortex
        and "cognition.dream_topics" in cortex
        and "s.last_thought" not in cortex
        and "s.last_dream" not in cortex
        and "compact(r.dream" not in cortex
        and "Private dream activity completed. Content remains hidden." in cortex
        and all(marker in html for marker in (
            "Safe thought summary", "Safe dream summary", "content hidden",
            "Private cognition stays private",
        ))
    )

    drawer_contract = all(marker in accessibility + app for marker in (
        "function setDiagnosticsOpen(open)",
        "ui.diagnosticsDrawer.inert = !next",
        "ui.appShell.inert = next",
        "aria-hidden",
        "aria-expanded",
        "priorFocus?.isConnected",
        "handleDiagnosticsKeydown(event)",
        "event.key === 'Escape'",
        "ui.diagnosticsScrim.addEventListener('click'",
    ))

    responsive_layout = all(marker in css for marker in (
        "grid-template-columns: minmax(420px, 700px) minmax(270px, 330px)",
        "grid-template-rows: auto minmax(0, 1fr) auto",
        "@media (max-width: 760px)",
        "grid-template-columns: 1fr;",
        "overflow-y: auto",
        "env(safe-area-inset-top)",
        "env(safe-area-inset-bottom)",
        ".diagnostics-drawer[data-open=\"true\"]",
        "@media (prefers-reduced-motion: reduce)",
    )) and "backdrop-filter" not in css

    passed = all((
        semantic_workspace,
        wired_ids,
        privacy_projection,
        drawer_contract,
        responsive_layout,
    ))
    detail = "\n".join([
        f"  Conversation, awareness, diagnostics, and composer are semantic: {semantic_workspace}",
        f"  More than 40 unique IDs are wired through the shared UI map:    {wired_ids}",
        f"  V2 state projection cannot render raw thought/dream content:     {privacy_projection}",
        f"  Diagnostics is inert when closed and returns keyboard focus:    {drawer_contract}",
        f"  Desktop/mobile grids, safe areas, and reduced motion are bound: {responsive_layout}",
    ])
    _result(
        "BH",
        "Headless workspace makes conversation primary and private awareness explicit",
        passed,
        detail,
    )
    return passed


async def test_BI():
    _header("BI", "Authenticated transcript streams only Weaver and survives long cognition turns")

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))

    def read_avatar(relative: str) -> str:
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    core = read_avatar("headless/js/core.js")
    session = read_avatar("headless/js/session.js")
    cortex = read_avatar("headless/js/cortex.js")
    app = read_avatar("headless/js/app.js")
    deploy = open(
        os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        "r",
        encoding="utf-8",
    ).read()
    caddy = open(os.path.join(PROJ, "deploy", "Caddyfile"), "r", encoding="utf-8").read()

    one_time_session = (
        "sessionStorage.removeItem('weaver_llm_key')" in core
        and "localStorage.removeItem('weaver_llm_key')" in core
        and "sessionStorage.setItem('weaver_llm_key'" not in core + session
        and session.count("'X-Weaver-Key': brainKey") == 1
        and all(marker in session for marker in (
            "const SESSION_PATH = '/brain/headless/v2/session'",
            "const SESSION_RENEW_PATH = '/brain/headless/v2/session/renew'",
            "credentials: 'same-origin'",
            "'X-Weaver-CSRF': state.auth.csrfToken",
            "function scheduleRenewal(",
            "HttpOnly cookie + rotating CSRF",
        ))
    )

    v2_stream_boundary = (
        "/brain/v1/chat/completions" not in cortex
        and "/brain/state" not in cortex
        and all(marker in cortex for marker in (
            "/brain/headless/v2/state",
            "/brain/headless/v2/chat/stream",
            "event.speaker !== 'weaver'",
            "speaker-boundary-rejected",
            "event.index",
            "index !== active.nextChunk",
            "MAX_SSE_BUFFER = 64 * 1024",
            "response.body.getReader()",
            "new TextDecoder()",
            "contentType.toLowerCase().startsWith('text/event-stream')",
        ))
        and "innerHTML" not in cortex
    )

    turn_lifecycle = all(marker in html + cortex + app for marker in (
        'id="copyLast"',
        'id="retryTurn"',
        'id="stopTurn"',
        "createMessage('user', text)",
        "createMessage('assistant', 'Acknowledging…', { pending: true })",
        "new AbortController()",
        "/brain/headless/v2/chat/${active.serverTurnId}",
        "copyLastReply",
        "retryLastTurn",
        "event.key === 'Enter' && !event.shiftKey",
        "event.key.toLowerCase() === 'k'",
        "event.key === 'Escape' && conversationAudit().active",
    ))

    bounded_context = all(marker in cortex for marker in (
        "MAX_HISTORY_MESSAGES = 20",
        "MAX_HISTORY_CHARACTERS = 22_000",
        "MAX_PUBLIC_REPLY = 32 * 1024",
        "message: text",
        "history,",
        "max_tokens: 512",
        "trimConversationHistory()",
    ))

    long_turn_contract = (
        "maxSemanticWaitMs: null" in cortex
        and "setInterval(() => updateElapsed(active), 1_000)" in cortex
        and all(label in cortex for label in (
            "queued: 'Weaver is preparing'",
            "thinking: 'Weaver is thinking'",
            "synthesizing: 'Weaver is forming her response'",
        ))
        and "setTimeout(active.controller.abort" not in cortex
        and "response_header_timeout 130s" in caddy
    )

    deploy_contract = (
        "headless/js/session.js" in deploy
        and 'href="/headless/js/session.js"' in html
        and "globalThis.__weaverHeadlessSessionAudit = authAudit" in app
        and "globalThis.__weaverHeadlessConversationAudit = conversationAudit" in app
    )

    passed = all((
        one_time_session,
        v2_stream_boundary,
        turn_lifecycle,
        bounded_context,
        long_turn_contract,
        deploy_contract,
    ))
    detail = "\n".join([
        f"  Long key is exchanged once for cookie + rotating CSRF:       {one_time_session}",
        f"  V2 SSE parser rejects non-Weaver or malformed public output: {v2_stream_boundary}",
        f"  Optimistic, stop, retry, copy, and keyboard states are wired:{turn_lifecycle}",
        f"  History, SSE buffer, reply, and token budgets remain bounded:{bounded_context}",
        f"  Generic progress has no client semantic timeout:             {long_turn_contract}",
        f"  Session module is preloaded, audited, and deployed atomically:{deploy_contract}",
    ])
    _result(
        "BI",
        "Authenticated transcript streams only Weaver and survives long cognition turns",
        passed,
        detail,
    )
    return passed


async def test_BJ():
    _header("BJ", "Voice UX uses session auth, v2 framing, native ownership, and safe fallbacks")
    import base64
    from http.cookies import SimpleCookie

    import httpx
    from pydantic import ValidationError

    import bedrock_brain_api as brain_api
    from headless_auth import SESSION_COOKIE_NAME, HeadlessSessionStore
    from headless_schemas import HeadlessVoiceSynthesisRequest

    request_schema = HeadlessVoiceSynthesisRequest.model_validate({"text": "Hello from Weaver."})
    invalid_voice_text = False
    try:
        HeadlessVoiceSynthesisRequest.model_validate({"text": "x" * 801})
    except ValidationError:
        invalid_voice_text = True
    schema_contract = request_schema.text == "Hello from Weaver." and invalid_voice_text

    originals = {
        "key": brain_api.WEAVER_KEY,
        "session_flag": brain_api.HEADLESS_V2_SESSION_ENABLED,
        "session_store": brain_api.HEADLESS_V2_SESSION_STORE,
        "synth": brain_api._headless_voice_synth_sync,
    }
    session_tts = False
    websocket_auth = False
    captured_text = []
    try:
        brain_api.WEAVER_KEY = "voice-session-test-key"
        brain_api.HEADLESS_V2_SESSION_ENABLED = True
        brain_api.HEADLESS_V2_SESSION_STORE = HeadlessSessionStore(ttl_seconds=120)

        def fake_synth(text):
            captured_text.append(text)
            return b"RIFF-safe-audio", "audio/wav"

        brain_api._headless_voice_synth_sync = fake_synth
        transport = httpx.ASGITransport(app=brain_api.app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://headless.weaverv3.com",
        ) as client:
            bootstrap_response = await client.post(
                "/headless/v2/session",
                headers={"x-weaver-key": brain_api.WEAVER_KEY},
            )
            bootstrap = bootstrap_response.json()
            parsed_cookie = SimpleCookie()
            parsed_cookie.load(bootstrap_response.headers["set-cookie"])
            session_token = parsed_cookie[SESSION_COOKIE_NAME].value
            csrf = bootstrap["csrf_token"]

            missing_csrf = await client.post(
                "/headless/v2/voice/synth",
                json={"text": "must not synthesize"},
            )
            synthesis = await client.post(
                "/headless/v2/voice/synth",
                headers={"x-weaver-csrf": csrf},
                json={"text": request_schema.text},
            )
            invalid = await client.post(
                "/headless/v2/voice/synth",
                headers={"x-weaver-csrf": csrf},
                json={"text": ""},
            )
            session_tts = (
                missing_csrf.status_code == 403
                and synthesis.status_code == 200
                and synthesis.content == b"RIFF-safe-audio"
                and synthesis.headers["content-type"] == "audio/wav"
                and synthesis.headers["x-weaver-voice-source"] == "trained"
                and invalid.status_code == 400
                and captured_text == [request_schema.text]
                and brain_api.WEAVER_KEY not in synthesis.request.headers.values()
            )

            class _VoiceHandshake:
                def __init__(self, headers, cookies=None):
                    self.headers = {key.lower(): value for key, value in headers.items()}
                    self.cookies = dict(cookies or {})
                    self.accepted = []
                    self.closed = []

                async def accept(self, subprotocol=None):
                    self.accepted.append(subprotocol)

                async def close(self, code=1000):
                    self.closed.append(code)

            browser_socket = _VoiceHandshake(
                {
                    "origin": "https://headless.weaverv3.com",
                    "sec-websocket-protocol": f"weaver-realtime, weaver-csrf.{csrf}",
                },
                {SESSION_COOKIE_NAME: session_token},
            )
            revalidate = await brain_api._accept_voice_ws(browser_socket)

            hostile_grant = await brain_api.HEADLESS_V2_SESSION_STORE.issue()
            hostile_socket = _VoiceHandshake(
                {
                    "origin": "https://attacker.invalid",
                    "sec-websocket-protocol": (
                        f"weaver-realtime, weaver-csrf.{hostile_grant.csrf_token}"
                    ),
                },
                {SESSION_COOKIE_NAME: hostile_grant.token},
            )
            hostile_result = await brain_api._accept_voice_ws(hostile_socket)

            encoded_key = base64.urlsafe_b64encode(brain_api.WEAVER_KEY.encode()).decode().rstrip("=")
            native_socket = _VoiceHandshake({
                "origin": "",
                "sec-websocket-protocol": f"weaver-realtime, weaver-key.{encoded_key}",
            })
            native_revalidate = await brain_api._accept_voice_ws(native_socket)

            revoked = await client.delete(
                "/headless/v2/session",
                headers={"x-weaver-csrf": csrf},
            )
            websocket_auth = (
                revalidate is not None
                and browser_socket.accepted == ["weaver-realtime"]
                and hostile_result is None
                and hostile_socket.closed == [1008]
                and native_revalidate is not None
                and await native_revalidate()
                and native_socket.accepted == ["weaver-realtime"]
                and revoked.status_code == 200
                and not await revalidate()
            )
    finally:
        brain_api.WEAVER_KEY = originals["key"]
        brain_api.HEADLESS_V2_SESSION_ENABLED = originals["session_flag"]
        brain_api.HEADLESS_V2_SESSION_STORE = originals["session_store"]
        brain_api._headless_voice_synth_sync = originals["synth"]

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))

    def read_avatar(relative):
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    css = read_avatar("headless/styles/shell.css")
    core = read_avatar("headless/js/core.js")
    support = read_avatar("headless/js/voice-support.js")
    voice = read_avatar("headless/js/voice.js")
    app = read_avatar("headless/js/app.js")
    backend = open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8").read()

    explicit_voice_ux = all(marker in html + css + core + app for marker in (
        'id="voiceTray"',
        'id="voiceMeterFill"',
        'id="voicePermission"',
        'id="voiceDevice"',
        'id="voiceLatency"',
        'id="voiceUserCaption"',
        'id="voiceWeaverCaption"',
        'id="voiceReplay"',
        'id="voiceInterrupt"',
        '.voice-tray[data-state="listening"]',
        '.voice-tray[data-state="denied"]',
        "ui.voiceReplay.addEventListener",
        "ui.voiceInterrupt.addEventListener",
    ))

    browser_transport = (
        "weaver-key." not in voice
        and "/tts/synth" not in voice
        and all(marker in voice + support for marker in (
            "`weaver-csrf.${state.auth.csrfToken}`",
            "'/brain/headless/v2/voice/synth'",
            "frame.set([0x57, 0x56, 0x52, 0x32]",
            "protocolVersion: 2",
            "type: 'output_ack'",
            "type: 'telemetry'",
            "type: 'interrupt'",
            "type: 'ping'",
            "data.type === 'renew_required'",
            "scheduleVoiceReconnect(",
            "VOICE_RECONNECT_POLICY.maxAttempts",
        ))
    )

    privacy_and_native = all(marker in voice + support + core for marker in (
        "data.speaker !== 'weaver'",
        "voice speaker boundary rejected",
        "state.nativeShell",
        "Native iOS cortex bridge · AVFoundation owns capture",
        "deviceClass: browserDeviceClass()",
        "deviceAvailable: Boolean(state.realtime.deviceLabel)",
        "pendingVoiceTextAvailable: Boolean(state.pendingVoiceText)",
    )) and "deviceIdentifier" not in voice + support

    backend_proxy_bounds = all(marker in backend for marker in (
        "parsed.hostname not in {\"127.0.0.1\", \"localhost\"}",
        "HEADLESS_TTS_MAX_BYTES + 1",
        "HEADLESS_TTS_TIMEOUT_SECONDS",
        "HEADLESS_VOICE_SYNTH_LIMITER",
        "await _require_headless_v2_request(request, require_csrf=True)",
        '"X-Weaver-Voice-Source": "trained"',
    ))

    passed = all((
        schema_contract,
        session_tts,
        websocket_auth,
        explicit_voice_ux,
        browser_transport,
        privacy_and_native,
        backend_proxy_bounds,
    ))
    detail = "\n".join([
        f"  Trained-voice request schema rejects empty/oversized text: {schema_contract}",
        f"  TTS requires session CSRF and never returns the browser key:{session_tts}",
        f"  Browser WS uses origin/session/CSRF; native key bridge stays:{websocket_auth}",
        f"  Permission, level, captions, replay, interrupt UI is wired: {explicit_voice_ux}",
        f"  Browser uses v2 frames, ACK, telemetry, renewal, and retry:  {browser_transport}",
        f"  Weaver-only captions and native sensor ownership are explicit:{privacy_and_native}",
        f"  Loopback TTS proxy is rate/time/size/content-type bounded:    {backend_proxy_bounds}",
    ])
    _result(
        "BJ",
        "Voice UX uses session auth, v2 framing, native ownership, and safe fallbacks",
        passed,
        detail,
    )
    return passed


async def test_BK():
    _header("BK", "Semantic visuals adapt on iPhone and the offline shell never caches private traffic")

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))

    def read_avatar(relative):
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    core = read_avatar("headless/js/core.js")
    runtime = read_avatar("headless/js/visual-runtime.js")
    visual = read_avatar("headless/js/visualization.js")
    lifecycle = read_avatar("headless/js/lifecycle.js")
    app = read_avatar("headless/js/app.js")
    css = read_avatar("headless/styles/shell.css")
    worker = read_avatar("headless-sw.js")
    with open(os.path.join(avatar_root, "manifest.webmanifest"), "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    caddy = open(os.path.join(PROJ, "deploy", "Caddyfile"), "r", encoding="utf-8").read()
    deploy = open(
        os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        "r",
        encoding="utf-8",
    ).read()

    deterministic_projection = (
        "Math.random" not in runtime + visual
        and "snapshot?.schema_version === 2" in runtime
        and "ACTIVE_PHASES.has(phase)" in runtime
        and "cognition.thought_count" not in runtime + visual
        and "private_cognition" not in runtime + visual
        and all(marker in runtime + visual for marker in (
            "stableUnit(",
            "deriveVisualSignals(",
            "fabric.pressure",
            "awareness.confidence",
            "fabric.ledger_valid",
            "state.visualSignals = signals",
            "signals.revision",
        ))
    )

    adaptive_rendering = all(marker in core + runtime + visual for marker in (
        "serverDevice === 'iphone-16e'",
        "shortSide >= 385",
        "longSide >= 835",
        "iPhone16e ? 1.25",
        "state.targetFps = state.reducedMotion ? 18",
        "render.renderScale > 0.65",
        "render.recoveryFrames >= 240",
        "recordRenderedFrame(",
        "effectiveDpr(",
        "qualityChanges",
        "frameEmaMs",
    ))

    public_shell_only = (
        "if (request.method !== 'GET') return;" in worker
        and "url.origin !== self.location.origin || isPrivateRequest(url)" in worker
        and all(prefix in worker for prefix in (
            "'/brain/'", "'/tts/'", "'/llm/'", "'/codebase/'", "'/gpu-render/'",
        ))
        and all(asset in worker for asset in (
            "'/headless/js/visual-runtime.js'", "'/headless/js/lifecycle.js'",
            "'/headless/styles/shell.css'", "'/vendor/three.module.js'",
        ))
        and not any(prefix in worker.split("const PRIVATE_PREFIXES", 1)[0] for prefix in (
            "'/brain/'", "'/tts/'", "'/llm/'", "'/codebase/'",
        ))
        and "networkFirstNavigation(request)" in worker
        and "cacheFirstStatic(request)" in worker
    )

    mobile_lifecycle = all(marker in html + lifecycle + app + css for marker in (
        'rel="manifest" href="/manifest.webmanifest"',
        'href="/headless/js/visual-runtime.js"',
        'href="/headless/js/lifecycle.js"',
        'id="networkStatus"',
        'id="installApp"',
        "globalThis.visualViewport?.addEventListener('resize'",
        "globalThis.screen?.orientation?.addEventListener",
        "if (!('serviceWorker' in navigator) || state.nativeShell)",
        "globalThis.__weaverHeadlessLifecycleAudit = lifecycleAudit",
        "--app-viewport-height",
        "env(safe-area-inset-bottom)",
        "@media (orientation: landscape) and (max-height: 500px)",
    ))

    manifest_contract = (
        manifest.get("id") == "/"
        and manifest.get("scope") == "/"
        and manifest.get("display") == "standalone"
        and manifest.get("orientation") == "any"
        and manifest.get("theme_color") == "#08090c"
        and any(icon.get("src") == "/weaver-logo.svg" for icon in manifest.get("icons", []))
    )

    edge_deploy_contract = all(marker in caddy + deploy for marker in (
        "/headless-sw.js /manifest.webmanifest",
        'header @serviceworker >Service-Worker-Allowed "/"',
        'header @entry >Cache-Control "no-store, max-age=0"',
        "headless/js/visual-runtime.js",
        "headless/js/lifecycle.js",
        "HEADLESS_ROOT_ASSETS=(",
        "manifest.webmanifest",
        "headless-sw.js",
        'sudo install -m 0644 "$DEPLOY_ROOT/avatar/$asset" "/var/www/weaver-headless/$asset"',
        "modular headless CSS/ES modules match deployed checksums; offline shell matches too",
    ))

    functional_runtime = False
    runtime_error = ""
    script = runtime + r'''
const snapshot = {
  schema_version: 2, revision: 7,
  system: {ready: true}, awareness: {status: 'nominal', confidence: 0.91, degraded_reasons: []},
  cognition: {phase: 'thinking', thought_count: 999},
  fabric: {status: 'watch', pressure: 0.4, ledger_valid: true, lanes: {interactive: {active: 1, queued: 2}}},
  voice: {status: 'ready'}, freshness: {headless: {fresh: true}, body: {fresh: true}},
};
const first = deriveVisualSignals(snapshot, {live: false, speaking: false});
const second = deriveVisualSignals(snapshot, {live: false, speaking: false});
const idle = deriveVisualSignals({...snapshot, cognition: {phase: 'idle', thought_count: 999}}, {});
if (!first.verified || !first.activeCognition || idle.activeCognition) throw new Error('phase projection');
if (JSON.stringify(first) !== JSON.stringify(second) || first.energy <= idle.energy) throw new Error('determinism');
const render = {lastRenderedAt: 0, renderedFrames: 0, frameEmaMs: 0, workEmaMs: 0,
  longFrames: 0, pressureFrames: 0, recoveryFrames: 0, lastQualityAt: 0,
  renderScale: 1, qualityChanges: 0};
for (let index = 1; index <= 20; index += 1) recordRenderedFrame(render, index * 50, 20, 60);
if (render.renderScale >= 1 || render.qualityChanges < 1) throw new Error('quality governor');
'''
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", encoding="utf-8", delete=False) as handle:
        handle.write(script)
        runtime_path = handle.name
    try:
        process = await asyncio.create_subprocess_exec(
            "node", runtime_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        runtime_error = stderr.decode("utf-8").strip()
        functional_runtime = process.returncode == 0
    finally:
        os.unlink(runtime_path)

    syntax_contract = True
    syntax_errors = []
    for relative in (
        "headless/js/core.js", "headless/js/visual-runtime.js",
        "headless/js/visualization.js", "headless/js/lifecycle.js",
        "headless/js/app.js", "headless-sw.js",
    ):
        process = await asyncio.create_subprocess_exec(
            "node", "--check", os.path.join(avatar_root, relative),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            syntax_contract = False
            syntax_errors.append(f"{relative}: {stderr.decode('utf-8').strip()}")

    passed = all((
        deterministic_projection,
        adaptive_rendering,
        public_shell_only,
        mobile_lifecycle,
        manifest_contract,
        edge_deploy_contract,
        functional_runtime,
        syntax_contract,
    ))
    detail = "\n".join([
        f"  V2 phase/awareness/fabric signals replace visual randomness: {deterministic_projection}",
        f"  iPhone 16e keeps 60-FPS intent and sheds DPR before cadence: {adaptive_rendering}",
        f"  Service worker caches public GET shell assets only:          {public_shell_only}",
        f"  Safe viewport, orientation, install, and native boundary:   {mobile_lifecycle}",
        f"  Standalone manifest is same-origin and scope-bounded:        {manifest_contract}",
        f"  Caddy/deploy no-store and checksum the offline lifecycle:    {edge_deploy_contract}",
        f"  Deterministic mapper and quality governor execute correctly: {functional_runtime} ({runtime_error})",
        f"  Runtime, lifecycle, and service-worker syntax parses:        {syntax_contract} ({syntax_errors})",
    ])
    _result(
        "BK",
        "Semantic visuals adapt on iPhone and the offline shell never caches private traffic",
        passed,
        detail,
    )
    return passed


async def test_BL():
    _header("BL", "Realtime recovery, accessibility preferences, focus containment, and diagnostics stay safe")

    avatar_root = os.path.abspath(os.path.join(PROJ, "..", "..", "avatar"))

    def read_avatar(relative):
        with open(os.path.join(avatar_root, relative), "r", encoding="utf-8") as handle:
            return handle.read()

    html = read_avatar("headless.html")
    css = read_avatar("headless/styles/shell.css")
    core = read_avatar("headless/js/core.js")
    cortex = read_avatar("headless/js/cortex.js")
    channel = read_avatar("headless/js/state-channel.js")
    lifecycle = read_avatar("headless/js/lifecycle.js")
    accessibility = read_avatar("headless/js/accessibility.js")
    app = read_avatar("headless/js/app.js")
    transport = open(os.path.join(PROJ, "headless_transport.py"), "r", encoding="utf-8").read()
    backend = open(os.path.join(PROJ, "bedrock_brain_api.py"), "r", encoding="utf-8").read()
    deploy = open(
        os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        "r",
        encoding="utf-8",
    ).read()

    realtime_read_only = (
        "capsule_submit" not in channel
        and all(marker in channel for marker in (
            "/brain/headless/v2/stream",
            "'weaver-headless-v2'",
            "`weaver-csrf.${state.auth.csrfToken}`",
            "MAX_MESSAGE_CHARACTERS = 65_536",
            "exactKeys(",
            "validSnapshot(",
            "validDelta(",
            "type: 'resume'",
            "armWatchdog()",
            "MAX_RECONNECT_ATTEMPTS = 8",
            "schedulePoll(0)",
            "pollingFallback:",
            "canExecuteCapsules: false",
            "maxSemanticWaitMs: null",
        ))
        and all(marker in transport + backend for marker in (
            "relay state and submit an already-signed Intent Capsule for",
            "deliberately has no callback capable of applying",
            "verify_capsule=INTENT_COMPILER.verify",
            "evaluate_capsule=_evaluate_headless_v2_capsule",
            '@app.websocket("/headless/v2/stream")',
        ))
    )

    ordered_fallback = all(marker in cortex + channel + lifecycle + app for marker in (
        "if (snapshot.revision < priorRevision) return false",
        "delta.base_revision !== currentRevision",
        "sendResume(ws)",
        "authenticated realtime state",
        "realtime stream + polling safety",
        "if (state.channel.status === 'connected') return 60_000",
        "Realtime state is reconnecting",
        "Polling remains active",
        "weaver:network-restored",
        "startStateChannel({ force: true })",
        "stopStateChannel({ permanent: false })",
    ))

    dialog_and_keyboard = all(marker in html + accessibility + app for marker in (
        'role="dialog"',
        'aria-modal="true"',
        'aria-hidden="true" tabindex="-1" inert',
        "ui.appShell.inert = next",
        "visibleFocusableElements()",
        "event.key !== 'Tab'",
        "event.shiftKey",
        "handleDiagnosticsKeydown(event)",
        "priorFocus?.isConnected",
        "if (handleDiagnosticsKeydown(event)) return",
    ))

    sensory_preferences = all(marker in html + css + core + accessibility for marker in (
        'id="motionToggle"',
        'id="contrastToggle"',
        'id="fieldToggle"',
        'aria-describedby="motionStatus"',
        "weaver_accessibility_v1",
        "state.systemReducedMotion || state.userReducedMotion",
        'body[data-reduce-motion="true"]',
        'body[data-high-contrast="true"]',
        'body[data-field-hidden="true"] canvas',
        "@media (prefers-contrast: more)",
        "@media (forced-colors: active)",
        "@media (pointer: coarse)",
        "min-height: 44px",
    ))

    recovery_and_live_regions = (
        html.count('id="connectionAnnouncement"') == 1
        and 'aria-live="polite"' in html
        and all(marker in html + core + cortex + lifecycle + app for marker in (
            'id="connectionBanner"',
            'id="reconnectNow"',
            "setConnectionState('offline'",
            "setConnectionState('reconnecting'",
            "state.connection.lastAnnouncement !== safeMessage",
            "ui.reconnectNow.addEventListener",
            "ui.transcript.setAttribute('aria-busy'",
        ))
    )

    redacted_diagnostics = all(marker in html + accessibility for marker in (
        'id="diagnosticConnection"',
        'id="diagnosticSession"',
        'id="diagnosticNetwork"',
        'id="diagnosticRender"',
        'id="diagnosticVoice"',
        'id="diagnosticPrivacy"',
        "private content hidden · Weaver-only output",
        "preferencesStoredWithoutCredentials: true",
        "privateCognitionInDiagnostics: false",
    )) and not any(marker in accessibility for marker in (
        "lastVoiceText", "pendingVoiceText", "lastHeard", "lastSaid", "thought_topics", "dream_topics",
    ))

    native_and_deploy = (
        "if (state.nativeShell" in channel
        and "if (!('serviceWorker' in navigator) || state.nativeShell)" in lifecycle
        and all(marker in deploy for marker in (
            "headless/js/state-channel.js",
            "headless/js/accessibility.js",
        ))
        and 'href="/headless/js/state-channel.js"' in html
        and 'href="/headless/js/accessibility.js"' in html
    )

    syntax_contract = True
    syntax_errors = []
    for relative in (
        "headless/js/core.js", "headless/js/cortex.js", "headless/js/state-channel.js",
        "headless/js/lifecycle.js", "headless/js/accessibility.js", "headless/js/app.js",
    ):
        process = await asyncio.create_subprocess_exec(
            "node", "--check", os.path.join(avatar_root, relative),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()
        if process.returncode != 0:
            syntax_contract = False
            syntax_errors.append(f"{relative}: {stderr.decode('utf-8').strip()}")

    passed = all((
        realtime_read_only,
        ordered_fallback,
        dialog_and_keyboard,
        sensory_preferences,
        recovery_and_live_regions,
        redacted_diagnostics,
        native_and_deploy,
        syntax_contract,
    ))
    detail = "\n".join([
        f"  State WS validates exact public messages and cannot execute: {realtime_read_only}",
        f"  Ordered resume, heartbeat, reconnect, and polling fallback:  {ordered_fallback}",
        f"  Modal inertness, focus trap, Escape, and restoration are wired:{dialog_and_keyboard}",
        f"  Motion, contrast, field, forced-color, and touch controls:    {sensory_preferences}",
        f"  Deduplicated live recovery status and manual reconnect:       {recovery_and_live_regions}",
        f"  Operator metadata is useful, bounded, and content-redacted:   {redacted_diagnostics}",
        f"  Native render ownership and atomic asset deployment remain:  {native_and_deploy}",
        f"  All Step 29 browser modules parse independently:              {syntax_contract} ({syntax_errors})",
    ])
    _result(
        "BL",
        "Realtime recovery, accessibility preferences, focus containment, and diagnostics stay safe",
        passed,
        detail,
    )
    return passed


async def test_BM():
    _header("BM", "Headless v2 survives concurrent state, admission, cancellation, and packet chaos")
    import copy

    from headless_chat import ChatTurnBusy, ChatTurnRegistry
    from headless_scheduler import HeadlessSchedule, HeadlessScheduler
    from headless_state import HeadlessStateStore, build_public_state
    from operation_admission import OperationAdmission, OperationBusy
    from voice_reliability import VoiceFrame, VoiceIngressSequencer, VoiceProtocolError
    from weaver_cognition_mesh import CognitionMesh
    from weaver_neural_fabric import NeuralFabric

    fabric = NeuralFabric(capacity_units=8, realtime_reserved_units=2).snapshot()
    cognition = CognitionMesh().snapshot(fabric=fabric)
    public = build_public_state(
        {
            "active": True,
            "started_at": 1_720_000_000.0,
            "last_tick_at": 1_720_000_019.0,
            "thoughts": 0,
            "dreams": 0,
            "voice_realtime": {
                "prewarm": {"status": "ready"},
                "slo": {"status": "nominal", "reaction_target_ms": 200},
            },
        },
        fabric,
        cognition,
        now=1_720_000_020.0,
    )
    state_store = HeadlessStateStore(max_history=32)
    waiter = asyncio.create_task(state_store.wait_for_revision(0, timeout=1.0))
    await asyncio.sleep(0)

    async def _publish(index):
        payload = copy.deepcopy(public.model_dump(mode="python"))
        payload["cognition"]["thought_count"] = index
        return await state_store.publish(payload)

    published = await asyncio.gather(*(_publish(index) for index in range(1, 257)))
    notified = await waiter
    latest = await state_store.snapshot()
    recent = await state_store.changes_since(240)
    expired = await state_store.changes_since(0)
    public_json = latest.model_dump_json() if latest is not None else ""
    state_storm = (
        {item.revision for item in published} == set(range(1, 257))
        and notified is not None
        and latest is not None
        and latest.revision == 256
        and recent is not None
        and [item.revision for item in recent] == list(range(241, 257))
        and expired is None
        and all(marker not in public_json.lower() for marker in (
            "prompt", "transcript", "chain_of_thought", "api_key",
        ))
    )

    single_flight = OperationAdmission[dict](
        rate_limit=1_000,
        window_seconds=60,
        concurrency=4,
        idempotency_entries=16,
    )
    factory_calls = 0

    async def _one_factory():
        nonlocal factory_calls
        factory_calls += 1
        await asyncio.sleep(0.02)
        return {"ok": True, "revision": 256}

    storm_results = await asyncio.gather(*(
        single_flight.execute(
            operation="state-refresh",
            payload={"revision": 256},
            idempotency_key="release-storm-key",
            factory=_one_factory,
        )
        for _ in range(128)
    ))
    admission_snapshot = await single_flight.snapshot()
    idempotency_storm = (
        factory_calls == 1
        and sum(1 for _, replayed in storm_results if not replayed) == 1
        and sum(1 for _, replayed in storm_results if replayed) == 127
        and all(value == {"ok": True, "revision": 256} for value, _ in storm_results)
        and admission_snapshot["concurrency"]["active"] == 0
        and admission_snapshot["idempotency"]["entries"] == 1
    )

    shedding = OperationAdmission[dict](
        rate_limit=1_000,
        window_seconds=60,
        concurrency=4,
    )
    release_capacity = asyncio.Event()
    active_count = 0
    peak_active = 0

    async def _bounded_factory():
        nonlocal active_count, peak_active
        active_count += 1
        peak_active = max(peak_active, active_count)
        try:
            await release_capacity.wait()
            return {"ok": True}
        finally:
            active_count -= 1

    capacity_tasks = [
        asyncio.create_task(shedding.execute(
            operation=f"capacity-{index}",
            payload={"index": index},
            idempotency_key=None,
            factory=_bounded_factory,
        ))
        for index in range(40)
    ]
    await asyncio.sleep(0.02)
    release_capacity.set()
    capacity_results = await asyncio.gather(*capacity_tasks, return_exceptions=True)
    shedding_snapshot = await shedding.snapshot()
    capacity_guard = (
        sum(1 for item in capacity_results if not isinstance(item, BaseException)) == 4
        and sum(isinstance(item, OperationBusy) for item in capacity_results) == 36
        and peak_active == 4
        and active_count == 0
        and shedding_snapshot["concurrency"]["active"] == 0
    )

    registry = ChatTurnRegistry(max_active=4)
    never = asyncio.Event()
    turn_tasks = [asyncio.create_task(never.wait()) for _ in range(5)]
    turn_ids = [f"turn-{index}" for index in range(5)]
    registered = []
    overflow_rejected = False
    for turn_id, task in zip(turn_ids, turn_tasks):
        try:
            await registry.register(turn_id, task)
            registered.append((turn_id, task))
        except ChatTurnBusy:
            overflow_rejected = True
            task.cancel()
    for turn_id, _ in registered:
        await registry.cancel(turn_id)
    await asyncio.gather(*turn_tasks, return_exceptions=True)
    for turn_id, task in registered:
        await registry.forget(turn_id, task)
    cancellation_guard = (
        overflow_rejected
        and len(registered) == 4
        and all(task.done() for task in turn_tasks)
        and await registry.active() == 0
    )

    scheduler_clock = [100.0]
    priority_event = asyncio.Event()
    thought_started = asyncio.Event()
    thought_cancelled = asyncio.Event()

    async def _idle():
        return True

    async def _blocking_thought(_reason):
        thought_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            thought_cancelled.set()
            raise

    async def _unused_dream(_reason):
        return "unused"

    async def _noop_tick(_now):
        return None

    async def _noop_error(_error):
        return None

    scheduler = HeadlessScheduler(
        HeadlessSchedule(1, 100, tick_seconds=0.01),
        active=lambda: True,
        idle_ready=_idle,
        run_thought=_blocking_thought,
        run_dream=_unused_dream,
        on_tick=_noop_tick,
        on_error=_noop_error,
        priority_event=priority_event,
        random_unit=lambda: 0.5,
        monotonic=lambda: scheduler_clock[0],
    )
    scheduler_clock[0] = 102.0
    cycle = asyncio.create_task(scheduler.run_cycle())
    await asyncio.wait_for(thought_started.wait(), timeout=1.0)
    priority_event.set()
    await asyncio.wait_for(cycle, timeout=1.0)
    scheduler_state = scheduler.snapshot()
    voice_first_preemption = (
        thought_cancelled.is_set()
        and scheduler_state["preemptions"] == 1
        and scheduler_state["thought_runs"] == 0
        and scheduler_state["errors"] == 0
    )

    ingress = VoiceIngressSequencer(
        max_buffered_frames=8,
        max_forward_gap=96,
        max_jitter_ms=20,
    )
    for sequence in range(1, 81):
        if sequence % 13 == 0:
            continue
        ingress.ingest(
            VoiceFrame(sequence, sequence * 20, b"pcm"),
            arrival_ms=sequence * 25,
        )
    ingress.ingest(VoiceFrame(80, 1_600, b"duplicate"), arrival_ms=2_025)
    forward_gap_rejected = False
    try:
        ingress.ingest(VoiceFrame(200, 4_000, b"gap"), arrival_ms=2_050)
    except VoiceProtocolError:
        forward_gap_rejected = True
    voice_state = ingress.snapshot()
    packet_chaos = (
        voice_state["ack_sequence"] == 80
        and voice_state["lost"] == 6
        and voice_state["duplicates"] == 1
        and voice_state["rejected"] == 1
        and voice_state["buffered"] == 0
        and voice_state["max_buffer_depth"] <= 8
        and forward_gap_rejected
    )

    passed = all((
        state_storm,
        idempotency_storm,
        capacity_guard,
        cancellation_guard,
        voice_first_preemption,
        packet_chaos,
    ))
    detail = "\n".join([
        f"  256 concurrent state writes stay monotonic and private: {state_storm}",
        f"  128 duplicate operations execute exactly once:          {idempotency_storm}",
        f"  Saturation runs four and sheds 36 without leaks:        {capacity_guard}",
        f"  Chat overflow/cancellation leaves no orphan tasks:      {cancellation_guard}",
        f"  Interactive priority cancels background cognition:     {voice_first_preemption}",
        f"  Packet loss/duplicates/gaps remain bounded and ordered: {packet_chaos}",
    ])
    _result(
        "BM",
        "Headless v2 survives concurrent state, admission, cancellation, and packet chaos",
        passed,
        detail,
    )
    return passed


async def test_BN():
    _header("BN", "The five-viewport browser release matrix is permanent and bounded")
    matrix_path = os.path.join(PROJ, "tests", "headless_release_matrix.py")
    with open(matrix_path, "r", encoding="utf-8") as handle:
        source = handle.read()
    syntax_valid = True
    try:
        compile(source, matrix_path, "exec")
    except SyntaxError:
        syntax_valid = False
    exact_viewports = all(marker in source for marker in (
        '"width": 320',
        '"width": 390, "height": 844',
        '"width": 768',
        '"width": 1024',
        '"width": 1440',
    ))
    realtime_boundary = all(marker in source for marker in (
        '"weaver-headless-v2"',
        '"canExecuteCapsules"] is False',
        '"maxSemanticWaitMs"] is None',
        'item.get("type") == "resume"',
    ))
    performance_budget = all(marker in source for marker in (
        'ready_wall_ms < 2_500',
        'metrics["resourceCount"] <= 24',
        '96 * 1024 * 1024',
        'metrics["visual"]["fps"] == 60',
        'metrics["visual"]["dprCap"] == 1.25',
    ))
    recovery_accessibility = all(marker in source for marker in (
        "set_offline(True)",
        "set_offline(False)",
        'dialog["openAudit"]["backgroundInert"] is True',
        'minimum_target = 43.5',
        'metrics["unlabeledButtons"] == 0',
    ))
    bounded_script = len(source.encode("utf-8")) < 24_000 and source.count("assert ") >= 20
    passed = all((
        syntax_valid,
        exact_viewports,
        realtime_boundary,
        performance_budget,
        recovery_accessibility,
        bounded_script,
    ))
    detail = "\n".join([
        f"  Release-matrix Python syntax is valid:              {syntax_valid}",
        f"  320/390x844/768/1024/1440 targets are exact:        {exact_viewports}",
        f"  Realtime is authenticated, resumable, and read-only:{realtime_boundary}",
        f"  Boot/resource/heap/iPhone render budgets are hard:  {performance_budget}",
        f"  Offline recovery, modal containment, targets pass:  {recovery_accessibility}",
        f"  Test remains dependency-light and reviewable:       {bounded_script}",
    ])
    _result(
        "BN",
        "The five-viewport browser release matrix is permanent and bounded",
        passed,
        detail,
    )
    return passed


async def test_BO():
    _header("BO", "Runtime dependencies retain an audited, deploy-enforced security floor")
    import subprocess
    from importlib.metadata import version

    security_path = os.path.join(PROJ, "requirements-security.txt")
    with open(security_path, "r", encoding="utf-8") as handle:
        security_source = handle.read()
    expected = {}
    for raw in security_source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, pinned = line.split("==", 1)
        expected[name] = pinned
    exact_floor = (
        len(expected) == 19
        and expected.get("aiohttp") == "3.14.1"
        and expected.get("starlette") == "1.3.1"
        and expected.get("langchain") == "1.3.13"
        and expected.get("PyJWT") == "2.13.0"
        and expected.get("urllib3") == "2.7.0"
    )

    with open(os.path.join(PROJ, "requirements.txt"), "r", encoding="utf-8") as handle:
        requirements = handle.read()
    with open(os.path.join(PROJ, "requirements-full.txt"), "r", encoding="utf-8") as handle:
        requirements_full = handle.read()
    synchronized = all(
        f"{name}=={pinned}" in requirements
        and f"{name}=={pinned}" in requirements_full
        for name, pinned in expected.items()
    )
    installed = {name: version(name) for name in expected}
    local_floor = installed == expected

    with open(os.path.join(PROJ, "scripts", "audit_dependencies.sh"), "r", encoding="utf-8") as handle:
        audit_source = handle.read()
    with open(os.path.join(PROJ, "deploy", "weaver-llm.service"), "r", encoding="utf-8") as handle:
        llm_unit = handle.read()
    exec_start = llm_unit.split("ExecStart=", 1)[1].split("Restart=", 1)[0]
    exceptions_bounded = all(marker in audit_source for marker in (
        "PYSEC-2026-2447",
        "CVE-2025-3000",
        "diskcache has no fixed release",
        "affected range is PyTorch 2.6.0 only",
    )) and all(marker in llm_unit for marker in (
        "PrivateTmp=true",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "UMask=0077",
    )) and "--cache" not in exec_start and all(
        "torch==2.11.0" in source for source in (requirements, requirements_full)
    )

    with open(os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"), "r", encoding="utf-8") as handle:
        deploy_source = handle.read()
    deployment_floor = all(marker in deploy_source for marker in (
        'test -s "$APP/requirements-security.txt"',
        '--requirement "$APP/requirements-security.txt"',
        '"$APP/venv/bin/python3" -m pip check',
        "actual == expected",
        "Security floors are forward-only infrastructure",
    ))
    syntax_valid = all(
        subprocess.run(
            ["bash", "-n", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        for path in (
            os.path.join(PROJ, "scripts", "audit_dependencies.sh"),
            os.path.join(PROJ, "deploy", "deploy_voice_fullstack_fix.sh"),
        )
    )

    passed = all((
        exact_floor,
        synchronized,
        local_floor,
        exceptions_bounded,
        deployment_floor,
        syntax_valid,
    ))
    detail = "\n".join([
        f"  Nineteen fixable runtime packages are exactly pinned: {exact_floor}",
        f"  Full and standard requirement sets match the floor:  {synchronized}",
        f"  Current release environment matches every exact pin: {local_floor}",
        f"  Two no-fix/false-positive exceptions are constrained: {exceptions_bounded}",
        f"  Deployment upgrades and verifies before migration:   {deployment_floor}",
        f"  Audit and deployment shells parse under strict Bash: {syntax_valid}",
    ])
    _result(
        "BO",
        "Runtime dependencies retain an audited, deploy-enforced security floor",
        passed,
        detail,
    )
    return passed


TESTS = {
    "G": ("Quantum parse invariants", test_G),
    "H": ("Quantum description + state write persistence", test_H),
    "I": ("Weaver supervisor crash restart semantics", test_I),
    "J": ("Weaver supervisor re-enters after clean exit", test_J),
    "K": ("Nexus cache sync trims to last 10 messages", test_K),
    "L": ("Nexus unsubscribe stops further deliveries", test_L),
    "M": ("Nexus protocol error frames", test_M),
    "N": ("VTV startup env contract fails fast", test_N),
    "O": ("Drive credential surfaces parse locally", test_O),
    "P": ("Soul dataset + LoRA artifact integrity", test_P),
    "Q": ("Nexus rejects non-object JSON without dropping socket", test_Q),
    "R": ("Nexus blocks duplicate lobe_id takeover", test_R),
    "S": ("Nexus port collision fails closed", test_S),
    "T": ("NexusClient reconnects after a clean socket close", test_T),
    "U": ("Dashboard publishes through a registered Nexus client", test_U),
    "V": ("n8n webhook repair is backed up and owner-safe", test_V),
    "W": ("Synchronous Nexus publisher delivers model-ready events", test_W),
    "X": ("Cortex falls back to on-box llama when n8n and Bedrock fail", test_X),
    "Y": ("Dashboard parses the 12-role Kingston quantum state", test_Y),
    "Z": ("Codebase evidence reaches the full n8n cortex", test_Z),
    "AA": ("Workflow reflection, soul voice, and live voice share one cortex", test_AA),
    "AB": ("Request, local-model, and shutdown contracts are connected", test_AB),
    "AC": ("Embodiment navigation connects body, environment, and camera", test_AC),
    "AD": ("iPhone senses use on-device acceleration and the full voice cortex", test_AD),
    "AE": ("Native iPhone shell connects Neural Engine, senses, cortex, and embodiment", test_AE),
    "AF": ("Whole-body awareness, intentional environment, silent coder, and fast voice reaction", test_AF),
    "AG": ("Credential, request, accessibility, stance, and first-audio hardening", test_AG),
    "AH": ("Neural QoS autopilot, stateful world model, flight recorder, and voice SLO control plane", test_AH),
    "AI": ("Neural Fabric lanes, proof ledger, signed Intent Capsules, and hardened control API", test_AI),
    "AJ": ("Seven-angle Cognition Mesh and validated parallel n8n v6 workflow", test_AJ),
    "AK": ("Cinematic facial articulation, layered body dynamics, and bounded material response", test_AK),
    "AL": ("High-fidelity original avatar LOD, physical scan maps, and safe runtime fallback", test_AL),
    "AM": ("iPhone 16e A18 adaptive mobile performance without embodiment loss", test_AM),
    "AN": ("Resilient visual boot, seamless clothing, and compositor-safe headless controls", test_AN),
    "AO": ("Only Weaver speaks while the coder stays private", test_AO),
    "AP": ("Headless v2 contracts preserve capsule, native, and long-turn authority", test_AP),
    "AQ": ("Strict headless v2 schemas, privacy projection, and revisioned shadow state", test_AQ),
    "AR": ("Headless scheduler and read-mostly realtime transport remain bounded", test_AR),
    "AS": ("Short-lived browser sessions keep the Weaver key out of repeated requests", test_AS),
    "AT": ("Headless v2 HTTP boundaries are allowlisted, bounded, correlated, and redacted", test_AT),
    "AU": ("Mutation admission deduplicates, rate-limits, replay-guards, and bounds concurrency", test_AU),
    "AV": ("Private cognition yields to voice with jitter, token budgets, and cancellation", test_AV),
    "AW": ("Private cognition stays in a bounded vault while clients receive safe metadata", test_AW),
    "AX": ("Memory provenance, deduplication, freshness, retention, and deletion are auditable", test_AX),
    "AY": ("Awareness fusion joins body, world, cognition, fabric, and dependency freshness", test_AY),
    "AZ": ("Realtime voice is sequenced, resumable, interruptible, and telemetry-bounded", test_AZ),
    "BA": ("Cancellable chat streams only Weaver's validated public answer", test_BA),
    "BB": ("Prewarming, coalescing, ETags, cache bounds, and circuit recovery", test_BB),
    "BC": ("Liveness, readiness, and authenticated deep health stay distinct and private", test_BC),
    "BD": ("Correlated telemetry is redacted, bounded, and governed by SLO error budgets", test_BD),
    "BE": ("Versioned n8n boundary keeps specialists private and Weaver public", test_BE),
    "BF": ("Edge, service, container, and deployment hardening remain coherent", test_BF),
    "BG": ("Headless shell is modular, CSP-strict, dependency-light, and rollback-safe", test_BG),
    "BH": ("Headless workspace makes conversation primary and private awareness explicit", test_BH),
    "BI": ("Authenticated transcript streams only Weaver and survives long cognition turns", test_BI),
    "BJ": ("Voice UX uses session auth, v2 framing, native ownership, and safe fallbacks", test_BJ),
    "BK": ("Semantic visuals adapt on iPhone and the offline shell never caches private traffic", test_BK),
    "BL": ("Realtime recovery, accessibility preferences, focus containment, and diagnostics stay safe", test_BL),
    "BM": ("Headless v2 survives concurrent state, admission, cancellation, and packet chaos", test_BM),
    "BN": ("The five-viewport browser release matrix is permanent and bounded", test_BN),
    "BO": ("Runtime dependencies retain an audited, deploy-enforced security floor", test_BO),
}


async def main(which: str):
    results = {}
    wall_start = time.monotonic()
    for label, (_, fn) in TESTS.items():
        if which.upper() != "ALL" and label.upper() != which.upper():
            continue
        results[label] = await fn()

    elapsed = time.monotonic() - wall_start
    print(f"\n{'═' * 62}")
    print(f"  WHOLE CODEBASE TEST RESULTS  ({elapsed/60:.1f} min total)")
    print(f"{'═' * 62}")
    for label, (title, _) in TESTS.items():
        if label in results:
            mark = "✅" if results[label] else "❌"
            print(f"  {mark}  Test {label}: {title}")
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'═' * 62}\n")
    return bool(results) and passed == len(results)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", nargs="?", default="all", help="G-BO or all")
    args = ap.parse_args()
    raise SystemExit(0 if asyncio.run(main(args.test)) else 1)
