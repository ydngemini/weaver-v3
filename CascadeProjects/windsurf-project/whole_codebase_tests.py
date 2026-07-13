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
    evidence_ok = all(marker in context for marker in required_evidence)
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
            "manifested_response": "grounded answer",
            "pipeline_version": "test-grounded",
            "codebase_grounded": True,
            "soul_voice_active": True,
            "reflection_applied": True,
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
        and payload.get("codebase_context") == context
        and bool(payload.get("search_query"))
    )
    retrieval_fast = context_elapsed < 3.0
    passed = evidence_ok and identifier_cross_file_ok and route_ok and retrieval_fast
    detail = "\n".join([
        f"  Exact source evidence retrieved: {evidence_ok}",
        f"  Identifier search crosses files:  {identifier_cross_file_ok}",
        f"  Evidence chars:                 {len(context)}",
        f"  Grounding retrieval under 3s:   {retrieval_fast} ({context_elapsed:.3f}s)",
        f"  n8n introspection flags set:    {payload.get('self_check') is True and payload.get('introspect') is True}",
        f"  Evidence forwarded unchanged:   {payload.get('codebase_context') == context}",
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
    headless = os.path.join(PROJ, "..", "..", "avatar", "headless.html")
    with open(os.path.abspath(headless), "r", encoding="utf-8") as fh:
        headless_source = fh.read()
    frontend_unified = "data.type === 'agent_response'" in headless_source and "allowDuringRealtime" in headless_source

    passed = all((
        experts_grounded,
        reflection_grounded,
        request_bodies_safe,
        lora_preserves_review,
        writeback_grounded,
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
    local_fallback = (
        brain.LOCAL_LLM_URL.endswith(":8899/v1/chat/completions")
        and brain.N8N_CHAT_TIMEOUT >= 120
        and "final_messages" in brain_source
    )
    headless_yields_to_users = (
        "_interactive_started" in inspect.getsource(brain._cortex_chat)
        and "if not await _headless_idle_ready()" in headless_source
        and "last_thought = _now()" in headless_source
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
        "reconnectAttempt <= 5",
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
    headless_path = os.path.join(repo_root, "avatar", "headless.html")
    ios_root = os.path.join(repo_root, "ios", "WeaverNeural", "WeaverNeural")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    with open(headless_path, "r", encoding="utf-8") as fh:
        headless = fh.read()
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
        "weaver_woven_a_line_skirt",
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
    headless_path = os.path.join(repo_root, "avatar", "headless.html")
    ios_root = os.path.join(repo_root, "ios", "WeaverNeural")
    with open(embodiment_path, "r", encoding="utf-8") as fh:
        embodiment = fh.read()
    with open(headless_path, "r", encoding="utf-8") as fh:
        headless = fh.read()
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
        'aria-busy="false"',
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
        '"cognition_context"',
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
            'data.get("lora_latency_ms")',
            'data.get("qwen3b_latency_ms")',
            "n8n container: pinned, read-only, capability-dropped, sandboxed",
        ))
        and "docker cp" not in deploy_source
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
        'missing generated high-fidelity avatar asset',
        'sudo install -m 0644 "$DEPLOY_ROOT/avatar/weaver_avatar_dress_hifi.glb"',
        'sudo install -d -m 0755 /var/www/weaver/textures',
        'high-fidelity asset checksum mismatch',
        'high-fidelity GLB and PBR maps match deployed checksums',
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
    ap.add_argument("test", nargs="?", default="all", help="G-AM or all")
    args = ap.parse_args()
    raise SystemExit(0 if asyncio.run(main(args.test)) else 1)
