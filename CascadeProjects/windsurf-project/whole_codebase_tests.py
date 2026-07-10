#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import importlib
import json
import os
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
        async with websockets.connect("ws://localhost:9999") as pub:
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

        async with websockets.connect("ws://localhost:9999") as sub:
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
        async with websockets.connect("ws://localhost:9999") as pub, websockets.connect("ws://localhost:9999") as sub:
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
        async with websockets.connect("ws://localhost:9999") as ws:
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
        async with websockets.connect("ws://localhost:9999") as ws:
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
        async with websockets.connect("ws://localhost:9999") as original, websockets.connect("ws://localhost:9999") as intruder, websockets.connect("ws://localhost:9999") as pub:
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
        async with websockets.connect("ws://localhost:9999") as sub:
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
        async with websockets.connect("ws://localhost:9999") as sub:
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

    async def no_moe(_user_text):
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
    ap.add_argument("test", nargs="?", default="all", help="G-S or all")
    args = ap.parse_args()
    raise SystemExit(0 if asyncio.run(main(args.test)) else 1)
