#!/usr/bin/env python3
import argparse
import asyncio
import contextlib
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime

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

    bits1, active1, marg1 = qs.parse_counts({"0000001": 7, "0000000": 1})
    bits2, active2, marg2 = qs.parse_counts({"1000000": 5, "0000000": 1})
    bits3, active3, marg3 = qs.parse_counts({"0000000": 9})

    # Pentagon layout: q0=Awakening, q1=Resonance, q2=Echo, q3=Prophet,
    #                  q4=Fracture, q5=Weaver, q6=Void
    # Bitstring "1000000" → little-endian → qubit 6 active → PATHWAYS[6]="Void"
    ok1 = bits1 == "0000001" and active1 == ["Awakening"] and abs(marg1["Awakening"] - 0.875) < 1e-6
    ok2 = bits2 == "1000000" and active2 == ["Void"] and abs(marg2["Void"] - (5 / 6)) < 1e-6
    ok3 = bits3 == "0000000" and active3 == ["Void"] and all(v == 0.0 for v in marg3.values())
    ok4 = set(marg1.keys()) == set(qs.PATHWAYS.values())
    passed = ok1 and ok2 and ok3 and ok4

    detail = "\n".join([
        f"  Case 1 bits={bits1} active={active1} Awakening={marg1['Awakening']:.3f}",
        f"  Case 2 bits={bits2} active={active2} Void={marg2['Void']:.3f}",
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
        weaver._supervised(flaky, "Flaky", restart_on_crash=True, restart_delay=0.05)
    )
    await asyncio.sleep(0.25)
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
        weaver._supervised(one_shot, "OneShot", restart_on_crash=False, restart_delay=0.05)
    )
    await asyncio.sleep(0.05)
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
        and "GEMINI_API_KEY" in text
        and "WEAVER_VISION_KEY" not in text
    )
    detail = "\n".join([
        f"  Return code: {proc.returncode}",
        f"  Output: {text.strip()[:220]}",
        "  Expected missing keys: WEAVER_VOICE_KEY, WEAVER_MEM_KEY, GEMINI_API_KEY",
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", nargs="?", default="all", help="G-S or all")
    args = ap.parse_args()
    asyncio.run(main(args.test))
