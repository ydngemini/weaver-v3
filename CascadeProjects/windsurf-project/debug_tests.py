#!/usr/bin/env python3
"""
debug_tests.py — Weaver 5-Test Suite
Each test runs for exactly 5 minutes and checks a specific subsystem.
Run: python3 debug_tests.py [1|2|3|4|5|all]
"""
import argparse
import asyncio
import math
import os
import struct
import subprocess
import sys
import time
from datetime import datetime

PROJ = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(PROJ, "venv", "bin", "python3")
sys.path.insert(0, PROJ)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJ, ".env"))

TEST_DURATION = 300   # 5 minutes per test
BAR = "─" * 60

def _ts():
    return datetime.now().strftime("%H:%M:%S")

def _header(n, title):
    print(f"\n{BAR}\n[{_ts()}] TEST {n}: {title}  ({TEST_DURATION//60} min)\n{BAR}", flush=True)

def _result(n, title, passed, notes):
    mark = "✅  PASS" if passed else "❌  FAIL"
    print(f"\n{BAR}\n{mark}  Test {n}: {title}\n{notes}\n{BAR}\n", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Nexus Bus  (WebSocket pub/sub stability over 5 min)
# ─────────────────────────────────────────────────────────────────────────────
async def _test1():
    import websockets, json
    _header(1, "Nexus Bus — WebSocket pub/sub stability")

    # Start nexus_bus as subprocess
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "nexus_bus.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(1.5)   # let it bind

    msgs_sent = 0
    msgs_recv = 0
    errors = []
    start = time.monotonic()

    async def _drain_sync(ws):
        """Consume the initial SYNC message sent on connect (may or may not arrive)."""
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            msg = json.loads(raw)
            if msg.get("type") != "sync":
                return msg  # not a sync — return it so caller can process
        except asyncio.TimeoutError:
            pass
        return None

    try:
        async with websockets.connect("ws://localhost:9999") as pub:
            async with websockets.connect("ws://localhost:9999") as sub:
                # Drain initial SYNC messages
                await _drain_sync(pub)
                await _drain_sync(sub)

                # Subscribe sub to "test" topic (topics must be an array)
                await sub.send(json.dumps({"action": "subscribe", "topics": ["test"]}))
                # Drain ACK
                try:
                    await asyncio.wait_for(sub.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

                # Publish every second for 5 min
                while time.monotonic() - start < TEST_DURATION:
                    await pub.send(json.dumps({
                        "action": "publish",
                        "topic": "test",
                        "payload": {"t": time.monotonic(), "n": msgs_sent},
                    }))
                    msgs_sent += 1

                    try:
                        raw = await asyncio.wait_for(sub.recv(), timeout=2.0)
                        msg = json.loads(raw)
                        # Server sends {"type":"broadcast","topic":"test","payload":{...}}
                        if msg.get("type") == "broadcast" and msg.get("payload", {}).get("n") == msgs_sent - 1:
                            msgs_recv += 1
                        else:
                            errors.append(f"T={time.monotonic()-start:.1f}s unexpected: {msg.get('type')}")
                    except asyncio.TimeoutError:
                        errors.append(f"T={time.monotonic()-start:.1f}s timeout")

                    if msgs_sent % 30 == 0:
                        print(f"  [{_ts()}] sent={msgs_sent} recv={msgs_recv} errors={len(errors)}", flush=True)

                    await asyncio.sleep(1.0)

    finally:
        proc.terminate()
        await proc.wait()

    passed = msgs_recv >= msgs_sent * 0.95 and len(errors) < 10
    _result(1, "Nexus Bus",
            passed,
            f"  Sent: {msgs_sent}  Received: {msgs_recv}  Errors: {len(errors)}\n"
            + (f"  Last errors: {errors[-3:]}" if errors else "  No errors."))
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — Audio Pipeline  (sounddevice capture + aplay silence stability)
# ─────────────────────────────────────────────────────────────────────────────
async def _test2():
    _header(2, "Audio Pipeline — pacat capture (PipeWire-native) + aplay silence keepalive")

    SOURCE = "alsa_input.pci-0000_00_1f.3.analog-stereo"
    CHUNK_BYTES = 4800  # 2400 samples × 2 bytes @ 24kHz = 100ms
    SILENCE_CHUNK = bytes(2400 * 2)

    rms_readings = []
    samples_total = 0
    chunk_count = 0
    timeout_count = 0
    q: asyncio.Queue[bytes] = asyncio.Queue()

    # Start aplay first (output codec warmup)
    aplay = await asyncio.create_subprocess_exec(
        "aplay", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "24000", "-q",
        "--buffer-time=20000", "--period-time=5000", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await asyncio.sleep(1.0)

    # Start pacat recorder (PipeWire-native — survives WirePlumber graph changes)
    pacat = await asyncio.create_subprocess_exec(
        "pacat", "-r",
        f"--device={SOURCE}",
        "--format=s16le", "--rate=24000", "--channels=1",
        "--latency-msec=50",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def _reader():
        buf = b""
        while True:
            try:
                data = await asyncio.wait_for(pacat.stdout.read(CHUNK_BYTES), timeout=2.0)
                if not data:
                    break
                buf += data
                while len(buf) >= CHUNK_BYTES:
                    q.put_nowait(buf[:CHUNK_BYTES])
                    buf = buf[CHUNK_BYTES:]
            except asyncio.TimeoutError:
                print(f"\n  [{_ts()}] ⚠ pacat read timeout", flush=True)
            except Exception:
                break

    reader = asyncio.create_task(_reader())

    # Calibrate RMS baseline (2 sec after both processes are running)
    await asyncio.sleep(2.0)
    cal_data = b""
    while not q.empty():
        cal_data += q.get_nowait()
    if cal_data:
        s = struct.unpack_from("<" + "h" * (len(cal_data) // 2), cal_data)
        baseline_rms = int(math.sqrt(sum(x * x for x in s) / max(len(s), 1)))
    else:
        baseline_rms = 0
    print(f"  [{_ts()}] Baseline RMS after aplay started: {baseline_rms}", flush=True)

    start = time.monotonic()
    report_at = start + 30

    while time.monotonic() - start < TEST_DURATION:
        # Trickle silence to keep A2DP codec alive
        try:
            aplay.stdin.write(SILENCE_CHUNK)
            await aplay.stdin.drain()
        except Exception:
            pass

        # Drain all available chunks
        drained = 0
        while not q.empty():
            try:
                data = q.get_nowait()
                s = struct.unpack_from("<" + "h" * (len(data) // 2), data)
                rms = int(math.sqrt(sum(x * x for x in s) / max(len(s), 1)))
                rms_readings.append(rms)
                samples_total += len(s)
                chunk_count += 1
                drained += 1
            except Exception:
                break

        if drained == 0:
            try:
                data = await asyncio.wait_for(q.get(), timeout=0.5)
                s = struct.unpack_from("<" + "h" * (len(data) // 2), data)
                rms = int(math.sqrt(sum(x * x for x in s) / max(len(s), 1)))
                rms_readings.append(rms)
                samples_total += len(s)
                chunk_count += 1
            except asyncio.TimeoutError:
                timeout_count += 1
                print(f"  [{_ts()}] ⚠ STREAM DROPOUT #{timeout_count}", flush=True)

        if time.monotonic() >= report_at:
            avg = sum(rms_readings[-50:]) / max(len(rms_readings[-50:]), 1)
            print(f"  [{_ts()}] avg_rms={avg:.0f} (floor≈{baseline_rms})  dropouts={timeout_count}  chunks={chunk_count}", flush=True)
            report_at = time.monotonic() + 30
        await asyncio.sleep(0.05)

    reader.cancel()
    pacat.terminate()
    await pacat.wait()
    aplay.terminate()
    await aplay.wait()

    avg_rms = sum(rms_readings) / max(len(rms_readings), 1)
    expected_chunks = TEST_DURATION // 0.1  # ~3000 chunks at 100ms each
    passed = chunk_count > expected_chunks * 0.8 and timeout_count < 5
    _result(2, "Audio Pipeline",
            passed,
            f"  Baseline RMS: {baseline_rms}  Avg RMS: {avg_rms:.0f}\n"
            f"  Chunks: {chunk_count}/{int(expected_chunks)}  Stream dropouts: {timeout_count}\n"
            f"  Total samples: {samples_total}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Quantum Soul  (IBM job submission + state file write)
# ─────────────────────────────────────────────────────────────────────────────
async def _test3():
    _header(3, "Quantum Soul — IBM job loop stability")

    state_file = os.path.join(PROJ, "Nexus_Vault", "quantum_state.txt")
    mtime_before = os.path.getmtime(state_file) if os.path.exists(state_file) else 0

    # Run quantum_soul as subprocess — it runs one job then waits 5 min
    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "quantum_soul.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    output_lines = []
    start = time.monotonic()
    errors = []

    async def _read():
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                if not line:
                    break
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    output_lines.append(decoded)
                    print(f"  [{_ts()}] {decoded}", flush=True)
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - start > TEST_DURATION:
                break

    reader = asyncio.create_task(_read())
    await asyncio.sleep(TEST_DURATION)
    reader.cancel()
    proc.terminate()
    await proc.wait()

    job_submitted = any("Job ID:" in l for l in output_lines)
    job_complete = any("Job complete" in l for l in output_lines)
    state_written = os.path.exists(state_file) and os.path.getmtime(state_file) > mtime_before
    error_lines = [l for l in output_lines if "ERROR" in l or "error" in l.lower()]

    passed = job_submitted and (job_complete or "waiting" in " ".join(output_lines).lower())
    _result(3, "Quantum Soul",
            passed,
            f"  Job submitted: {job_submitted}  Job complete: {job_complete}\n"
            f"  State file updated: {state_written}\n"
            f"  Errors: {error_lines[:3] if error_lines else 'none'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — VTV Core threads  (camera + face tracker + vision stability)
# ─────────────────────────────────────────────────────────────────────────────
async def _test4():
    _header(4, "VTV Core — camera / face-tracker / vision threads (5 min)")

    proc = await asyncio.create_subprocess_exec(
        VENV, "-c",
        f"""
import sys, os, asyncio, time
PROJ = '/media/ydn/SYPHER_CORE/weaver v3/CascadeProjects/windsurf-project'
sys.path.insert(0, PROJ)
from dotenv import load_dotenv; load_dotenv()
import cv2, warnings
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')

async def run():
    from insightface.app import FaceAnalysis
    face_app = await asyncio.to_thread(lambda: (lambda a: (a.prepare(ctx_id=-1, det_size=(160,160)), a)[1])(FaceAnalysis(name='buffalo_sc', providers=['CPUExecutionProvider'])))
    print('[T4] Face app loaded', flush=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('[T4] Camera FAILED', flush=True)
        return

    frames = 0
    faces_detected = 0
    start = time.monotonic()
    while time.monotonic() - start < {TEST_DURATION}:
        ret, frame = await asyncio.to_thread(cap.read)
        if ret:
            frames += 1
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = await asyncio.to_thread(face_app.get, frame)
            faces_detected += len(result)
        await asyncio.sleep(0.5)
        if frames % 20 == 0:
            print(f'[T4] frames={{frames}} faces_detected={{faces_detected}}', flush=True)
    cap.release()
    print(f'[T4] DONE frames={{frames}} faces_detected={{faces_detected}}', flush=True)

asyncio.run(run())
""",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJ,
    )

    output_lines = []
    start = time.monotonic()

    async def _read():
        while True:
            try:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                if not line:
                    break
                decoded = line.decode(errors="replace").rstrip()
                if decoded:
                    output_lines.append(decoded)
                    print(f"  {decoded}", flush=True)
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - start > TEST_DURATION:
                break

    # Also check stderr
    async def _read_err():
        while True:
            try:
                line = await asyncio.wait_for(proc.stderr.readline(), timeout=1.0)
                if not line:
                    break
                decoded = line.decode(errors="replace").rstrip()
                if decoded and "WARNING" not in decoded and "INFO" not in decoded:
                    print(f"  [ERR] {decoded}", flush=True)
            except asyncio.TimeoutError:
                pass
            if time.monotonic() - start > TEST_DURATION:
                break

    r = asyncio.create_task(_read())
    e = asyncio.create_task(_read_err())
    await asyncio.sleep(TEST_DURATION)
    r.cancel(); e.cancel()
    import contextlib as _cl
    with _cl.suppress(ProcessLookupError):
        proc.terminate()
    with _cl.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=3.0)

    loaded = any("loaded" in l.lower() or "ready" in l.lower() for l in output_lines)
    done = any("DONE" in l for l in output_lines)
    err_lines = [l for l in output_lines if "fail" in l.lower() or "error" in l.lower()]

    passed = loaded and not err_lines
    _result(4, "VTV Core threads",
            passed,
            f"  Face app loaded: {loaded}  Completed: {done}\n"
            f"  Errors: {err_lines[:3] if err_lines else 'none'}")
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Full Weaver Integration  (all lobes for 5 min)
# ─────────────────────────────────────────────────────────────────────────────
async def _test5():
    _header(5, "Full Weaver Integration — all lobes for 5 min")

    proc = await asyncio.create_subprocess_exec(
        VENV, os.path.join(PROJ, "weaver.py"),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=PROJ,
    )

    events = {
        "nexus_live":    False,
        "quantum_start": False,
        "vtv_start":     False,
        "realtime_conn": False,
        "mic_streaming": False,
        "greeting":      False,
        "vision":        False,
        "quantum_job":   False,
    }
    crashes = []
    start = time.monotonic()

    async def _monitor():
        async for line in proc.stdout:
            decoded = line.decode(errors="replace").rstrip()
            if not decoded:
                continue
            t = time.monotonic() - start
            print(f"  [{t:5.1f}s] {decoded}", flush=True)
            if "Nexus Bus binding" in decoded or "Nexus Bus LIVE" in decoded:
                events["nexus_live"]    = True
            if "Quantum Soul started"     in decoded: events["quantum_start"] = True
            if "VTV Core started"         in decoded: events["vtv_start"]     = True
            if "REALTIME] Connected"      in decoded: events["realtime_conn"] = True
            if "[MIC] Streaming"          in decoded: events["mic_streaming"] = True
            if "[WEAVER]:"                in decoded: events["greeting"]      = True
            if "[VISION]"                 in decoded: events["vision"]        = True
            if "Job complete"             in decoded or "Job ID" in decoded:
                events["quantum_job"] = True
            if "crashed" in decoded.lower(): crashes.append(decoded)

    mon = asyncio.create_task(_monitor())
    await asyncio.sleep(TEST_DURATION)
    mon.cancel()
    proc.terminate()
    await proc.wait()

    all_lobes = events["nexus_live"] and events["quantum_start"] and events["vtv_start"]
    audio_ok   = events["realtime_conn"] and events["mic_streaming"]
    passed = all_lobes and audio_ok and events["greeting"]
    # Note: quantum_job not required — IBM queue can take >5 min

    lines = []
    for k, v in events.items():
        lines.append(f"  {'✓' if v else '✗'} {k}")
    lines.append(f"  Crashes detected: {len(crashes)}")

    _result(5, "Full Weaver Integration", passed, "\n".join(lines))
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
TESTS = {
    1: ("Nexus Bus",           _test1),
    2: ("Audio Pipeline",      _test2),
    3: ("Quantum Soul",        _test3),
    4: ("VTV Core threads",    _test4),
    5: ("Full Integration",    _test5),
}

async def main(which):
    results = {}
    total_start = time.monotonic()
    for n, (title, fn) in TESTS.items():
        if which != "all" and n != which:
            continue
        results[n] = await fn()

    elapsed = time.monotonic() - total_start
    print(f"\n{'═'*60}")
    print(f"  FINAL RESULTS  ({elapsed/60:.1f} min total)")
    print(f"{'═'*60}")
    for n, (title, _) in TESTS.items():
        if n in results:
            mark = "✅" if results[n] else "❌"
            print(f"  {mark}  Test {n}: {title}")
    passed = sum(1 for v in results.values() if v)
    print(f"\n  {passed}/{len(results)} passed")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("test", nargs="?", default="all",
                    help="Which test to run: 1-5 or 'all' (default)")
    args = ap.parse_args()
    which = int(args.test) if args.test.isdigit() else "all"
    asyncio.run(main(which))
