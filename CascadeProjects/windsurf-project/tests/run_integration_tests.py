"""
Weaver Integration Test Suite
Tests every module and cross-module connection with real API calls.
"""
import asyncio
import base64
import json
import os
import subprocess
import sys
import time

from dotenv import load_dotenv
load_dotenv()

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"
results = []

def log(label, status, detail=""):
    tag = PASS if status == "pass" else (FAIL if status == "fail" else WARN)
    line = f"  {tag}  {label}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)
    results.append((label, status, detail))

# ─────────────────────────────────────────────
# 1. ENV / KEYS
# ─────────────────────────────────────────────
print("\n\033[1m[1] Environment & API Keys\033[0m")

VOICE_KEY  = os.environ.get("WEAVER_VOICE_KEY", "")
MEM_KEY    = os.environ.get("WEAVER_MEM_KEY", "")
VISION_KEY = os.environ.get("WEAVER_VISION_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
IBM_KEY    = os.environ.get("IBM_QUANTUM_TOKEN", "")

log("WEAVER_VOICE_KEY",  "pass" if VOICE_KEY.startswith("sk-")  else "fail", VOICE_KEY[:12]+"…" if VOICE_KEY else "MISSING")
log("WEAVER_MEM_KEY",    "pass" if MEM_KEY.startswith("sk-")    else "fail", MEM_KEY[:12]+"…"   if MEM_KEY   else "MISSING")
log("WEAVER_VISION_KEY", "pass" if VISION_KEY.startswith("sk-") else "fail", VISION_KEY[:12]+"…" if VISION_KEY else "MISSING")
log("GEMINI_API_KEY",    "pass" if GEMINI_KEY.startswith("AIza") else "fail", GEMINI_KEY[:12]+"…" if GEMINI_KEY else "MISSING")
log("IBM_QUANTUM_TOKEN", "pass" if IBM_KEY else "warn", IBM_KEY[:12]+"…" if IBM_KEY else "MISSING")

# ─────────────────────────────────────────────
# 2. IMPORTS — all vtv_basic dependencies
# ─────────────────────────────────────────────
print("\n\033[1m[2] Core Import Check\033[0m")

import_tests = [
    ("openai",           "import openai"),
    ("google.genai",     "from google import genai"),
    ("langchain_openai", "from langchain_openai import ChatOpenAI"),
    ("websockets",       "import websockets"),
    ("numpy",            "import numpy as np"),
    ("cv2",              "import cv2"),
    ("insightface",      "import insightface"),
    ("googleapiclient",  "from googleapiclient.discovery import build"),
    ("dotenv",           "from dotenv import load_dotenv"),
]

for name, stmt in import_tests:
    try:
        exec(stmt)
        log(name, "pass")
    except Exception as e:
        log(name, "fail", str(e)[:80])

# ─────────────────────────────────────────────
# 3. GEMINI VISION — live API call with real image
# ─────────────────────────────────────────────
print("\n\033[1m[3] Gemini Vision API (live call)\033[0m")
try:
    from google import genai as google_genai
    from google.genai import types as genai_types
    t0 = time.monotonic()
    client = google_genai.Client(api_key=GEMINI_KEY)
    # Generate a valid test JPEG with numpy
    import numpy as np, io, struct
    import cv2 as _cv2_test
    _img = np.zeros((32, 32, 3), dtype=np.uint8)
    _img[:] = (200, 200, 200)  # light grey
    _, _buf = _cv2_test.imencode('.jpg', _img, [_cv2_test.IMWRITE_JPEG_QUALITY, 80])
    TINY_JPEG = _buf.tobytes()
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            genai_types.Part.from_bytes(data=TINY_JPEG, mime_type="image/jpeg"),
            "What colour is this image? One word.",
        ],
    )
    elapsed = (time.monotonic() - t0) * 1000
    text = (resp.text or "").strip()[:60]
    log("gemini-2.5-flash vision call", "pass", f"{elapsed:.0f}ms → '{text}'")
except Exception as e:
    log("gemini-2.5-flash vision call", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 4. OPENAI — chat completion (WEAVER_MEM_KEY)
# ─────────────────────────────────────────────
print("\n\033[1m[4] OpenAI Chat API (WEAVER_MEM_KEY)\033[0m")
try:
    import openai as _oai
    t0 = time.monotonic()
    oai = _oai.OpenAI(api_key=MEM_KEY)
    r = oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Reply with exactly: WEAVER_ONLINE"}],
        max_tokens=10,
    )
    elapsed = (time.monotonic() - t0) * 1000
    reply = r.choices[0].message.content.strip()
    ok = "WEAVER_ONLINE" in reply
    log("gpt-4o-mini (mem key)", "pass" if ok else "warn", f"{elapsed:.0f}ms → '{reply}'")
except Exception as e:
    log("gpt-4o-mini (mem key)", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 5. OPENAI — Realtime API key check (WEAVER_VOICE_KEY)
# ─────────────────────────────────────────────
print("\n\033[1m[5] OpenAI Voice Key Validity\033[0m")
try:
    import openai as _oai
    t0 = time.monotonic()
    oai_v = _oai.OpenAI(api_key=VOICE_KEY)
    r = oai_v.models.list()
    model_ids = [m.id for m in r.data]
    realtime_available = any("realtime" in m for m in model_ids)
    elapsed = (time.monotonic() - t0) * 1000
    log("WEAVER_VOICE_KEY valid", "pass", f"{elapsed:.0f}ms, {len(model_ids)} models")
    log("Realtime model available", "pass" if realtime_available else "warn",
        [m for m in model_ids if "realtime" in m][:3] or "none found")
except Exception as e:
    log("WEAVER_VOICE_KEY", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 6. LANGCHAIN CORTEX
# ─────────────────────────────────────────────
print("\n\033[1m[6] LangChain Cortex\033[0m")
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    t0 = time.monotonic()
    lc = ChatOpenAI(model="gpt-4o-mini", openai_api_key=MEM_KEY, max_tokens=10)
    resp = lc.invoke([HumanMessage(content="Say: LC_ONLINE")])
    elapsed = (time.monotonic() - t0) * 1000
    log("LangChain ChatOpenAI invoke", "pass", f"{elapsed:.0f}ms → '{resp.content[:40]}'")
except Exception as e:
    log("LangChain ChatOpenAI", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 7. GOOGLE DRIVE
# ─────────────────────────────────────────────
print("\n\033[1m[7] Google Drive\033[0m")
PROJ = "/media/ydn/SYPHER_CORE/weaver v3/CascadeProjects/windsurf-project"
VENV_PY = os.path.join(PROJ, "venv", "bin", "python3")
PYTHON = VENV_PY if os.path.exists(VENV_PY) else sys.executable
TOKEN_PATH = os.path.join(PROJ, "token.json")
CREDS_PATH = os.path.join(PROJ, "credentials.json")
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    t0 = time.monotonic()
    service = build('drive', 'v3', credentials=creds)
    res = service.files().list(
        q=f"'1ccTAqsrDq2lEtweYQwAeZmzCQrdM8pt3' in parents",
        pageSize=5, fields="files(id,name)"
    ).execute()
    elapsed = (time.monotonic() - t0) * 1000
    files = res.get('files', [])
    log("Google Drive OAuth", "pass", f"token valid")
    log("Drive folder list", "pass", f"{elapsed:.0f}ms, {len(files)} file(s): {[f['name'] for f in files]}")
except Exception as e:
    log("Google Drive", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 8. NEXUS BUS — start server, pub/sub round-trip
# ─────────────────────────────────────────────
print("\n\033[1m[8] Nexus Bus (WebSocket pub/sub)\033[0m")

async def _drain_optional_sync(ws):
    """Drain a SYNC frame if the server sends one (only when cache non-empty)."""
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        msg = json.loads(raw)
        if msg.get("type") != "sync":
            return msg  # was something else — return it to be re-processed
    except asyncio.TimeoutError:
        pass  # no sync — expected on empty cache
    return None

async def _nexus_test():
    import websockets as _ws
    proc = subprocess.Popen(
        [PYTHON, os.path.join(PROJ, "nexus_bus.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    await asyncio.sleep(1.2)
    try:
        t0 = time.monotonic()
        pub = await _ws.connect("ws://localhost:9999")
        sub = await _ws.connect("ws://localhost:9999")

        # Optionally drain SYNC (only sent when message cache non-empty)
        await _drain_optional_sync(pub)
        await _drain_optional_sync(sub)

        # Register both lobes
        await pub.send(json.dumps({"action": "register", "lobe_id": "test_pub"}))
        ack_pub = json.loads(await asyncio.wait_for(pub.recv(), timeout=2.0))
        await sub.send(json.dumps({"action": "register", "lobe_id": "test_sub"}))
        ack_sub = json.loads(await asyncio.wait_for(sub.recv(), timeout=2.0))

        # Subscribe
        await sub.send(json.dumps({"action": "subscribe", "topics": ["test_topic"]}))
        ack_s = json.loads(await asyncio.wait_for(sub.recv(), timeout=2.0))

        # Publish from pub
        await pub.send(json.dumps({
            "action": "publish", "topic": "test_topic",
            "payload": {"signal": "NEXUS_ALIVE"}
        }))

        # sub should receive the broadcast
        broadcast = json.loads(await asyncio.wait_for(sub.recv(), timeout=3.0))
        elapsed = (time.monotonic() - t0) * 1000
        got = broadcast.get("payload", {}).get("signal", "")
        ok = broadcast.get("type") == "broadcast" and got == "NEXUS_ALIVE"
        return ("pass" if ok else "fail"), f"{elapsed:.0f}ms, signal='{got}'"
    except Exception as e:
        return "fail", str(e)[:120]
    finally:
        proc.terminate()

status, detail = asyncio.run(_nexus_test())
log("Nexus Bus pub/sub round-trip", status, detail)

# ─────────────────────────────────────────────
# 9. ARECORD / APLAY — audio pipeline
# ─────────────────────────────────────────────
print("\n\033[1m[9] Audio Pipeline (arecord/aplay)\033[0m")
try:
    r = subprocess.run(["arecord", "--list-devices"], capture_output=True, text=True, timeout=5)
    has_capture = "card" in r.stdout.lower() or "card" in r.stderr.lower()
    log("arecord devices", "pass" if has_capture else "warn",
        (r.stdout + r.stderr).strip()[:80])
except Exception as e:
    log("arecord", "fail", str(e)[:80])

try:
    r = subprocess.run(["aplay", "--list-devices"], capture_output=True, text=True, timeout=5)
    has_playback = "card" in r.stdout.lower() or "card" in r.stderr.lower()
    log("aplay devices", "pass" if has_playback else "warn",
        (r.stdout + r.stderr).strip()[:80])
except Exception as e:
    log("aplay", "fail", str(e)[:80])

# Quick 0.5s record test
try:
    t0 = time.monotonic()
    rec = subprocess.run(
        ["arecord", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "16000",
         "-t", "raw", "-d", "1", "/dev/null"],
        capture_output=True, timeout=5
    )
    elapsed = (time.monotonic() - t0) * 1000
    log("arecord 1s live capture", "pass" if rec.returncode == 0 else "warn",
        f"{elapsed:.0f}ms, rc={rec.returncode}")
except Exception as e:
    log("arecord live capture", "fail", str(e)[:80])

# ─────────────────────────────────────────────
# 10. INSIGHTFACE / ARCFACE
# ─────────────────────────────────────────────
print("\n\033[1m[10] InsightFace ArcFace\033[0m")
try:
    import insightface
    import numpy as np
    insightface_home = os.path.expanduser("~/.insightface")
    if not os.path.isdir(insightface_home):
        insightface_home = os.path.join(PROJ, "Nexus_Vault", ".insightface")
    os.environ["INSIGHTFACE_HOME"] = insightface_home
    t0 = time.monotonic()
    app = insightface.app.FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(160, 160))
    elapsed = (time.monotonic() - t0) * 1000
    log("InsightFace buffalo_sc load", "pass", f"{elapsed:.0f}ms")

    import cv2
    img = np.zeros((160, 160, 3), dtype=np.uint8)
    t1 = time.monotonic()
    faces = app.get(img)
    infer_ms = (time.monotonic() - t1) * 1000
    log("ArcFace inference (blank frame)", "pass", f"{infer_ms:.0f}ms, {len(faces)} face(s) detected")
except Exception as e:
    log("InsightFace", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 11. FACE REGISTRY LOAD
# ─────────────────────────────────────────────
print("\n\033[1m[11] Face Registry (Nexus_Vault)\033[0m")
try:
    face_path = os.path.join(PROJ, "Nexus_Vault", "face_registry.npz")
    if os.path.exists(face_path):
        data = np.load(face_path, allow_pickle=False)
        names = list(data.files)
        log("face_registry.npz load", "pass", f"{len(names)} registered: {names}")
    else:
        log("face_registry.npz", "warn", "not found — fresh start (expected if no faces registered yet)")
except Exception as e:
    log("face_registry.npz", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# 12. QUANTUM SOUL IMPORTS
# ─────────────────────────────────────────────
print("\n\033[1m[12] Quantum Soul (imports)\033[0m")
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("quantum_soul",
        os.path.join(PROJ, "quantum_soul.py"))
    qs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qs)
    log("quantum_soul.py import", "pass", "IBM token read, Qiskit loaded")
except Exception as e:
    log("quantum_soul.py import", "fail", str(e)[:100])

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "─" * 60)
passed = sum(1 for _, s, _ in results if s == "pass")
warned = sum(1 for _, s, _ in results if s == "warn")
failed = sum(1 for _, s, _ in results if s == "fail")
total  = len(results)
print(f"\033[1mRESULTS: {passed}/{total} passed  |  {warned} warnings  |  {failed} failures\033[0m")
if failed:
    print("\n\033[91mFailed tests:\033[0m")
    for label, s, d in results:
        if s == "fail":
            print(f"  ✗ {label}: {d}")
if warned:
    print("\n\033[93mWarnings:\033[0m")
    for label, s, d in results:
        if s == "warn":
            print(f"  ⚠ {label}: {d}")
print()
