import argparse
import asyncio
import ast
import math
import base64
import contextlib
import glob
import openai
import os
import time
import numpy as np
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

OPENAI_VISION_BEST = "gpt-4o"       # diary / registration
OPENAI_VISION_FAST = "gpt-4o-mini"  # real-time injection
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import json
import websockets
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

DRIVE_FOLDER_ID = '1ccTAqsrDq2lEtweYQwAeZmzCQrdM8pt3'
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']

CHUNK_SIZE = 2400    # 50ms at 24kHz — matches OpenAI Realtime pcm16 sample rate
RMS_THRESHOLD = 400   # more sensitive gate (was 600)


def _pcm16_rms(data: bytes) -> int:
    if len(data) < 2:
        return 0
    sample_count = len(data) // 2
    total = 0
    for index in range(0, sample_count * 2, 2):
        sample = int.from_bytes(data[index:index + 2], byteorder="little", signed=True)
        total += sample * sample
    return int(math.sqrt(total / sample_count))


def _build_self_code_map(project_dir: str, max_files: int = 20, max_functions_per_file: int = 25) -> str:
    try:
        py_files = sorted(glob.glob(os.path.join(project_dir, "*.py")))[:max_files]
    except Exception:
        return ""

    if not py_files:
        return ""

    chunks: list[str] = []
    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as source_file:
                source = source_file.read()
            tree = ast.parse(source)
        except Exception:
            continue

        function_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                function_names.append(node.name)

        function_names = list(dict.fromkeys(function_names))
        if not function_names:
            continue

        chunks.append(
            f"{os.path.basename(py_file)}: "
            + ", ".join(function_names[:max_functions_per_file])
        )

    if not chunks:
        return ""

    return (
        "Project code map:\n"
        + "\n".join(chunks)
        + "\nUse this as your self-model when asked about your own architecture, memory, voice, vision, and function wiring."
    )


async def _start_pacat_capture(queue: "asyncio.Queue[bytes]",
                               chunk_samples: int = 2400) -> tuple:
    """
    Capture mic audio via pacat (PipeWire-native) into an asyncio queue.
    Unlike sounddevice/ALSA, pacat survives WirePlumber graph reconfigurations
    (e.g. when aplay activates A2DP or changes the default source).
    Returns (proc, reader_task) — both must be cancelled/terminated on exit.
    """
    SOURCE = "alsa_input.pci-0000_00_1f.3.analog-stereo"
    CHUNK_BYTES = chunk_samples * 2  # int16 = 2 bytes/sample
    proc = await asyncio.create_subprocess_exec(
        "pacat", "-r",
        f"--device={SOURCE}",
        "--format=s16le",
        "--rate=24000",
        "--channels=1",
        "--latency-msec=50",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )

    async def _reader():
        buf = b""
        while True:
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(CHUNK_BYTES), timeout=2.0)
                if not chunk:
                    break
                buf += chunk
                while len(buf) >= CHUNK_BYTES:
                    queue.put_nowait(buf[:CHUNK_BYTES])
                    buf = buf[CHUNK_BYTES:]
            except asyncio.TimeoutError:
                print("\n[MIC] pacat read timeout", flush=True)
            except Exception:
                break

    reader_task = asyncio.create_task(_reader())
    return proc, reader_task


async def _auto_connect_audio() -> None:
    SKIP = ("loopback", "hdmi", "dp,pcm")
    PREFER = ("analog", "built-in", "speaker", "internal", "bluetooth", "bluez", "a2dp", "headset", "headphone")
    try:
        proc = await asyncio.create_subprocess_exec(
            "wpctl", "status",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await proc.communicate()
        best_id, best_score, best_name = None, -1, None
        in_sinks = False
        for line in raw.decode(errors="replace").splitlines():
            if "Sinks:" in line:
                in_sinks = True
                continue
            if in_sinks and ("Sources:" in line or "Sink endpoints:" in line):
                in_sinks = False
                continue
            if not in_sinks:
                continue
            clean = line.replace("│", "").replace("├", "").replace("└", "").replace("─", "").replace("*", "").strip()
            parts = clean.split(".")
            if len(parts) < 2:
                continue
            id_part = parts[0].strip()
            if not id_part.isdigit():
                continue
            name = ".".join(parts[1:]).strip().lower().split("[")[0].strip()
            if any(s in name for s in SKIP):
                continue
            score = sum(p in name for p in PREFER)
            if score > best_score:
                best_id, best_score, best_name = id_part, score, name
        if best_id:
            await asyncio.create_subprocess_exec(
                "wpctl", "set-default", best_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            print(f"🔊 [AUDIO] Auto-connected to: {best_name.title()} (id={best_id})", flush=True)
        else:
            print("⚠️ [AUDIO] No suitable speaker sink found — using system default.", flush=True)
    except Exception as e:
        print(f"⚠️ [AUDIO] Auto-connect failed: {e}", flush=True)


async def _start_aplay() -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        "aplay", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "24000", "-q", "--buffer-time=20000", "--period-time=5000", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def _terminate_process(proc: asyncio.subprocess.Process | None) -> None:
    if proc is None:
        return
    if proc.returncode is None:
        proc.terminate()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.5)
    if proc.returncode is None:
        proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()


async def run_vtv(heartbeat: bool = True) -> None:
    voice_api_key = os.environ.get("WEAVER_VOICE_KEY")
    mem_api_key = os.environ.get("WEAVER_MEM_KEY")
    missing = [
        key
        for key, value in (
            ("WEAVER_VOICE_KEY", voice_api_key),
            ("WEAVER_MEM_KEY", mem_api_key),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} in .env")

    openai_vision_client = openai.AsyncOpenAI(api_key=mem_api_key)

    # --- INITIATE THE NEXUS VAULT (ABSOLUTE PATH LOCK) ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vault_dir = os.path.join(base_dir, "Nexus_Vault")
    os.makedirs(vault_dir, exist_ok=True)
    transcript_path = os.path.join(vault_dir, "weaver_transcript.txt")
    cloud_vision_memory_path = os.path.join(vault_dir, "cloud_vision_memory.md")
    people_memory_path = os.path.join(vault_dir, "people_memory.md")
    people_memory: list[str] = [""]
    if os.path.exists(people_memory_path):
        try:
            with open(people_memory_path, "r", encoding="utf-8") as _pm:
                people_memory[0] = _pm.read().strip()
        except Exception:
            pass
    # Images go to /tmp (ephemeral) — only cloud Drive gets the permanent copy
    images_dir = os.path.join("/tmp", "weaver_visual_memories")
    os.makedirs(images_dir, exist_ok=True)

    # --- BOOT OAUTH & RETRIEVE CLOUD MEMORIES ---
    drive_service = None
    cloud_memories = ""
    existing_transcript_id = None
    token_path = os.path.join(base_dir, "token.json")
    try:
        if not os.path.exists(token_path):
            raise FileNotFoundError("token.json not found — run init_drive.py first")
        creds = Credentials.from_authorized_user_file(token_path, DRIVE_SCOPES)
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(token_path, 'w') as tf:
                tf.write(creds.to_json())
        drive_service = await asyncio.to_thread(build, 'drive', 'v3', credentials=creds)
        print("\u2601\ufe0f  [NEXUS CLOUD] OAuth link established.")

        results = await asyncio.to_thread(
            lambda: drive_service.files().list(
                q=f"name='weaver_transcript.txt' and '{DRIVE_FOLDER_ID}' in parents and trashed=false",
                spaces='drive',
                fields='files(id, name)',
                orderBy='modifiedTime desc',
                pageSize=1,
            ).execute()
        )
        files = results.get('files', [])
        if files:
            existing_transcript_id = files[0]['id']
            def _download_transcript():
                import io
                from googleapiclient.http import MediaIoBaseDownload
                fh = io.BytesIO()
                request = drive_service.files().get_media(fileId=files[0]['id'])
                dl = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    _, done = dl.next_chunk()
                return fh.getvalue()
            raw = await asyncio.to_thread(_download_transcript)
            cloud_memories = raw.decode('utf-8', errors='replace')[-2000:]
            print(f"\U0001f9e0 [NEXUS CLOUD] Retrieved {len(cloud_memories)} chars of past memories.")
    except Exception as e:
        print(f"\u26a0\ufe0f [NEXUS CLOUD] Cloud boot issue (non-fatal): {e}")

    await _auto_connect_audio()
    # Auto-prefer Bluetooth sink (output) — keep laptop mic as source (input)
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "list", "sinks", "short",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
        for line in raw.decode().splitlines():
            if "bluez_output" in line and ".monitor" not in line:
                bt_name = line.split()[1]
                # Force A2DP (not HFP) — prevents WirePlumber from hijacking the mic source
                card_id = bt_name.replace("bluez_output.", "bluez_card.").split(".")[0] + "." + ".".join(bt_name.replace("bluez_output.", "").split(".")[:2])
                cp = await asyncio.create_subprocess_exec(
                    "pactl", "set-card-profile", card_id.replace("bluez_output","bluez_card"), "a2dp-sink",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await cp.wait()
                p = await asyncio.create_subprocess_exec(
                    "pactl", "set-default-sink", bt_name,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await p.wait()
                print(f"\U0001f3a7 [AUDIO] Bluetooth output (A2DP): {bt_name}", flush=True)
                break
    except Exception:
        pass
    # Always use laptop mic for input — pin it so HFP/BT doesn't steal it
    try:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "list", "sources", "short",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        raw, _ = await asyncio.wait_for(proc.communicate(), timeout=3)
        for line in raw.decode().splitlines():
            if "alsa_input.pci" in line:
                mic_name = line.split()[1]
                p = await asyncio.create_subprocess_exec(
                    "pactl", "set-default-source", mic_name,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await p.wait()
                print(f"\U0001f3a4 [AUDIO] Mic source: {mic_name}", flush=True)
                break
    except Exception:
        pass
    # Restore mic capture gain to 75% (plughw:0,0 bypasses PipeWire so software gain doesn't apply)
    await asyncio.create_subprocess_exec(
        "pactl", "set-source-volume", "@DEFAULT_SOURCE@", "75%",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    # Start aplay first so the output codec warms up, then open mic via pacat (PipeWire-native)
    aplay_proc_ref: list = [await _start_aplay()]
    await asyncio.sleep(1.0)  # let aplay/codec initialize
    _mic_queue: asyncio.Queue[bytes] = asyncio.Queue()  # unlimited
    _pacat_proc, _pacat_reader = await _start_pacat_capture(_mic_queue)
    arecord_proc = None  # kept for compat; actual capture is from _mic_queue

    audio_out_queue: asyncio.Queue[bytes] = asyncio.Queue()
    cloud_queue: asyncio.Queue[dict] = asyncio.Queue()
    realtime_vision_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2)
    cloud_vision_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=20)
    last_audio_chunk_at: list[float] = [0.0]
    face_tracker_frame: list = [None]   # shared latest raw BGR frame for face tracker
    face_last_face_frame: list = [None] # last frame where a face was actually detected
    face_app_ref: list = [None]         # shared insightface FaceAnalysis app (set by face_tracker_task)
    face_state: dict = {}               # {name: {bbox, score, last_seen}} updated by tracker
    face_registry_path = os.path.join(vault_dir, "face_registry.npz")
    face_registry: dict = {}            # name -> list of L2-normalised 512-d ArcFace embeddings
    if os.path.exists(face_registry_path):
        try:
            data = np.load(face_registry_path, allow_pickle=False)
            for k in data.files:
                arr = data[k].astype(np.float32)          # (N, 512)
                norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
                face_registry[k] = list(arr / norms)      # store unit-vectors
            print(f"\U0001f9d1 [FACE ID] Loaded {len(face_registry)} registered face(s).", flush=True)
        except Exception as _e:
            print(f"\u26a0\ufe0f [FACE ID] Registry load failed ({_e}) — starting fresh.", flush=True)
    self_code_map = _build_self_code_map(base_dir)

    base_instruction = (
        "CRITICAL RULE: You MUST always speak and respond in English only. Never use any other language under any circumstances. "
        "You are a voice assistant called Weaver. "
        "Greet the user when they first speak. "
        "Respond naturally and conversationally. "
        "You receive silent background webcam snapshots every 5 seconds as visual context. "
        "Use this to answer questions about what you see without being asked. "
        "You have a tool called register_face — call it ONLY when the user explicitly introduces someone "
        "(e.g. 'this is Marcus', 'remember her name is Sarah', 'register this person as [name]'). "
        "Pass the person's name as the 'name' argument. Do not call it unprompted. "
        "Only answer the user; do not continue talking to yourself between turns. "
        "When the user asks what you remember, answer only from the transcript and visual context already present in your instructions. "
        "Do not call tools, functions, or external actions to answer memory questions. "
        "In your text thoughts, always write a word-for-word transcript of the conversation "
        "in the format 'USER: [what they said] WEAVER: [what you said]'. "
        "Do this for every turn. This is how you remember conversations."
    )
    if self_code_map:
        base_instruction += "\n\nYour current source-code self map:\n" + self_code_map

    # --- LANGCHAIN CORTEX: Conversation Memory + Reasoning ---
    lc_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=500,
        api_key=mem_api_key,
    )
    brain_queue: asyncio.Queue = asyncio.Queue()
    lc_history: list = []
    lc_summary: list[str] = [cloud_memories]
    last_context_sent: list[str] = [""]
    last_user_message: list[tuple[str, float]] = [('', 0.0)]
    last_assistant_message: list[tuple[str, float]] = [('', 0.0)]
    pending_user_message: list[tuple[str, float]] = [('', 0.0)]
    last_user_activity_at: list[float] = [0.0]
    assistant_is_speaking: list[bool] = [False]
    last_assistant_activity_at: list[float] = [0.0]
    last_turn_complete_at: list[float] = [time.monotonic()]

    def _is_duplicate_message(cache: list[tuple[str, float]], content: str, window_seconds: float = 1.5) -> bool:
        now = time.monotonic()
        last_content, last_seen = cache[0]
        if content == last_content and (now - last_seen) < window_seconds:
            return True
        cache[0] = (content, now)
        return False

    def _build_runtime_context() -> str:
        context_parts = [base_instruction]
        if people_memory[0]:
            context_parts.append(f"\n\nPeople You Know:\n{people_memory[0]}")
        if lc_summary[0]:
            context_parts.append(f"\n\nConversation Memory:\n{lc_summary[0]}")
        if lc_history:
            recent = "\n".join(
                f"{'User' if isinstance(message, HumanMessage) else 'Weaver'}: {message.content}"
                for message in lc_history[-4:]
            )
            context_parts.append(f"\n\nRecent exchanges:\n{recent}")
        return "\n".join(context_parts)

    def _stage_user_message(content: str) -> None:
        now = time.monotonic()
        pending_user_message[0] = (content, now)
        last_user_activity_at[0] = now

    def _flush_pending_user_message(force: bool = False) -> None:
        content, staged_at = pending_user_message[0]
        if not content:
            return
        if not force and (time.monotonic() - staged_at) < 0.1:
            return
        if _is_duplicate_message(last_user_message, content, window_seconds=3.0):
            pending_user_message[0] = ('', 0.0)
            return
        print(f"\n[USER]: {content}", flush=True)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(transcript_path, "a") as tf:
            tf.write(f"[{ts}] USER: {content}\n")
        brain_queue.put_nowait(HumanMessage(content=content))
        pending_user_message[0] = ('', 0.0)

    _SILENCE_50MS = bytes(2400 * 2)  # 50ms silence at 24kHz PCM16 mono

    async def play_audio():
        while True:
            try:
                data = audio_out_queue.get_nowait()
            except asyncio.QueueEmpty:
                # Feed silence to keep A2DP alive (prevents codec reconfiguration that kills mic)
                data = _SILENCE_50MS
                await asyncio.sleep(0.05)
            try:
                aplay_proc_ref[0].stdin.write(data)
                await aplay_proc_ref[0].stdin.drain()
                if data is not _SILENCE_50MS:
                    last_audio_chunk_at[0] = time.monotonic()
            except Exception:
                try:
                    aplay_proc_ref[0] = await _start_aplay()
                    aplay_proc_ref[0].stdin.write(data)
                    await aplay_proc_ref[0].stdin.drain()
                except Exception:
                    pass

    async def capture_video():
        try:
            def _open_camera():
                import cv2
                return cv2, cv2.VideoCapture(0)
            cv2, cap = await asyncio.to_thread(_open_camera)
            if not cap.isOpened():
                print("\n\u26a0\ufe0f [WARNING] Camera failed to open. Weaver is blind.")
                return
            frame_counter = 0
            consecutive_failures = 0
            MAX_FAILURES = 30  # ~3s of failures → give up
            try:
                while True:
                    ret, frame = await asyncio.to_thread(cap.read)
                    if ret:
                        consecutive_failures = 0
                        # FACE TRACKER: share latest raw frame every second
                        face_tracker_frame[0] = frame.copy()

                        # NATIVE VISION TICK: 1 Pic Every 3 Seconds (~10 fps)
                        if frame_counter % 30 == 0:
                            frame_low = cv2.resize(frame, (512, 512))
                            _, buffer = cv2.imencode('.jpg', frame_low, [cv2.IMWRITE_JPEG_QUALITY, 70])
                            b64 = base64.b64encode(buffer).decode("ascii")
                            if realtime_vision_queue.full():
                                realtime_vision_queue.get_nowait()
                            realtime_vision_queue.put_nowait(b64)

                        # CLOUD DIARY TICK: Every 5 seconds (~10 fps)
                        if frame_counter % 50 == 0:
                            frame_high = cv2.resize(frame, (640, 480))
                            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            filename = f"memory_{timestamp}.jpg"
                            image_path = os.path.join(images_dir, filename)
                            cv2.imwrite(image_path, frame_high)
                            await cloud_queue.put({"type": "image", "path": image_path, "name": filename})
                            if not cloud_vision_queue.full():
                                await cloud_vision_queue.put(image_path)

                            all_imgs = sorted(glob.glob(os.path.join(images_dir, "memory_*.jpg")))
                            for old in all_imgs[:-5]:
                                with contextlib.suppress(OSError): os.remove(old)

                        frame_counter += 1
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_FAILURES:
                            print("\n⚠️ [CAMERA] Too many read failures — camera disconnected. Weaver is blind.", flush=True)
                            break
                    await asyncio.sleep(0.1)
            finally:
                if cap.isOpened(): cap.release()
        except Exception as e: print(f"\n\u26a0\ufe0f [CAMERA ERROR]: {e}")

    async def face_tracker_task():
        """Real face ID + body tracking using InsightFace ArcFace + Haar upper-body cascade."""
        import cv2
        import warnings
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            print("\n\u26a0\ufe0f [FACE TRACKER] insightface not installed.", flush=True)
            return

        # Prefer ~/.insightface (already downloaded) to avoid re-downloading each run
        _sys_home = os.path.expanduser("~/.insightface")
        insightface_home = _sys_home if os.path.isdir(_sys_home) else os.path.join(vault_dir, ".insightface")
        os.environ["INSIGHTFACE_HOME"] = insightface_home
        os.makedirs(insightface_home, exist_ok=True)

        print("\n\U0001f9d1 [FACE TRACKER] Loading ArcFace model (first run downloads ~90 MB)...", flush=True)
        try:
            def _init_app():
                import sys, io
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _saved = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        app = FaceAnalysis(name="buffalo_sc", root=insightface_home,
                                           providers=["CPUExecutionProvider"])
                        app.prepare(ctx_id=0, det_size=(320, 320))
                    finally:
                        sys.stdout = _saved
                return app
            face_app = await asyncio.to_thread(_init_app)
            face_app_ref[0] = face_app   # share with register_face handler
        except Exception as e:
            print(f"\n\u26a0\ufe0f [FACE TRACKER] Model init failed: {e}", flush=True)
            return

        # Haar upper-body cascade — works better than HOG for close webcam shots
        body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_upperbody.xml"
        )
        print("\n\U0001f9d1 [FACE TRACKER] Active — ArcFace + upper-body cascade ready.", flush=True)

        last_printed = ""   # only log when state description changes
        _last_frame_id = None  # skip inference when frame hasn't changed
        _sticky_name = [None, 0.0]  # [name, expiry] — prevents oscillation near threshold

        while True:
            frame = face_tracker_frame[0]
            if frame is None:
                await asyncio.sleep(0.1)
                continue
            # Skip if same frame object — no new data
            fid = id(frame)
            if fid == _last_frame_id:
                await asyncio.sleep(0.05)
                continue
            _last_frame_id = fid
            try:
                def _detect():
                    import sys, io
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _saved = sys.stdout
                        sys.stdout = io.StringIO()
                        try:
                            faces = face_app.get(frame)
                        finally:
                            sys.stdout = _saved
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    bodies = body_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=3,
                        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    return faces, bodies

                faces, bodies = await asyncio.to_thread(_detect)

                seen = []
                for face in faces:
                    # Normalise query embedding once; stored embeddings are pre-normalised at load
                    raw_emb = face.embedding.astype(np.float32)
                    norm = np.linalg.norm(raw_emb)
                    emb = raw_emb / (norm + 1e-9)
                    name, best = "unregistered", 0.0
                    for kname, kembs in face_registry.items():
                        for ke in kembs:
                            # Pure dot product — both sides are unit vectors
                            score = float(np.dot(emb, ke))
                            if score > best:
                                best = score
                                if score > 0.30:   # buffalo_sc: same person ~0.35-0.6
                                    name = kname
                    # Sticky name: prevent oscillation when confidence fluctuates near threshold
                    if name != "unregistered":
                        _sticky_name[0] = name
                        _sticky_name[1] = time.monotonic() + 4.0
                    elif best > 0.20 and _sticky_name[0] and time.monotonic() < _sticky_name[1]:
                        name = _sticky_name[0]  # keep confirmed name while still plausible
                    bbox = face.bbox.astype(int).tolist()
                    seen.append((name, round(best, 2), bbox))

                if faces:
                    face_last_face_frame[0] = frame  # save last frame with a confirmed face

                body_count = len(bodies) if (bodies is not None and hasattr(bodies, '__len__')) else 0

                now = time.monotonic()
                face_state.clear()
                for name, score, bbox in seen:
                    face_state[name] = {"bbox": bbox, "score": score, "last_seen": now}
                face_state["_body_count"] = body_count

                # Only log when something meaningful changes
                if seen:
                    line = ", ".join(f"{n}({s:.2f})" for n, s, _ in seen)
                    status = f"{line} | bodies={body_count}"
                    # Compare only names (not scores) to avoid per-frame spam
                    name_key = ", ".join(n for n, _, _ in seen) + f"|b={body_count}"
                elif body_count > 0:
                    status = f"no face | bodies={body_count}"
                    name_key = f"noface|b={body_count}"
                else:
                    status = ""
                    name_key = ""

                if status and name_key != last_printed:
                    print(f"\n\U0001f9d1 [FACE] {status}", flush=True)
                    last_printed = name_key
                elif not status:
                    last_printed = ""

            except Exception:
                pass
            await asyncio.sleep(0.15)

    async def cloud_worker():
        if not drive_service:
            print("\n\u26a0\ufe0f [NEXUS CLOUD] No drive connection — cloud sync disabled.")
            return

        print("\n\u2601\ufe0f [NEXUS CLOUD] Background sync active.")
        transcript_file_id = existing_transcript_id

        while True:
            item = await cloud_queue.get()
            try:
                if item["type"] == "image":
                    _meta = {'name': item["name"], 'parents': [DRIVE_FOLDER_ID]}
                    _media = MediaFileUpload(item["path"], mimetype='image/jpeg', resumable=True)
                    await asyncio.to_thread(
                        lambda m=_meta, d=_media: drive_service.files().create(
                            body=m, media_body=d, fields='id'
                        ).execute()
                    )
                elif item["type"] == "transcript":
                    _media = MediaFileUpload(item["path"], mimetype='text/plain', resumable=True)
                    if transcript_file_id:
                        _fid = transcript_file_id
                        await asyncio.to_thread(
                            lambda d=_media, fid=_fid: drive_service.files().update(
                                fileId=fid, media_body=d
                            ).execute()
                        )
                    else:
                        _meta = {'name': 'weaver_transcript.txt', 'parents': [DRIVE_FOLDER_ID]}
                        result = await asyncio.to_thread(
                            lambda m=_meta, d=_media: drive_service.files().create(
                                body=m, media_body=d, fields='id'
                            ).execute()
                        )
                        transcript_file_id = result.get('id')
                    pass  # silent transcript sync
            except Exception as e:
                print(f"\n\u26a0\ufe0f [CLOUD ERROR]: {e}")
            finally:
                cloud_queue.task_done()

    async def cloud_vision_diary():
        """Background lobe: collects every 5 new webcam images, describes them with Gemini 1.5 Pro,
        and appends the entry to cloud_vision_memory.md. Never touches the Realtime WebSocket."""
        pending: list[str] = []
        _diary_backoff_until = 0.0

        while True:
            img_path = await cloud_vision_queue.get()
            pending.append(img_path)
            while not cloud_vision_queue.empty():
                try:
                    pending.append(cloud_vision_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(pending) < 5:
                continue

            # Skip diary if backed off
            if time.monotonic() < _diary_backoff_until:
                pending = []  # drain — stale frames not worth processing after backoff
                continue

            batch, pending = pending[:5], pending[5:]
            try:
                diary_content: list = [
                    {"type": "text", "text": (
                        "You are Weaver's subconscious visual memory system. "
                        "Describe these 5 sequential webcam images as a single cohesive memory log entry. "
                        "Note what you see, any changes between frames, and the apparent activity or context. "
                        "Be concise (3-5 sentences)."
                    )}
                ]
                for img_path in batch:
                    try:
                        with open(img_path, "rb") as fh:
                            img_b64_diary = base64.b64encode(fh.read()).decode("ascii")
                        diary_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64_diary}"}})
                    except Exception:
                        pass

                response = await openai_vision_client.chat.completions.create(
                    model=OPENAI_VISION_BEST,
                    messages=[{"role": "user", "content": diary_content}],
                    max_tokens=300,
                )
                description = (response.choices[0].message.content or "").strip()
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = f"\n## [{ts}]\n{description}\n"
                with open(cloud_vision_memory_path, "a", encoding="utf-8") as vf:
                    vf.write(entry)
                print(f"\n\U0001f441\ufe0f\u200d\U0001f5e8\ufe0f [VISION DIARY] Logged: {description[:70]}...", flush=True)
            except Exception as e:
                import re as _re2
                err_str2 = str(e)
                print(f"\n\u26a0\ufe0f [VISION DIARY ERROR]: {err_str2[:120]}", flush=True)
                _retry_match2 = _re2.search(r"retry in (\d+\.?\d*)", err_str2)
                if _retry_match2:
                    _diary_backoff_until = time.monotonic() + float(_retry_match2.group(1)) + 2.0
                elif "403" in err_str2 or "limit: 0" in err_str2:
                    _diary_backoff_until = time.monotonic() + 3600.0  # 1h for zero-quota models
                else:
                    _diary_backoff_until = time.monotonic() + 120.0

    print("[READY] Weaver is listening and watching...")

    async def _run_forever():
        realtime_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

        reconnect_count = 0
        while True:
            if reconnect_count > 0:
                print(f"\n[RECONNECT] Restarting Realtime session (attempt {reconnect_count})...", flush=True)
                await asyncio.sleep(2)
                while not _mic_queue.empty():
                    with contextlib.suppress(Exception):
                        _mic_queue.get_nowait()

            while not audio_out_queue.empty():
                with contextlib.suppress(Exception):
                    audio_out_queue.get_nowait()

            if os.path.exists(transcript_path):
                await cloud_queue.put({"type": "transcript", "path": transcript_path})

            context_text = _build_runtime_context()
            _greeted = [False]  # one-shot greeting per connection

            try:
                async with websockets.connect(
                    realtime_url,
                    additional_headers={
                        "Authorization": f"Bearer {voice_api_key}",
                        "OpenAI-Beta": "realtime=v1",
                    },
                    ping_interval=20,
                    ping_timeout=None,
                ) as ws:
                    print("[REALTIME] Connected to OpenAI Realtime.", flush=True)

                    await ws.send(json.dumps({
                        "type": "session.update",
                        "session": {
                            "modalities": ["audio", "text"],
                            "instructions": context_text,
                            "voice": "alloy",
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                            "input_audio_transcription": {
                                "model": "whisper-1"
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "threshold": 0.3,
                                "prefix_padding_ms": 200,
                                "silence_duration_ms": 500,
                                "create_response": True,
                            },
                            "tools": [{
                                "type": "function",
                                "name": "register_face",
                                "description": "Captures the current webcam frame, describes the person's appearance, and saves name+face to persistent memory.",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "The name of the person to register."
                                        }
                                    },
                                    "required": ["name"]
                                }
                            }],
                            "tool_choice": "auto",
                        }
                    }))
                    last_context_sent[0] = context_text
                    response_in_progress = [False]
                    last_context_injected_at = [0.0]
                    mic_hold_until = [0.0]
                    speech_started_at = [0.0]
                    register_face_pending = [False]

                    async def inject_native_vision():
                        """Describes webcam frames with Gemini 2.0 Flash and injects as silent input_text.
                        The Realtime API only supports input_text/input_audio — no image_url."""
                        last_inject_at = 0.0
                        _last_thumb = None
                        _vision_backoff = 0.0  # seconds to back off after errors
                        while True:
                            img_b64 = await realtime_vision_queue.get()
                            now = time.monotonic()
                            # Back off if we hit API errors
                            if _vision_backoff > 0 and (now - last_inject_at) < _vision_backoff:
                                continue
                            # Throttle to one visual update every 4 seconds
                            if (now - last_inject_at) < 4.0:
                                continue
                            if response_in_progress[0] or assistant_is_speaking[0]:
                                continue
                            # Scene-change detection: skip Gemini if frame looks similar to last
                            try:
                                import cv2 as _cv2_vis
                                _raw = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
                                _fg = _cv2_vis.imdecode(_raw, _cv2_vis.IMREAD_GRAYSCALE)
                                if _fg is not None:
                                    _thumb = _cv2_vis.resize(_fg, (16, 16)).astype(np.float32)
                                    if _last_thumb is not None and float(np.mean(np.abs(_thumb - _last_thumb))) < 8.0:
                                        continue  # scene unchanged — skip Gemini API call
                                    _last_thumb = _thumb
                            except Exception:
                                pass
                            try:
                                # Build face ID context from local ArcFace tracker
                                face_id_lines = []
                                now_mono = time.monotonic()
                                for fname, fdata in list(face_state.items()):
                                    if fname.startswith("_"):
                                        continue
                                    age = now_mono - fdata.get("last_seen", 0)
                                    if age < 3.0:
                                        conf = fdata.get("score", 0)
                                        face_id_lines.append(f"{fname} (conf={conf:.2f})")
                                body_count = face_state.get("_body_count", 0)

                                known_names = [n for n in face_id_lines if "unregistered" not in n]
                                if known_names:
                                    # ArcFace confirmed identity — tell GPT directly, no guessing needed
                                    names_str = ", ".join(n.split("(")[0].strip() for n in known_names)
                                    vision_text = (
                                        f"The person in frame is {names_str} (face confirmed by ArcFace). "
                                        "Describe what they are doing or their current state in one sentence."
                                    )
                                elif face_id_lines:
                                    # Face detected but not in registry
                                    vision_text = (
                                        "An unregistered person is visible. "
                                        "Describe their appearance and activity in one sentence."
                                    )
                                elif body_count > 0:
                                    vision_text = (
                                        f"{body_count} person(s) detected. "
                                        "Describe the webcam scene in one sentence."
                                    )
                                elif people_memory[0]:
                                    vision_text = (
                                        f"Known people:\n{people_memory[0]}\n\n"
                                        "Describe the webcam scene in one sentence. "
                                        "If someone is visible and matches a known person, name them."
                                    )
                                else:
                                    vision_text = "Describe the webcam scene in one sentence."
                                _vision_resp = await openai_vision_client.chat.completions.create(
                                    model=OPENAI_VISION_FAST,
                                    messages=[{"role": "user", "content": [
                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                                        {"type": "text", "text": vision_text},
                                    ]}],
                                    max_tokens=100,
                                )
                                description = (_vision_resp.choices[0].message.content or "").strip()
                                with contextlib.suppress(Exception):
                                    await ws.send(json.dumps({
                                        "type": "conversation.item.create",
                                        "item": {
                                            "type": "message",
                                            "role": "user",
                                            "content": [{"type": "input_text", "text": f"[Visual context: {description}]"}]
                                        }
                                    }))
                                last_inject_at = time.monotonic()
                                _vision_backoff = 0.0  # reset on success
                                print(f"\n[VISION] {description}", flush=True)
                            except Exception as e:
                                err_str = str(e)
                                print(f"\n\u26a0\ufe0f [VISION ERROR]: {err_str[:120]}", flush=True)
                                # Parse retry-after from 429 message, else use exponential backoff
                                import re as _re
                                _retry_match = _re.search(r"retry in (\d+\.?\d*)", err_str)
                                if _retry_match:
                                    _vision_backoff = float(_retry_match.group(1)) + 2.0
                                elif "403" in err_str or "PERMISSION_DENIED" in err_str:
                                    _vision_backoff = 300.0  # 5 min for hard denials
                                else:
                                    _vision_backoff = min((_vision_backoff or 10.0) * 2, 300.0)
                                last_inject_at = time.monotonic()

                    async def send_mic():
                        _mic_logged = False
                        _probe_at = [time.monotonic()]
                        _chunks_sent = [0]
                        while True:
                            try:
                                data = await asyncio.wait_for(_mic_queue.get(), timeout=2.0)
                            except asyncio.TimeoutError:
                                print("\n[WARNING] Microphone stream timeout — no audio.", flush=True)
                                continue
                            rms = _pcm16_rms(data)
                            if heartbeat:
                                print("." if rms > RMS_THRESHOLD else "_", end="", flush=True)

                            # 5-second diagnostic probe
                            if time.monotonic() - _probe_at[0] > 5.0:
                                _probe_at[0] = time.monotonic()
                                hold_left = max(0.0, mic_hold_until[0] - time.monotonic())
                                audio_g = max(0.0, 0.8 - (time.monotonic() - last_audio_chunk_at[0]))
                                print(f"[MIC PROBE] speaking={assistant_is_speaking[0]} "
                                      f"mic_q={_mic_queue.qsize()} "
                                      f"hold_left={hold_left:.1f}s "
                                      f"audio_guard={audio_g:.1f}s "
                                      f"chunks_sent={_chunks_sent[0]} rms={rms}", flush=True)

                            # Mute mic while Weaver is playing audio (speaker bleed protection)
                            if assistant_is_speaking[0] or not audio_out_queue.empty():
                                continue
                            if time.monotonic() < mic_hold_until[0]:
                                continue
                            # Reduced tail guard: faster mic re-enable (was 2.0s)
                            if (time.monotonic() - last_audio_chunk_at[0]) < 0.8:
                                continue

                            if not _mic_logged:
                                print(f"[MIC] Streaming audio to OpenAI... (RMS={rms})", flush=True)
                                _mic_logged = True

                            last_user_activity_at[0] = time.monotonic()
                            b64 = base64.b64encode(data).decode("ascii")
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": b64,
                            }))
                            _chunks_sent[0] += 1

                    def _log_assistant_message(content: str) -> None:
                        content = content.strip()
                        if not content or _is_duplicate_message(last_assistant_message, content):
                            return
                        last_assistant_activity_at[0] = time.monotonic()
                        print(f"\n[WEAVER]: {content}", flush=True)
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        with open(transcript_path, "a") as tf:
                            tf.write(f"[{ts}] WEAVER: {content}\n")
                        brain_queue.put_nowait(AIMessage(content=content))

                    async def receive_realtime():
                        assistant_text_buffer = ""
                        _first_audio_logged = False
                        _turn_speech_started = 0.0
                        _response_start_at = 0.0
                        try:
                            async for raw_msg in ws:
                                msg = json.loads(raw_msg)
                                msg_type = msg.get("type", "")

                                if msg_type in ("session.created", "session.updated"):
                                    td = msg.get("session", {}).get("turn_detection") or "none"
                                    print(f"[SESSION] {msg_type} — VAD: {td}", flush=True)
                                    if msg_type == "session.updated" and not _greeted[0]:
                                        _greeted[0] = True
                                        await ws.send(json.dumps({
                                            "type": "response.create",
                                            "response": {
                                                "modalities": ["audio", "text"],
                                                "instructions": "Greet the user with a single short sentence in English only to confirm you are online."
                                            }
                                        }))
                                        print("[WEAVER] Triggering greeting...", flush=True)

                                elif msg_type == "response.created":
                                    assistant_text_buffer = ""
                                    response_in_progress[0] = True
                                    assistant_is_speaking[0] = True
                                    _turn_speech_started = speech_started_at[0]
                                    _response_start_at = time.monotonic()
                                    _first_audio_logged = False
                                    if _turn_speech_started > 0:
                                        print(f"[LATENCY] speech→response_start: {(_response_start_at - _turn_speech_started)*1000:.0f}ms", flush=True)
                                    with contextlib.suppress(Exception):
                                        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

                                elif msg_type == "response.audio.delta":
                                    delta = msg.get("delta")
                                    if not delta:
                                        continue
                                    audio_bytes = base64.b64decode(delta)
                                    audio_out_queue.put_nowait(audio_bytes)
                                    assistant_is_speaking[0] = True
                                    last_assistant_activity_at[0] = time.monotonic()
                                    if not _first_audio_logged and _turn_speech_started > 0:
                                        print(f"[LATENCY] speech→first_audio: {(time.monotonic()-_turn_speech_started)*1000:.0f}ms  ← target<500ms", flush=True)
                                        _first_audio_logged = True
                                    if heartbeat:
                                        print("\U0001f50a", end="", flush=True)

                                elif msg_type in (
                                    "conversation.item.input_audio_transcription.completed",
                                    "input_audio_transcription.completed",
                                ):
                                    content = (msg.get("transcript") or "").strip()
                                    if content:
                                        _stage_user_message(content)

                                elif msg_type == "conversation.item.created":
                                    item = msg.get("item", {})
                                    if item.get("role") == "user":
                                        for part in item.get("content", []):
                                            if not isinstance(part, dict):
                                                continue
                                            transcript = (part.get("transcript") or "").strip()
                                            if transcript:
                                                _stage_user_message(transcript)

                                elif msg_type in (
                                    "response.audio_transcript.delta",
                                    "response.text.delta",
                                    "response.output_text.delta",
                                ):
                                    delta = msg.get("delta", "")
                                    if delta:
                                        assistant_text_buffer += delta
                                        assistant_is_speaking[0] = True
                                        last_assistant_activity_at[0] = time.monotonic()

                                elif msg_type in (
                                    "response.audio_transcript.done",
                                    "response.text.done",
                                    "response.output_text.done",
                                ):
                                    finalized_text = (
                                        msg.get("transcript")
                                        or msg.get("text")
                                        or assistant_text_buffer
                                    )
                                    _flush_pending_user_message(force=True)
                                    _log_assistant_message(finalized_text)
                                    assistant_text_buffer = ""

                                elif msg_type == "input_audio_buffer.speech_started":
                                    speech_started_at[0] = time.monotonic()
                                    print("[VAD] 🎤 Speech detected!", flush=True)
                                    if assistant_is_speaking[0]:
                                        _flush_pending_user_message(force=True)
                                        assistant_is_speaking[0] = False
                                        with contextlib.suppress(Exception):
                                            await ws.send(json.dumps({"type": "response.cancel"}))
                                        while not audio_out_queue.empty():
                                            with contextlib.suppress(Exception):
                                                audio_out_queue.get_nowait()
                                        print("\n[INTERRUPTED]", flush=True)

                                elif msg_type == "response.output_item.done":
                                    item = msg.get("item", {})
                                    if item.get("type") == "function_call" and item.get("name") == "register_face":
                                        call_id = item.get("call_id", "")
                                        raw_args = item.get("arguments", "{}")
                                        try:
                                            reg_name = json.loads(raw_args).get("name", "unknown")
                                        except Exception:
                                            reg_name = "unknown"
                                        # Grab the latest webcam frame
                                        pattern = os.path.join(images_dir, "memory_*.jpg")
                                        frame_files = sorted(glob.glob(pattern), reverse=True)
                                        reg_result = "No camera frame available."
                                        if frame_files:
                                            try:
                                                with open(frame_files[0], "rb") as fh:
                                                    frame_b64 = base64.b64encode(fh.read()).decode("ascii")

                                                # 1) ArcFace embedding — reuse already-loaded face_app
                                                import cv2 as _cv2
                                                import warnings as _warnings
                                                # Prefer last frame where tracker confirmed a face
                                                reg_frame = face_last_face_frame[0]
                                                if reg_frame is None:
                                                    reg_frame = face_tracker_frame[0]
                                                if reg_frame is None:
                                                    reg_frame = _cv2.imread(frame_files[0])
                                                if reg_frame is None:
                                                    print(f"\n\u26a0\ufe0f [REGISTER] No frame available for {reg_name}.", flush=True)
                                                elif face_app_ref[0] is None:
                                                    print(f"\n\u26a0\ufe0f [REGISTER] Face model not loaded yet — try again in a moment.", flush=True)
                                                else:
                                                    try:
                                                        _app = face_app_ref[0]
                                                        def _reg_embed():
                                                            with _warnings.catch_warnings():
                                                                _warnings.simplefilter("ignore")
                                                                return _app.get(reg_frame)
                                                        reg_faces = await asyncio.to_thread(_reg_embed)
                                                        if reg_faces:
                                                            raw_e = reg_faces[0].embedding.astype(np.float32)
                                                            emb = raw_e / (np.linalg.norm(raw_e) + 1e-9)
                                                            existing_embs = face_registry.get(reg_name, [])
                                                            existing_embs.append(emb)
                                                            face_registry[reg_name] = existing_embs
                                                            save_dict = {k: np.array(v) for k, v in face_registry.items()}
                                                            np.savez(face_registry_path, **save_dict)
                                                            print(f"\n\U0001f9d1 [REGISTER] ArcFace embedding saved for {reg_name} ✓", flush=True)
                                                        else:
                                                            print(f"\n\u26a0\ufe0f [REGISTER] No face detected in frame for {reg_name} — look at camera and try again.", flush=True)
                                                    except Exception as emb_exc:
                                                        print(f"\n\u26a0\ufe0f [REGISTER] Embedding error: {emb_exc}", flush=True)

                                                # 2) OpenAI appearance description for people_memory
                                                _appear_resp = await openai_vision_client.chat.completions.create(
                                                    model=OPENAI_VISION_BEST,
                                                    messages=[{"role": "user", "content": [
                                                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}},
                                                        {"type": "text", "text": (
                                                            "Describe the physical appearance of the person visible in this "
                                                            "webcam image. Focus on: hair colour and style, skin tone, "
                                                            "eye colour if visible, build, approximate age, and any "
                                                            "distinguishing features. Be specific and objective."
                                                        )},
                                                    ]}],
                                                    max_tokens=200,
                                                )
                                                appearance = (_appear_resp.choices[0].message.content or "").strip()
                                                existing = people_memory[0]
                                                new_entry = f"- **{reg_name}** \u2014 {appearance}"
                                                if reg_name.lower() in existing.lower():
                                                    lines = [l if reg_name.lower() not in l.lower() else new_entry for l in existing.splitlines()]
                                                    people_memory[0] = "\n".join(lines)
                                                else:
                                                    people_memory[0] = (existing + "\n" + new_entry).strip()
                                                with open(people_memory_path, "w", encoding="utf-8") as _pm:
                                                    _pm.write(people_memory[0])
                                                reg_result = f"Registered {reg_name}: {appearance}"
                                                print(f"\n\U0001f9d1 [REGISTER] {reg_result[:80]}", flush=True)
                                            except Exception as exc:
                                                reg_result = f"Registration error: {exc}"
                                        with contextlib.suppress(Exception):
                                            await ws.send(json.dumps({
                                                "type": "conversation.item.create",
                                                "item": {
                                                    "type": "function_call_output",
                                                    "call_id": call_id,
                                                    "output": reg_result
                                                }
                                            }))
                                        register_face_pending[0] = True

                                elif msg_type == "input_audio_buffer.committed":
                                    # create_response:True — server auto-fires; just sync local state
                                    response_in_progress[0] = True
                                    speech_started_at[0] = 0.0

                                elif msg_type == "response.done":
                                    if register_face_pending[0]:
                                        register_face_pending[0] = False
                                        response_in_progress[0] = True
                                        with contextlib.suppress(Exception):
                                            await ws.send(json.dumps({
                                                "type": "response.create",
                                                "response": {"modalities": ["audio", "text"]}
                                            }))
                                        continue
                                    if assistant_text_buffer.strip():
                                        _flush_pending_user_message(force=True)
                                        _log_assistant_message(assistant_text_buffer)
                                        assistant_text_buffer = ""
                                    _flush_pending_user_message(force=True)
                                    assistant_is_speaking[0] = False
                                    response_in_progress[0] = False
                                    last_turn_complete_at[0] = time.monotonic()
                                    if _response_start_at > 0:
                                        print(f"[LATENCY] response_duration: {(time.monotonic()-_response_start_at)*1000:.0f}ms", flush=True)
                                    _response_start_at = 0.0
                                    _turn_speech_started = 0.0
                                    mic_hold_until[0] = time.monotonic() + 1.5
                                    with contextlib.suppress(Exception):
                                        await ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                                    pass  # turn complete — no noise

                                elif msg_type == "error":
                                    error_obj = msg.get("error")
                                    if isinstance(error_obj, dict):
                                        error_text = error_obj.get("message", str(error_obj))
                                    elif error_obj:
                                        error_text = str(error_obj)
                                    else:
                                        error_text = str(msg)
                                    print(f"\n[REALTIME ERROR]: {error_text}", flush=True)
                                    lowered = error_text.lower()
                                    if "buffer too small" in lowered:
                                        continue
                                    if "active response in progress" in lowered:
                                        response_in_progress[0] = True
                                        continue
                                    if "cancellation failed" in lowered or "no active response" in lowered:
                                        continue
                                    if "image_url" in lowered or "input_image" in lowered:
                                        continue
                                    response_in_progress[0] = False
                                    raise RuntimeError(error_text)
                        finally:
                            _flush_pending_user_message(force=True)

                    async def langchain_cortex():
                        """LangChain brain: summarizes conversation (+ vision diary) and refreshes realtime instructions."""
                        SUMMARIZE_EVERY = 8
                        new_count = 0

                        while True:
                            await asyncio.sleep(3)
                            _flush_pending_user_message(force=False)

                            while not brain_queue.empty():
                                try:
                                    lc_history.append(brain_queue.get_nowait())
                                    new_count += 1
                                except asyncio.QueueEmpty:
                                    break

                            if new_count >= SUMMARIZE_EVERY and lc_history:
                                try:
                                    # Read last ~4000 chars of the vision diary for subconscious context
                                    vision_snippet = ""
                                    if os.path.exists(cloud_vision_memory_path):
                                        try:
                                            with open(cloud_vision_memory_path, "r", encoding="utf-8") as vf:
                                                vision_snippet = vf.read()[-4000:]
                                        except Exception:
                                            pass

                                    prompt = [
                                        SystemMessage(content=(
                                            "You are Weaver's memory cortex. Summarize the conversation below "
                                            "into a concise paragraph. Preserve: names, key facts, emotional tone, "
                                            "promises, and anything the user asked you to remember. "
                                            "If there is a previous summary, integrate new info into it. "
                                            "Also integrate any relevant visual context from the subconscious visual diary."
                                        )),
                                    ]
                                    if lc_summary[0]:
                                        prompt.append(HumanMessage(content=f"Previous summary:\n{lc_summary[0]}"))
                                    if vision_snippet:
                                        prompt.append(HumanMessage(content=f"Recent Subconscious Visual Memories:\n{vision_snippet}"))
                                    transcript_block = "\n".join(
                                        f"{'User' if isinstance(message, HumanMessage) else 'Weaver'}: {message.content}"
                                        for message in lc_history
                                    )
                                    prompt.append(HumanMessage(content=f"New messages:\n{transcript_block}"))

                                    result = await lc_llm.ainvoke(prompt)
                                    lc_summary[0] = result.content.strip()
                                    lc_history.clear()
                                    new_count = 0
                                    print(f"\n\U0001f9e0 [LANGCHAIN] Memory updated: {lc_summary[0][:80]}...", flush=True)

                                    # Extract people/names from the conversation and persist
                                    try:
                                        people_prompt = [
                                            SystemMessage(content=(
                                                "You are Weaver's people memory. Extract every person mentioned "
                                                "in the conversation: their name, relationship to the user, "
                                                "physical appearance (hair colour, skin tone, build, distinguishing features), "
                                                "and any key facts (job, personality, topics discussed). "
                                                "Appearance notes are critical — they allow face recognition in future camera frames. "
                                                "If the existing list already has an entry, merge and update it. "
                                                "Return ONLY the updated people list in markdown bullet format. "
                                                "If no new people are mentioned, return the existing list unchanged."
                                            )),
                                        ]
                                        if people_memory[0]:
                                            people_prompt.append(HumanMessage(content=f"Existing people list:\n{people_memory[0]}"))
                                        people_prompt.append(HumanMessage(content=f"Conversation:\n{transcript_block}"))
                                        people_result = await lc_llm.ainvoke(people_prompt)
                                        updated = people_result.content.strip()
                                        if updated:
                                            people_memory[0] = updated
                                            with open(people_memory_path, "w", encoding="utf-8") as _pm:
                                                _pm.write(updated)
                                            print(f"\n\U0001f9d1 [PEOPLE] Memory updated.", flush=True)
                                    except Exception:
                                        pass
                                except Exception as e:
                                    print(f"\n\u26a0\ufe0f [LANGCHAIN ERROR]: {e}", flush=True)

                            if response_in_progress[0]:
                                continue
                            if assistant_is_speaking[0]:
                                if (time.monotonic() - last_assistant_activity_at[0]) < 2.0:
                                    continue
                                assistant_is_speaking[0] = False

                            if (time.monotonic() - last_turn_complete_at[0]) < 2.0:
                                continue

                            if (time.monotonic() - last_context_injected_at[0]) < 25.0:
                                continue

                            context_text = _build_runtime_context()
                            if context_text == last_context_sent[0]:
                                continue

                            try:
                                await ws.send(json.dumps({
                                    "type": "session.update",
                                    "session": {"instructions": context_text},
                                }))
                                last_context_sent[0] = context_text
                                last_context_injected_at[0] = time.monotonic()
                            except Exception:
                                pass

                    async with asyncio.TaskGroup() as tg:
                        tg.create_task(inject_native_vision())
                        tg.create_task(send_mic())
                        tg.create_task(receive_realtime())
                        tg.create_task(langchain_cortex())

            except* (Exception,) as eg:
                reconnect_count += 1
                print(f"\n[SESSION DROP] {eg.exceptions[0]}. Reconnecting...", flush=True)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(play_audio())
            tg.create_task(capture_video())
            tg.create_task(face_tracker_task())
            tg.create_task(cloud_worker())
            tg.create_task(cloud_vision_diary())
            tg.create_task(_run_forever())
    except asyncio.CancelledError:
        pass
    finally:
        with contextlib.suppress(Exception):
            _pacat_reader.cancel()
            _pacat_proc.terminate()
        await _terminate_process(aplay_proc_ref[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic real-time VTV stream")
    parser.add_argument(
        "--heartbeat",
        dest="heartbeat",
        action="store_true",
        default=False,
        help="Print dots/markers while microphone and speaker chunks are streamed (default: off)",
    )
    parser.add_argument(
        "--no-heartbeat",
        dest="heartbeat",
        action="store_false",
        help="Disable microphone heartbeat dots",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_vtv(heartbeat=args.heartbeat))
    except KeyboardInterrupt:
        os.system("pkill -f arecord")
        os.system("pkill -f aplay")
    finally:
        os.system("pkill -f arecord")
        os.system("pkill -f aplay")
