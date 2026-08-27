"""Weaver voice server — AWS Polly or OpenVoice TTS on 127.0.0.1:8092.

Default path: Amazon Polly generative speech, which is much more natural than
browser speech and avoids OpenAI. Set TTS_PROVIDER=openvoice to use the older
local clone path.

OpenVoice clone-ONCE design (the reference wav is processed a single time, ever):
  * First boot: extracts the speaker tone-color embedding from
    ~/tts/weaver_voice_ref.wav and saves it to ~/tts/weaver_voice_se.pth.
    Every later boot loads the saved file — the wav is never touched again.
  * Every synthesized line is cached to disk keyed by sha1(text): a repeated
    line (the wake greeting, stock phrases) replays from file with zero compute.

Endpoints (Caddy exposes this as /tts/* behind the X-Weaver-Key gate):
  GET  /health          → {"status": "ok", "cached": N}
  POST /synth {"text"}  → audio/mpeg by default, or audio/wav for OpenVoice

Runs as the weaver-tts systemd unit. 2 torch threads — the box has 2 vCPUs
shared with the LLM; do not raise this.
"""
import hashlib
import os
import threading
import time

# Config via env so ONE file serves both hosts:
#   AWS t4g box  → all defaults (cpu, 127.0.0.1:8092, ~/tts, no key — Caddy gates it)
#   RunPod GPU   → TTS_DEVICE=cuda TTS_HOST=0.0.0.0 TTS_PORT=8888 TTS_HOME=/workspace/tts
#                  WEAVER_TTS_KEY=<key>  (the RunPod proxy URL is public → self-gate)
DEVICE = os.getenv("TTS_DEVICE", "cpu")
HOST = os.getenv("TTS_HOST", "127.0.0.1")
PORT = int(os.getenv("TTS_PORT", "8092"))
KEY = os.getenv("WEAVER_TTS_KEY", "")  # if set, require X-Weaver-Key header
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "azure").strip().lower()
TTS_FALLBACK_PROVIDER = os.getenv("TTS_FALLBACK_PROVIDER", "polly").strip().lower()

# Azure Speech Services (primary TTS provider for Azure deployment).
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY", os.getenv("AZURE_OPENAI_KEY", ""))
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
AZURE_SPEECH_VOICE = os.getenv("AZURE_SPEECH_VOICE", "en-US-AriaNeural")
AZURE_SPEECH_STYLE = os.getenv("AZURE_SPEECH_STYLE", "warm")
AZURE_OUTPUT_FORMAT = os.getenv("AZURE_TTS_OUTPUT_FORMAT", "riff-24khz-16bit-mono-pcm")

# Amazon Polly (fallback TTS provider).
POLLY_REGION = (
    os.getenv("TTS_AWS_REGION")
    or os.getenv("AWS_REGION")
    or os.getenv("AWS_DEFAULT_REGION")
    or "us-east-1"
)
POLLY_ENGINE = os.getenv("TTS_POLLY_ENGINE", "generative")
POLLY_VOICE = os.getenv("TTS_POLLY_VOICE", "Ruth")
POLLY_OUTPUT_FORMAT = os.getenv("TTS_POLLY_OUTPUT_FORMAT", "mp3")
POLLY_SAMPLE_RATE = os.getenv("TTS_POLLY_SAMPLE_RATE", "")
POLLY_LANGUAGE_CODE = os.getenv("TTS_POLLY_LANGUAGE_CODE", "")
TTS_CACHE_MAX_FILES = int(os.getenv("TTS_CACHE_MAX_FILES", "1500"))

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

HOME = os.getenv("TTS_HOME", os.path.expanduser("~/tts"))
REF_WAV = os.path.join(HOME, "weaver_voice_ref.wav")
SAVED_SE = os.path.join(HOME, "weaver_voice_se.pth")
CKPT_DIR = os.path.join(HOME, "OpenVoiceV2")
CACHE = os.path.join(HOME, "cache")
os.makedirs(CACHE, exist_ok=True)

app = FastAPI()


class Req(BaseModel):
    text: str


SYNTH_LOCK = threading.Lock()   # 2-core box: serialize synthesis; also guards tmp files
_polly_client = None
_openvoice = None
_cache_locks: dict[str, threading.Lock] = {}
_cache_locks_guard = threading.Lock()

MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "ogg_vorbis": "audio/ogg",
    "ogg_opus": "audio/ogg",
    "pcm": "audio/pcm",
    "wav": "audio/wav",
}
EXTENSIONS = {
    "mp3": "mp3",
    "ogg_vorbis": "ogg",
    "ogg_opus": "ogg",
    "pcm": "pcm",
    "wav": "wav",
}


def provider_chain() -> list[str]:
    providers = []
    for value in (TTS_PROVIDER, TTS_FALLBACK_PROVIDER):
        value = (value or "").strip().lower()
        if value in ("aws", "aws-polly"):
            value = "polly"
        if value in ("azure", "azure-speech"):
            value = "azure"
        if value and value != "none" and value not in providers:
            providers.append(value)
    return providers or ["azure"]


def cache_count() -> int:
    return len([
        f for f in os.listdir(CACHE)
        if f.rsplit(".", 1)[-1].lower() in {"mp3", "ogg", "pcm", "wav"}
        and not f.startswith("_")
    ])


def cache_files() -> list[str]:
    return [
        os.path.join(CACHE, f)
        for f in os.listdir(CACHE)
        if f.rsplit(".", 1)[-1].lower() in {"mp3", "ogg", "pcm", "wav"}
        and not f.startswith("_")
    ]


def cache_identity(provider: str, text: str) -> tuple[str, str, str]:
    if provider == "polly":
        audio_format = POLLY_OUTPUT_FORMAT
        media_type = MEDIA_TYPES.get(audio_format, "audio/mpeg")
        identity = "|".join([
            provider,
            POLLY_REGION,
            POLLY_ENGINE,
            POLLY_VOICE,
            audio_format,
            POLLY_SAMPLE_RATE,
            POLLY_LANGUAGE_CODE,
            text.lower(),
        ])
        return identity, EXTENSIONS.get(audio_format, "mp3"), media_type
    if provider == "azure":
        identity = "|".join([provider, AZURE_SPEECH_REGION, AZURE_SPEECH_VOICE, AZURE_SPEECH_STYLE, AZURE_OUTPUT_FORMAT, text.lower()])
        return identity, "wav", "audio/wav"
    if provider == "openvoice":
        identity = "|".join([provider, DEVICE, "openvoice-v2", text.lower()])
        return identity, "wav", "audio/wav"
    raise ValueError(f"unsupported tts provider: {provider}")


def cache_path(provider: str, text: str) -> tuple[str, str]:
    identity, ext, _ = cache_identity(provider, text)
    key = hashlib.sha1(identity.encode()).hexdigest()
    return os.path.join(CACHE, f"{key}.{ext}"), key


def cache_lock(path: str) -> threading.Lock:
    with _cache_locks_guard:
        lock = _cache_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _cache_locks[path] = lock
        return lock


def prune_cache() -> None:
    if TTS_CACHE_MAX_FILES <= 0:
        return
    files = cache_files()
    excess = len(files) - TTS_CACHE_MAX_FILES
    if excess <= 0:
        return
    for path in sorted(files, key=lambda p: os.path.getmtime(p))[:excess]:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


_azure_speech_synthesizer = None


def azure_speech_synthesizer():
    global _azure_speech_synthesizer
    if _azure_speech_synthesizer is None:
        import azure.cognitiveservices.speech as speechsdk
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        _azure_speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    return _azure_speech_synthesizer


def synth_azure(text: str) -> tuple[bytes, str]:
    try:
        ssml = (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{AZURE_SPEECH_VOICE}">'
            f'<mstts:express-as style="{AZURE_SPEECH_STYLE}">'
            f'{text}</mstts:express-as></voice></speak>'
        )
        result = azure_speech_synthesizer().speak_ssml_async(ssml).get()
        if result.reason.name == "SynthesizingAudioCompleted":
            return result.audio_data, "audio/wav"
        raise RuntimeError(f"Azure TTS failed: {result.reason}")
    except Exception as exc:
        raise RuntimeError(f"Azure Speech synthesis error: {exc}")


def polly_client():
    global _polly_client
    if _polly_client is None:
        import boto3

        _polly_client = boto3.client("polly", region_name=POLLY_REGION)
    return _polly_client


def synth_polly(text: str) -> tuple[bytes, str]:
    request = {
        "Text": text,
        "TextType": "text",
        "OutputFormat": POLLY_OUTPUT_FORMAT,
        "VoiceId": POLLY_VOICE,
        "Engine": POLLY_ENGINE,
    }
    if POLLY_SAMPLE_RATE:
        request["SampleRate"] = POLLY_SAMPLE_RATE
    if POLLY_LANGUAGE_CODE:
        request["LanguageCode"] = POLLY_LANGUAGE_CODE
    response = polly_client().synthesize_speech(**request)
    stream = response.get("AudioStream")
    if stream is None:
        raise RuntimeError("Polly response did not include AudioStream")
    try:
        audio = stream.read()
    finally:
        stream.close()
    if not audio:
        raise RuntimeError("Polly returned empty audio")
    return audio, response.get("ContentType") or MEDIA_TYPES.get(POLLY_OUTPUT_FORMAT, "audio/mpeg")


def openvoice_runtime():
    global _openvoice
    if _openvoice is not None:
        return _openvoice

    import torch
    from melo.api import TTS
    from openvoice.api import ToneColorConverter

    if DEVICE == "cpu":
        torch.set_num_threads(2)  # 2-vCPU box; irrelevant on GPU

    print("[tts] loading MeloTTS EN…", flush=True)
    base = TTS(language="EN", device=DEVICE)
    speaker_id = base.hps.data.spk2id["EN-US"]

    print("[tts] loading tone-color converter…", flush=True)
    try:
        converter = ToneColorConverter(
            os.path.join(CKPT_DIR, "converter", "config.json"),
            device=DEVICE,
            enable_watermark=False,
        )
    except TypeError:  # older signature without the kwarg
        converter = ToneColorConverter(os.path.join(CKPT_DIR, "converter", "config.json"), device=DEVICE)
        converter.watermark_model = None
    converter.load_ckpt(os.path.join(CKPT_DIR, "converter", "checkpoint.pth"))
    source_se = torch.load(os.path.join(CKPT_DIR, "base_speakers", "ses", "en-us.pth"), map_location=DEVICE)

    # clone ONCE: saved embedding wins; extract only if it doesn't exist yet
    if os.path.exists(SAVED_SE):
        target_se = torch.load(SAVED_SE, map_location=DEVICE)
        print(f"[tts] loaded saved voice embedding {SAVED_SE}", flush=True)
    else:
        print(f"[tts] first boot: extracting voice embedding from {REF_WAV}…", flush=True)
        target_se = converter.extract_se([REF_WAV])
        torch.save(target_se, SAVED_SE)
        print(f"[tts] saved voice embedding → {SAVED_SE} (the wav won't be processed again)", flush=True)

    _openvoice = (base, speaker_id, converter, source_se, target_se)
    return _openvoice


def synth_openvoice(text: str) -> tuple[bytes, str]:
    with SYNTH_LOCK:
        base, speaker_id, converter, source_se, target_se = openvoice_runtime()
        tmp_base = os.path.join(CACHE, "_base_tmp.wav")
        tmp_out = os.path.join(CACHE, "_out_tmp.wav")
        base.tts_to_file(text, speaker_id, tmp_base, speed=1.0)
        converter.convert(audio_src_path=tmp_base, src_se=source_se, tgt_se=target_se,
                          output_path=tmp_out)
        with open(tmp_out, "rb") as f:
            return f.read(), "audio/wav"


def synth_provider(provider: str, text: str) -> tuple[bytes, str]:
    if provider == "azure":
        return synth_azure(text)
    if provider == "polly":
        return synth_polly(text)
    if provider == "openvoice":
        return synth_openvoice(text)
    raise ValueError(f"unsupported tts provider: {provider}")


def cache_put(path: str, audio: bytes):
    """Atomic write — a crash mid-synth must never leave a 0-byte cache entry."""
    tmp = path + ".part"
    with open(tmp, "wb") as f:
        f.write(audio)
    os.replace(tmp, path)
    prune_cache()


def cache_get(path: str):
    if os.path.exists(path) and os.path.getsize(path) > 44:   # >small header = real audio
        with open(path, "rb") as f:
            return f.read()
    if os.path.exists(path):
        os.remove(path)                                        # purge poisoned entry
    return None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "provider": TTS_PROVIDER,
        "fallback_provider": TTS_FALLBACK_PROVIDER,
        "cached": cache_count(),
        "azure": {
            "region": AZURE_SPEECH_REGION,
            "voice": AZURE_SPEECH_VOICE,
            "style": AZURE_SPEECH_STYLE,
        },
        "polly": {
            "region": POLLY_REGION,
            "engine": POLLY_ENGINE,
            "voice": POLLY_VOICE,
            "format": POLLY_OUTPUT_FORMAT,
        },
    }


@app.post("/synth")
def synth(req: Req, x_weaver_key: str | None = Header(default=None, alias="X-Weaver-Key")):
    if KEY and x_weaver_key != KEY:
        raise HTTPException(status_code=403, detail="invalid voice key")
    text = req.text.strip()[:800]
    if not text:
        return Response(status_code=400)
    errors = []
    for provider in provider_chain():
        path, key = cache_path(provider, text)
        _, _, cached_media_type = cache_identity(provider, text)
        cached = cache_get(path)                   # replay from file — zero compute
        if cached:
            return Response(cached, media_type=cached_media_type)
        t0 = time.time()
        try:
            with cache_lock(path):
                cached = cache_get(path)
                if cached:
                    return Response(cached, media_type=cached_media_type)
                audio, media_type = synth_provider(provider, text)
                cache_put(path, audio)
            print(f"[tts] {provider} synth {len(text)} chars in {time.time()-t0:.1f}s → cached {key[:8]}", flush=True)
            return Response(audio, media_type=media_type)
        except Exception as exc:
            msg = f"{provider}: {str(exc)[:180]}"
            errors.append(msg)
            print(f"[tts] provider failed: {msg}", flush=True)
    raise HTTPException(status_code=502, detail="; ".join(errors)[:500] or "tts failed")


def warm_cache():
    """Pre-render fixed lines without delaying process startup."""
    for line in ("I'm here.", "Hey. I can see you now."):
        provider = provider_chain()[0]
        path, _ = cache_path(provider, line)
        if cache_get(path) is not None:
            continue
        try:
            audio, _ = synth_provider(provider, line)
            cache_put(path, audio)
            print(f"[tts] warmed {provider}: {line!r}", flush=True)
        except Exception as e:
            print(f"[tts] warmup failed ({provider}): {e}", flush=True)


@app.on_event("startup")
def start_warmup():
    threading.Thread(target=warm_cache, name="tts-warmup", daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
