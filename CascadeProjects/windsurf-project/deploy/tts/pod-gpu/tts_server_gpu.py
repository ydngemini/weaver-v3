#!/usr/bin/env python3
"""Weaver GPU voice — XTTS-v2 cloned voice on 127.0.0.1:8092 (RunPod RTX 4090).

Clone-once: speaker latents are computed a single time from the reference wav and
reused for every line. /synth {text} -> audio/wav ; /health. Per-line disk cache
by sha1(text) with atomic writes. Same wire contract as the CPU OpenVoice server,
so embodiment.html needs no change — it just gets a faster, better voice.
"""
import hashlib
import io
import os
import time

os.environ.setdefault("COQUI_TOS_AGREED", "1")   # accept XTTS license non-interactively

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

REF = "/root/weaver_voice_ref.wav"
CACHE = "/root/tts/cache"
os.makedirs(CACHE, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 24000
KEY = os.getenv("WEAVER_TTS_KEY", "")  # RunPod proxy URL is public -> gate /synth

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

print("[tts] locating/downloading XTTS-v2 ...", flush=True)
model_path, config_path, _ = ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=False)
model.to(DEVICE)
model.eval()
print(f"[tts] XTTS-v2 loaded on {DEVICE}", flush=True)

print("[tts] cloning her voice from the reference (once) ...", flush=True)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[REF])
print("[tts] voice latents ready", flush=True)

app = FastAPI()


class Req(BaseModel):
    text: str


def synth(text: str) -> bytes:
    with torch.no_grad():
        out = model.inference(
            text, "en", gpt_cond_latent, speaker_embedding,
            temperature=0.65, enable_text_splitting=True,
        )
    wav = np.asarray(out["wav"], dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV")
    return buf.getvalue()


def cache_get(p):
    if os.path.exists(p) and os.path.getsize(p) > 44:
        return open(p, "rb").read()
    if os.path.exists(p):
        os.remove(p)
    return None


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": "xtts_v2",
            "cached": len([f for f in os.listdir(CACHE) if f.endswith(".wav")])}


@app.post("/synth")
def do(req: Req, x_weaver_key: str = Header(default="")):
    if KEY and x_weaver_key != KEY:
        raise HTTPException(status_code=403, detail="bad key")
    t = req.text.strip()[:800]
    if not t:
        return Response(status_code=400)
    key = hashlib.sha1(t.lower().encode()).hexdigest()
    p = os.path.join(CACHE, key + ".wav")
    c = cache_get(p)
    if c:
        return Response(c, media_type="audio/wav")
    t0 = time.time()
    wav = synth(t)
    tmp = p + ".part"
    open(tmp, "wb").write(wav)
    os.replace(tmp, p)
    print(f"[tts] synth {len(t)}c -> {len(wav)}b in {time.time() - t0:.2f}s", flush=True)
    return Response(wav, media_type="audio/wav")


for line in ["Hey. I can see you now."]:
    k = hashlib.sha1(line.lower().encode()).hexdigest()
    pp = os.path.join(CACHE, k + ".wav")
    if cache_get(pp) is None:
        try:
            open(pp, "wb").write(synth(line))
            print("[tts] warmed greeting", flush=True)
        except Exception as e:
            print("[tts] warmup fail:", e, flush=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("TTS_HOST", "0.0.0.0"), port=int(os.getenv("TTS_PORT", "8888")), log_level="warning")
