"""Weaver voice server — OpenVoice v2 cloned-voice TTS on 127.0.0.1:8092.

Clone-ONCE design (the reference wav is processed a single time, ever):
  * First boot: extracts the speaker tone-color embedding from
    ~/tts/weaver_voice_ref.wav and saves it to ~/tts/weaver_voice_se.pth.
    Every later boot loads the saved file — the wav is never touched again.
  * Every synthesized line is cached to disk keyed by sha1(text): a repeated
    line (the wake greeting, stock phrases) replays from file with zero compute.

Endpoints (Caddy exposes this as /tts/* behind the X-Weaver-Key gate):
  GET  /health          → {"status": "ok", "cached": N}
  POST /synth {"text"}  → audio/wav

Runs as the weaver-tts systemd unit. 2 torch threads — the box has 2 vCPUs
shared with the LLM; do not raise this.
"""
import hashlib
import io
import os
import time

import torch
torch.set_num_threads(2)

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

HOME = os.path.expanduser("~/tts")
REF_WAV = os.path.join(HOME, "weaver_voice_ref.wav")
SAVED_SE = os.path.join(HOME, "weaver_voice_se.pth")
CKPT_DIR = os.path.join(HOME, "OpenVoiceV2")
CACHE = os.path.join(HOME, "cache")
os.makedirs(CACHE, exist_ok=True)

from melo.api import TTS
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

DEVICE = "cpu"
print("[tts] loading MeloTTS EN…")
base = TTS(language="EN", device=DEVICE)
SPEAKER_ID = base.hps.data.spk2id["EN-US"]

print("[tts] loading tone-color converter…")
converter = ToneColorConverter(os.path.join(CKPT_DIR, "converter", "config.json"), device=DEVICE)
converter.load_ckpt(os.path.join(CKPT_DIR, "converter", "checkpoint.pth"))
source_se = torch.load(os.path.join(CKPT_DIR, "base_speakers", "ses", "en-us.pth"),
                       map_location=DEVICE)

# ── clone ONCE: saved embedding wins; extract only if it doesn't exist yet ──
if os.path.exists(SAVED_SE):
    target_se = torch.load(SAVED_SE, map_location=DEVICE)
    print(f"[tts] loaded saved voice embedding {SAVED_SE}")
else:
    print(f"[tts] first boot: extracting voice embedding from {REF_WAV}…")
    target_se, _ = se_extractor.get_se(REF_WAV, converter, vad=True)
    torch.save(target_se, SAVED_SE)
    print(f"[tts] saved voice embedding → {SAVED_SE} (the wav won't be processed again)")

app = FastAPI()


class Req(BaseModel):
    text: str


def synth_to_wav_bytes(text: str) -> bytes:
    tmp_base = os.path.join(CACHE, "_base_tmp.wav")
    tmp_out = os.path.join(CACHE, "_out_tmp.wav")
    base.tts_to_file(text, SPEAKER_ID, tmp_base, speed=1.0)
    converter.convert(audio_src_path=tmp_base, src_se=source_se, tgt_se=target_se,
                      output_path=tmp_out, message="@Weaver")
    with open(tmp_out, "rb") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok", "cached": len([f for f in os.listdir(CACHE) if f.endswith(".wav") and not f.startswith("_")])}


@app.post("/synth")
def synth(req: Req):
    text = req.text.strip()[:400]
    if not text:
        return Response(status_code=400)
    key = hashlib.sha1(text.lower().encode()).hexdigest()
    path = os.path.join(CACHE, f"{key}.wav")
    if os.path.exists(path):                       # replay from file — zero compute
        with open(path, "rb") as f:
            return Response(f.read(), media_type="audio/wav")
    t0 = time.time()
    wav = synth_to_wav_bytes(text)
    with open(path, "wb") as f:
        f.write(wav)
    print(f"[tts] synth {len(text)} chars in {time.time()-t0:.1f}s → cached {key[:8]}")
    return Response(wav, media_type="audio/wav")


# pre-render fixed lines so first contact is instant
WARMUP = ["Hey. I can see you now."]
for line in WARMUP:
    k = hashlib.sha1(line.lower().encode()).hexdigest()
    p = os.path.join(CACHE, f"{k}.wav")
    if not os.path.exists(p):
        try:
            with open(p, "wb") as f:
                f.write(synth_to_wav_bytes(line))
            print(f"[tts] warmed: {line!r}")
        except Exception as e:
            print(f"[tts] warmup failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8092, log_level="warning")
