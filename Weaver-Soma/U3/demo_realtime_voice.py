"""Run a local spoken status/help replay with real ASR and synthesized replies."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter_ns
import wave

import numpy as np

from audio_spectral import SAMPLE_RATE
from realtime_voice import RealtimeVoiceSession, VoiceEvent

FRAME = SAMPLE_RATE // 50
REQUESTS = ("what is the status", "please help me")
STATUS = ("The local voice session is running.",
          "I am about to explain the playback buffer, packet handling, and the audio classification benchmarks.")
HELP = ("I can check local status and handle a voice interruption.",)


def synthesize(text, voice):
    with TemporaryDirectory(prefix="soma_voice_") as folder:
        path = Path(folder) / "utterance.txt"
        path.write_text(text, encoding="utf-8")
        pcm = subprocess.check_output([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"flite=voice={voice}:textfile={path}",
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "s16le", "pipe:1",
        ], timeout=20)
    return pcm


def mono_wave(path, pcm):
    with wave.open(str(path), "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(SAMPLE_RATE)
        file.writeframes(pcm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    from pocketsphinx import Decoder

    decoder = Decoder(samprate=SAMPLE_RATE)
    decoder.add_jsgf_string(
        "soma_demo", "#JSGF V1.0; grammar soma_demo; public <request> = what is the status | please help me;",
    )
    decoder.activate_search("soma_demo")
    session = RealtimeVoiceSession()
    user = np.zeros(18 * SAMPLE_RATE, dtype=np.int16)
    recorded = np.zeros_like(user)
    first, second = [np.frombuffer(synthesize(text, "kal16"), dtype="<i2") for text in REQUESTS]
    first_start = FRAME * 12
    user[first_start:first_start + len(first)] = first
    playback = bytearray()
    response_id = None
    played = 0
    trace, turns, delays = [], [], []
    last_partial = ""
    mask16 = np.iinfo(np.int16)

    def log(event):
        data = {name: value for name, value in event.data.items() if name not in ("pcm", "frame")}
        if "pcm" in event.data:
            data["audio_samples"] = len(event.data["pcm"]) // 2
        trace.append({"type": event.type, "at_sample": event.at_sample, **data})

    for pos in range(0, len(user), FRAME):
        block = user[pos:pos + FRAME]
        heard = bytes(playback[:FRAME * 2])
        del playback[:len(heard)]
        if heard:
            phrase = np.frombuffer(heard, dtype="<i2").astype(np.int32)
            played += len(phrase)
            for event in session.acknowledge_playback(response_id, played):
                log(event)
            mixed = block.astype(np.int32)
            mixed[:len(phrase)] += phrase
            recorded[pos:pos + FRAME] = mixed.clip(mask16.min, mask16.max)
        else:
            recorded[pos:pos + FRAME] = block
        session.push_pcm16le(pos, block.astype("<i2", copy=False).tobytes())
        for event in session.seal_input(pos + FRAME):
            if event.type != "speech_frame":
                log(event)
            if event.type == "speech_started":
                last_partial = ""
                decoder.start_utt()
            elif event.type == "speech_frame":
                frame = event.data["frame"]
                pcm = (frame.samples.numpy() * 32768).round().clip(mask16.min, mask16.max).astype("<i2")
                decoder.process_raw(pcm.tobytes(), False, False)
                current = decoder.hyp()
                if current and current.hypstr != last_partial:
                    last_partial = current.hypstr
                    trace.append({"type": "transcript_partial", "at_sample": event.at_sample,
                                  "turn_id": event.data["turn_id"], "text": last_partial})
            elif event.type == "playback_clear":
                playback.clear()
                response_id = None
            elif event.type == "speech_stopped":
                begin_ns = perf_counter_ns()
                decoder.end_utt()
                hyp = decoder.hyp()
                spoken = hyp.hypstr if hyp else ""
                turns.append(spoken)
                log(VoiceEvent("transcript_final", event.at_sample, {
                    "turn_id": event.data["turn_id"], "text": spoken,
                }))
                if not event.data["response_eligible"]:
                    continue
                turn = session.start_response(event.data["turn_id"])
                if turn is None:
                    continue
                log(turn)
                response_id = turn.data["response_id"]
                played = 0
                intent = set(spoken.split())
                if "status" in intent:
                    phrases = STATUS
                elif "help" in intent:
                    phrases = HELP
                else:
                    phrases = ("Please say status or help.",)
                for index, phrase in enumerate(phrases):
                    voice = synthesize(phrase, "slt")
                    log(session.put_audio(response_id, voice, text=phrase))
                    playback.extend(voice)
                    if index == 0:
                        delays.append((perf_counter_ns() - begin_ns) / 1e6)
                        if phrases == STATUS:
                            second_start = pos + FRAME * (1 + math.ceil(len(voice) / (2 * FRAME)) + 6)
                            user[second_start:second_start + len(second)] = second
                for notice in session.finish_output(response_id):
                    log(notice)
        if len(turns) >= 2 and not playback:
            break

    interrupts = [e for e in trace if e["type"] == "response_interrupted"]
    assert len(turns) == 2 and "status" in turns[0].split() and "help" in turns[1].split(), turns
    assert len(interrupts) == 1 and interrupts[0]["discarded_samples"] > 0, interrupts
    assert interrupts[0]["heard_text"] == STATUS[0]
    assert any(e["type"] == "transcript_partial" for e in trace)
    assert any(e["type"] == "response_played" and e["heard_text"] == HELP[0] for e in trace)
    mono_wave(args.run_dir / "soma_realtime_demo.wav", recorded[:pos + FRAME].astype("<i2").tobytes())
    report = {
        "input": "two synthetic utterances on a separate microphone track",
        "recognizer": "PocketSphinx English, restricted status/help grammar", "synthesizer": "Flite via FFmpeg",
        "responder": "local status and help intents",
        "onset_ms": session.config.start_frames * session.config.frame_ms,
        "endpoint_silence_ms": session.config.end_frames * session.config.frame_ms,
        "final_asr_to_first_pcm_wall_ms": delays,
        "transcripts": turns, "interruptions": interrupts, "events": trace,
    }
    (args.run_dir / "voice_replay.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
