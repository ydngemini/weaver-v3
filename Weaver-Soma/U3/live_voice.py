"""Local duplex device runner with a deterministic loopback over the same transport."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import monotonic, sleep
import wave

import numpy as np

from audio_spectral import SAMPLE_RATE
from demo_realtime_voice import HELP, STATUS, synthesize
from realtime_voice import RealtimeVoiceSession, VoiceConfig

FRAME = SAMPLE_RATE // 50
BLOCK_BYTES = 2 * FRAME
SILENCE = bytes(BLOCK_BYTES)
REPLIES = {"status": STATUS, "help": HELP}


@dataclass(frozen=True)
class Playback:
    response_id: int
    pcm: bytes


@dataclass(frozen=True)
class CallbackBlock:
    offset: int
    pcm: bytes | None
    adc: float
    dac: float
    now: float
    response_id: int | None
    start: int
    count: int


class DuplexTransport:
    """One callback producer, one session worker. Only the worker publishes plans."""

    def __init__(self):
        self.blocks: deque[CallbackBlock] = deque(maxlen=12)
        self.plan: Playback | None = None
        self.offset = 0
        self.error: str | None = None
        self._active: Playback | None = None
        self._cursor = 0

    def exchange(self, pcm, adc, dac, now):
        """Fill one device block and enqueue clocked capture/submit evidence."""
        if len(self.blocks) == self.blocks.maxlen:
            self.error = "callback backlog overflow"
            return SILENCE
        if not all(math.isfinite(t) for t in (adc, dac, now)) or adc == dac == now == 0:
            self.error = "audio backend did not supply usable ADC/DAC timestamps"
            return SILENCE
        plan = self.plan
        if plan is not self._active:
            self._active = plan
            self._cursor = 0
        begin = self._cursor
        data = b"" if plan is None else plan.pcm[2 * begin:2 * begin + BLOCK_BYTES]
        count = len(data) // 2
        self._cursor += count
        self.blocks.append(CallbackBlock(
            self.offset, pcm, adc, dac, now, plan.response_id if count else None, begin, count))
        self.offset += FRAME
        return data.ljust(BLOCK_BYTES, b"\0")

    def callback(self, indata, outdata, frames, clock, status):
        if frames != FRAME or status.output_underflow or status.output_overflow:
            self.error = "unexpected device block size or output xrun"
        if self.error:
            outdata[:] = bytes(2 * frames)
            return
        pcm = None if status.input_overflow else bytes(indata)
        outdata[:] = self.exchange(pcm, clock.inputBufferAdcTime, clock.outputBufferDacTime,
                                    clock.currentTime)


class VoiceWorker:
    """Owns recognition, session state, playback accounting, and the JSONL timeline."""

    def __init__(self, transport, log, endpoint_ms, capture=None):
        from pocketsphinx import Decoder

        self.transport = transport
        self.log = log
        self.capture = capture
        self.session = RealtimeVoiceSession(VoiceConfig(end_frames=endpoint_ms // 20))
        self.asr = Decoder(samprate=SAMPLE_RATE)
        self.asr.add_jsgf_string(
            "local", "#JSGF V1.0; grammar local; public <request> = what is the status | please help me;",
        )
        self.asr.activate_search("local")
        self.replies = {name: [(synthesize(text, "slt"), text) for text in lines]
                        for name, lines in REPLIES.items()}
        self.pending: deque[CallbackBlock] = deque()
        self.active_id = None
        self.acknowledged = 0
        self.cancelled = None
        self.partial = ""

    def record(self, kind, **fields):
        self.log({"type": kind, **fields})

    def acknowledge(self, now):
        while self.pending:
            packet = self.pending[0]
            played = min(packet.count, max(0, int((now - packet.dac) * SAMPLE_RATE + 1e-6)))
            if packet.response_id == self.active_id and packet.start + played > self.acknowledged:
                self.acknowledged = packet.start + played
                for event in self.session.acknowledge_playback(self.active_id, self.acknowledged):
                    self.record(event.type, **event.data, device_time_s=now)
            if played < packet.count:
                return
            self.pending.popleft()

    def consume(self, block):
        if self.capture is not None:
            self.capture(SILENCE if block.pcm is None else block.pcm)
        if block.count and block.response_id == self.active_id:
            self.pending.append(block)
        self.acknowledge(block.now)
        if self.cancelled is not None and block.offset >= self.cancelled[1] and block.count == 0:
            self.record("silence_scheduled", response_id=self.cancelled[0], dac_s=block.dac)
            self.cancelled = None
        self.record("device_block", offset=block.offset, adc_s=block.adc, dac_s=block.dac,
                    callback_s=block.now, response_id=block.response_id, output_samples=block.count,
                    missing_input=block.pcm is None)
        if block.pcm is not None:
            self.session.push_pcm16le(block.offset, block.pcm)
        events = self.session.seal_input(block.offset + FRAME)
        for event in events:
            if event.type == "playback_clear":
                self.transport.plan = None
                self.pending.clear()
                self.cancelled = (event.data["response_id"], self.transport.offset)
                self.active_id = None
        for event in events:
            data = event.data
            if event.type != "speech_frame":
                fields = {k: v for k, v in data.items() if k not in ("pcm", "frame")}
                self.record(event.type, sample=event.at_sample, **fields)
            if event.type == "speech_started":
                self.asr.start_utt()
                self.partial = ""
            elif event.type == "speech_frame":
                frame = data["frame"]
                pcm = (frame.samples.numpy() * 32768).round().astype("<i2").tobytes()
                self.asr.process_raw(pcm, False, False)
                hypothesis = self.asr.hyp()
                if hypothesis and hypothesis.hypstr != self.partial:
                    self.partial = hypothesis.hypstr
                    self.record("transcript_partial", turn_id=data["turn_id"], text=self.partial)
            elif event.type == "speech_stopped":
                self.asr.end_utt()
                hypothesis = self.asr.hyp()
                spoken = hypothesis.hypstr if hypothesis else ""
                self.record("transcript_final", turn_id=data["turn_id"], text=spoken)
                if data["response_eligible"]:
                    self.reply(data["turn_id"], spoken)

    def drain(self):
        if self.transport.error:
            self.transport.plan = None
            self.record("device_error", message=self.transport.error)
            raise RuntimeError(self.transport.error)
        while self.transport.blocks:
            self.consume(self.transport.blocks.popleft())

    def reply(self, turn_id, spoken):
        words = set(spoken.split())
        name = next((intent for intent in REPLIES if intent in words), None)
        if name is None:
            self.record("unrecognized_turn", turn_id=turn_id)
            return
        event = self.session.start_response(turn_id)
        if event is None:
            return
        response_id = event.data["response_id"]
        self.record(event.type, **event.data, sample=event.at_sample)
        parts = self.replies[name]
        for pcm, text in parts:
            chunk = self.session.put_audio(response_id, pcm, text=text)
            self.record("response_audio", response_id=response_id, start_sample=chunk.data["start_sample"],
                        samples=len(pcm) // 2)
        for notice in self.session.finish_output(response_id):
            self.record(notice.type, **notice.data)
        self.active_id = response_id
        self.acknowledged = 0
        self.transport.plan = Playback(response_id, b"".join(pcm for pcm, _ in parts))


def voiced_blocks(data):
    framed = np.pad(data, (0, -len(data) % FRAME)).reshape(-1, FRAME)
    energy = np.mean((framed.astype(np.float32) / 32768) ** 2, axis=1)
    return framed, np.flatnonzero(energy >= VoiceConfig().speech_rms ** 2)


def trim_voiced(text):
    framed, audible = voiced_blocks(np.frombuffer(synthesize(text, "kal16"), "<i2"))
    return framed[audible[0]:audible[-1] + 1].reshape(-1)


def loopback(transport, worker, *, interrupt, echo_gain=0., fixture_cancel=False, pause_ms=0):
    """Synthetic clocks and a known one-block echo path. Uses the device transport."""
    requests = [np.frombuffer(synthesize(text, "kal16"), "<i2")
                for text in ("what is the status", "please help me")]
    user = np.zeros(16 * SAMPLE_RATE, dtype=np.int16)
    first = FRAME * 12
    resume = None
    if pause_ms:
        head = trim_voiced("what is")
        tail = trim_voiced("the status")
        pause = np.zeros(pause_ms * SAMPLE_RATE // 1000, dtype=np.int16)
        requests[0] = np.concatenate((head, pause, tail))
        resume = first + len(head) + len(pause)
    user[first:first + len(requests[0])] = requests[0]
    previous = np.zeros(FRAME, dtype=np.int16)
    placed = None
    for sample in range(0, len(user), FRAME):
        now = 1. + sample / SAMPLE_RATE
        clean = user[sample:sample + FRAME]
        echo = (echo_gain * previous).astype(np.int16)
        mic = clean.astype(np.int32) + echo
        if fixture_cancel:
            mic -= echo
        out = transport.exchange(mic.clip(-32768, 32767).astype("<i2").tobytes(),
                                 now - .020, now + .020, now)
        previous = np.frombuffer(out, "<i2")
        worker.drain()
        if (interrupt and placed is None and transport.plan is not None
                and transport.plan.response_id == 1):
            played_first = len(worker.replies["status"][0][0]) // 2
            placed = sample + (math.ceil(played_first / FRAME) + 9) * FRAME
            user[placed:placed + len(requests[1])] = requests[1]
    if placed is None:
        return None, None, resume
    inserted = 1. + placed / SAMPLE_RATE - .020
    first_audible = int(voiced_blocks(requests[1])[1][0]) * FRAME / SAMPLE_RATE
    return inserted, inserted + first_audible, resume


def write_loopback(output, endpoint_ms, interrupt, echo_gain=0., fixture_cancel=False, pause_ms=0):
    events = []
    transport = DuplexTransport()
    worker = VoiceWorker(transport, events.append, endpoint_ms)
    placed_adc_s, onset_adc_s, resume = loopback(transport, worker, interrupt=interrupt, echo_gain=echo_gain,
                                                 fixture_cancel=fixture_cancel, pause_ms=pause_ms)
    events.insert(0, {"type": "run", "mode": "synthetic_loopback", "endpoint_ms": endpoint_ms,
                      "sample_rate_hz": SAMPLE_RATE, "frame_samples": FRAME,
                      "injected_user_second_start_adc_s": placed_adc_s,
                      "reference_energy_onset_adc_s": onset_adc_s,
                      "pause_ms": pause_ms, "resume_sample": resume,
                      "echo_gain": echo_gain, "fixture_echo_subtraction": fixture_cancel})
    output.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--devices", action="store_true")
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--loopback", action="store_true")
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--capture-input", type=Path, metavar="WAV",
                        help="save live microphone samples as a mono WAV for consented annotation")
    parser.add_argument("--endpoint-ms", type=int, choices=(300, 600), default=300)
    args = parser.parse_args()
    if args.capture_input and (not args.live or not args.trace):
        parser.error("--capture-input requires --live and --trace")
    if args.loopback:
        if args.trace is None:
            parser.error("--trace is required for loopback")
        write_loopback(args.trace, args.endpoint_ms, True)
        return
    try:
        import sounddevice as sd
    except (ImportError, OSError) as error:
        parser.exit(1, f"voice: {error}\n")
    try:
        if args.devices:
            print(sd.query_devices())
            return
        transport = DuplexTransport()
        with ExitStack() as stack:
            trace = stack.enter_context(
                args.trace.open("x", encoding="utf-8") if args.trace else open("/dev/null", "w"))
            capture = None
            if args.capture_input:
                file = stack.enter_context(args.capture_input.open("xb"))
                wav = stack.enter_context(wave.open(file, "wb"))
                wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                capture = wav.writeframesraw
            worker = VoiceWorker(transport, lambda fields: trace.write(json.dumps(fields) + "\n"),
                                 args.endpoint_ms, capture)
            devices = tuple(int(name) if name and name.isdecimal() else name
                            for name in (args.input_device, args.output_device))
            with sd.RawStream(samplerate=SAMPLE_RATE, blocksize=FRAME, dtype="int16", channels=1,
                              device=devices, latency="low", callback=transport.callback):
                print("Say 'what is the status' or 'please help me'. Ctrl+C ends the session.", flush=True)
                worker.record("run", mode="device", endpoint_ms=args.endpoint_ms, devices=devices,
                              sample_rate_hz=SAMPLE_RATE, frame_samples=FRAME)
                while True:
                    worker.drain()
                    sleep(.002)
    except KeyboardInterrupt:
        return
    except (OSError, RuntimeError, sd.PortAudioError) as error:
        parser.exit(1, f"voice: {error}\n")


if __name__ == "__main__":
    main()
