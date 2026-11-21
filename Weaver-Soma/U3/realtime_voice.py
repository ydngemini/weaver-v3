"""Deterministic input and playback coordination for local SOMA voice sessions."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import isfinite

from audio_stream import AudioWindow, AudioWindowBuffer, PacketReceipt, _sample_index
from audio_spectral import SAMPLE_RATE
from voice_endpoint import TranscriptEndpointModel


@dataclass(frozen=True)
class VoiceConfig:
    frame_ms: int = 20
    start_frames: int = 3
    end_frames: int = 15
    prefix_frames: int = 10
    max_turn_frames: int = 1500
    min_valid_fraction: float = .75
    speech_rms: float = .012

    def __post_init__(self):
        if (not all(type(n) is int for n in (self.frame_ms, self.start_frames, self.end_frames,
                                            self.prefix_frames, self.max_turn_frames))
                or not 1 <= self.frame_ms <= 100 or self.start_frames < 1 or self.end_frames < 1
                or self.prefix_frames < 0
                or self.max_turn_frames <= self.prefix_frames + self.start_frames
                or not 0 < self.min_valid_fraction <= 1
                or not isfinite(self.speech_rms) or self.speech_rms <= 0):
            raise ValueError("invalid realtime voice configuration")


@dataclass(frozen=True)
class VoiceEvent:
    type: str
    at_sample: int
    data: dict[str, object]


@dataclass
class _Response:
    id: int
    turn_id: int
    emitted: int = 0
    played: int = 0
    output_done: bool = False
    pending_text: deque[tuple[int, str]] = field(default_factory=deque)
    heard: list[str] = field(default_factory=list)


class RealtimeVoiceSession:
    """Process input while response audio is pending. Calls are serialized by the caller.

    The client owns playback and reports what actually played. A responder
    consumes input events and publishes PCM chunks against a response ID.
    """

    def __init__(self, config: VoiceConfig | None = None, *, start_sample: int = 0,
                 endpoint_model: TranscriptEndpointModel | None = None):
        self.config = VoiceConfig() if config is None else config
        frame = self.config.frame_ms * SAMPLE_RATE // 1000
        self.input = AudioWindowBuffer(start_sample=start_sample, window_samples=frame,
                                       max_pending_samples=max(SAMPLE_RATE, frame))
        self._prefix: deque[AudioWindow] = deque(maxlen=self.config.prefix_frames + self.config.start_frames)
        self._voiced = 0
        self._quiet = 0
        self._missing = 0
        self._turn_frames = 0
        self._turn_id = 0
        self._speaking = False
        self._ready_turn: int | None = None
        self._response_counter = 0
        self._response: _Response | None = None
        self._now = start_sample
        self._endpoint_model = endpoint_model
        self._last_voiced_until = start_sample
        self._transcript_until = -1
        self._transcript_evidence: tuple[float, dict[str, float]] | None = None

    def push_pcm16le(self, start_sample: int, pcm: bytes) -> PacketReceipt:
        return self.input.push_pcm16le(start_sample, pcm)

    def report_transcript(self, turn_id: int, text: str, through_sample: int) -> bool:
        """Submit a whole-turn ASR partial and the latest input sample it covers."""
        if self._endpoint_model is None or not self._speaking or turn_id != self._turn_id:
            return False
        through = _sample_index(through_sample)
        if not self._transcript_until <= through <= self._now:
            return False
        self._transcript_until = through
        self._transcript_evidence = self._endpoint_model.evidence(text)
        return True

    def _event(self, name: str, **data) -> VoiceEvent:
        return VoiceEvent(name, self._now, data)

    def seal_input(self, end_sample: int) -> list[VoiceEvent]:
        end = _sample_index(end_sample)
        if end > self.input.next_window_start + self.input.capacity:
            raise BufferError("advance the realtime audio watermark in smaller steps")
        self.input.seal(end)
        events = []
        while (frame := self.input.pop_window()) is not None:
            self._now = frame.end_sample
            events.extend(self._consume_frame(frame))
        return events

    def _consume_frame(self, frame: AudioWindow) -> list[VoiceEvent]:
        observed = int(frame.valid.sum())
        usable = observed >= self.config.min_valid_fraction * len(frame.valid)
        energy = float(frame.samples.square().sum()) / max(observed, 1)
        voiced = usable and energy >= self.config.speech_rms ** 2
        if not self._speaking:
            self._prefix.append(frame)
            self._voiced = self._voiced + 1 if voiced else 0
            if self._voiced < self.config.start_frames:
                return []
            self._turn_id += 1
            self._speaking = True
            self._ready_turn = None
            self._quiet = self._missing = self._voiced = 0
            self._last_voiced_until = frame.end_sample
            self._transcript_until = -1
            self._transcript_evidence = None
            self._turn_frames = len(self._prefix)
            started = self._event(
                "speech_started", turn_id=self._turn_id,
                onset_sample=self._now - self.config.start_frames * len(frame.samples),
                first_frame_sample=self._prefix[0].start_sample,
            )
            events = [started]
            if self._response is not None:
                events.extend(self._finish_playback(interrupted=True))
            events.extend(self._event("speech_frame", turn_id=self._turn_id, frame=window)
                          for window in self._prefix)
            self._prefix.clear()
            return events

        self._turn_frames += 1
        if voiced:
            self._last_voiced_until = frame.end_sample
        self._quiet = self._quiet + 1 if usable and not voiced else 0
        self._missing = self._missing + 1 if not usable else 0
        events = [self._event("speech_frame", turn_id=self._turn_id, frame=frame)]
        reason = None
        endpoint = None
        if self._missing >= self.config.end_frames:
            reason = "input_gap"
        elif self._quiet >= self.config.end_frames:
            reason, endpoint = self._silence_endpoint()
        if reason is None and self._turn_frames >= self.config.max_turn_frames:
            reason = "turn_limit"
        if reason is not None:
            self._speaking = False
            self._ready_turn = self._turn_id if reason != "input_gap" else None
            events.append(self._event(
                "speech_stopped", turn_id=self._turn_id, reason=reason,
                response_eligible=self._ready_turn is not None,
                **({"endpoint": endpoint} if endpoint is not None else {}),
            ))
        return events

    def _silence_endpoint(self) -> tuple[str | None, dict | None]:
        if self._endpoint_model is None:
            return "silence", None
        evidence = self._transcript_evidence if self._transcript_until >= self._last_voiced_until else None
        if evidence is not None and evidence[0] >= self._endpoint_model.threshold:
            rule = "transcript_complete"
        elif self._quiet >= 2 * self.config.end_frames:
            rule = "silence_cap"
        else:
            return None, None
        return "silence", {
            "rule": rule, "quiet_ms": self._quiet * self.config.frame_ms,
            "threshold": self._endpoint_model.threshold,
            "transcript_logit": evidence[0] if evidence is not None else None,
            "contributions": evidence[1] if evidence is not None else {},
        }

    def start_response(self, turn_id: int) -> VoiceEvent | None:
        """Start replying to the eligible turn. Results for superseded turns are ignored."""
        if self._ready_turn != _sample_index(turn_id) or self._response is not None:
            return None
        self._ready_turn = None
        self._response_counter += 1
        self._response = _Response(self._response_counter, turn_id)
        return self._event("response_started", turn_id=turn_id, response_id=self._response.id)

    def put_audio(self, response_id: int, pcm: bytes, *, text: str = "") -> VoiceEvent | None:
        """Emit one audio segment. Any supplied text must cover just this segment."""
        response = self._response
        if response is None or response.id != response_id or response.output_done:
            return None
        if not pcm or len(pcm) % 2:
            raise ValueError("response audio must contain whole PCM16 samples")
        start = response.emitted
        response.emitted += len(pcm) // 2
        if text:
            response.pending_text.append((response.emitted, text))
        return self._event("response_audio", response_id=response.id, start_sample=start, pcm=pcm)

    def finish_output(self, response_id: int) -> list[VoiceEvent]:
        response = self._response
        if response is None or response.id != response_id or response.output_done:
            return []
        response.output_done = True
        events = [self._event("response_output_done", response_id=response.id)]
        if response.played == response.emitted:
            events.extend(self._finish_playback(interrupted=False))
        return events

    def acknowledge_playback(self, response_id: int, played_samples: int) -> list[VoiceEvent]:
        response = self._response
        if response is None or response.id != response_id:
            return []
        played = _sample_index(played_samples)
        if not response.played <= played <= response.emitted:
            raise ValueError("played samples must be monotonic and cannot exceed emitted audio")
        response.played = played
        while response.pending_text and response.pending_text[0][0] <= played:
            _, text = response.pending_text.popleft()
            response.heard.append(text)
        if response.output_done and played == response.emitted:
            return self._finish_playback(interrupted=False)
        return []

    def _finish_playback(self, *, interrupted: bool) -> list[VoiceEvent]:
        response = self._response
        self._response = None
        fields = {
            "response_id": response.id, "turn_id": response.turn_id,
            "played_samples": response.played, "discarded_samples": response.emitted - response.played,
            "heard_text": " ".join(response.heard),
        }
        if interrupted:
            return [self._event("playback_clear", **fields), self._event("response_interrupted", **fields)]
        return [self._event("response_played", **fields)]
