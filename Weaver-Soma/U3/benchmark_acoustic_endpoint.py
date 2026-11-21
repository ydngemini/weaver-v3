"""CPU acoustic endpoint pilot on human AMI headset recordings.

The same causal energy gate locates quiet runs for all policies. AMI
annotations label them offline. The classifier uses only pre-pause PCM.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from hashlib import file_digest
import json
from pathlib import Path
import re
import wave
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import numpy as np
import torch

from torch.nn import functional as F

from audio_spectral import SAMPLE_RATE
from realtime_voice import VoiceConfig


VOICE_CONFIG = VoiceConfig()
FRAME = SAMPLE_RATE * VOICE_CONFIG.frame_ms // 1000
EARLY_MS = 300
LATE_MS = 600
LOOKBACK = SAMPLE_RATE
EARLY_SAMPLES = EARLY_MS * SAMPLE_RATE // 1000
TERMINAL_GAP = 900 * SAMPLE_RATE // 1000
MIN_RUNUP = 400 * SAMPLE_RATE // 1000
SPLITS = {
    "train": ("ES2002a", "ES2003a", "ES2005a", "ES2006a", "ES2007a", "ES2008a"),
    "calibration": ("ES2009a", "ES2010a", "ES2011a"),
    "test": ("ES2012a", "IS1000a", "IS1001a"),
}
# Budget for extra short-pause cuts over 600 ms on calibration data.
CALIBRATION_BUDGET = .02
NITE = "{http://nite.sourceforge.net/}"
WORD_RANGE = re.compile(r"#id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?$")


@dataclass(frozen=True)
class Event:
    meeting: str
    boundary_sample: int
    gap_samples: int
    terminal: bool
    features: dict[str, float]


def _frame_features(samples: np.ndarray) -> tuple[np.ndarray, ...]:
    frames = np.lib.stride_tricks.sliding_window_view(samples, 2 * FRAME)[::FRAME].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    log_rms = 20 * np.log10(np.maximum(rms, 1e-5))
    crossing = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    spectrum = np.fft.rfft(frames, n=4 * FRAME, axis=1)
    corr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)
    lags = np.arange(SAMPLE_RATE // 400, SAMPLE_RATE // 75 + 1)
    peak = np.argmax(corr[:, lags], axis=1)
    periodicity = corr[np.arange(len(corr)), lags[peak]] / np.maximum(corr[:, 0], 1e-12)
    pitch = np.log2(SAMPLE_RATE / lags[peak])
    return log_rms, crossing, periodicity, pitch


def acoustic_features(waveform: np.ndarray, boundary: int) -> dict[str, float]:
    """Use exactly one second of mono PCM before the fixed policy clock starts."""
    if boundary < LOOKBACK or boundary > len(waveform):
        raise ValueError("insufficient acoustic history")
    window = waveform[boundary - LOOKBACK:boundary].astype(np.float64) / 32768
    db, crossing, periodicity, pitch = _frame_features(window)
    voiced = (periodicity >= .55) & (db >= np.max(db) - 25)
    voiced_pitches = pitch[voiced]
    tail_pitches = pitch[-8:][voiced[-8:]]
    return {
        "tail_rms_db": float(np.median(db[-8:])),
        "tail_energy_drop_db": float(np.median(db[-5:]) - np.median(db[-15:-5])),
        "tail_relative_db": float(np.median(db[-8:]) - np.median(db[:-8])),
        "tail_energy_range_db": float(np.ptp(db[-10:])),
        "tail_zcr": float(np.median(crossing[-8:])),
        "tail_periodicity": float(np.median(periodicity[-8:])),
        "tail_voiced_fraction": float(np.mean(voiced[-8:])),
        "context_voiced_fraction": float(np.mean(voiced)),
        "tail_pitch_change_octaves": float(tail_pitches[-1] - tail_pitches[0])
        if len(tail_pitches) >= 2 else 0.0,
        "tail_relative_pitch_octaves": float(np.median(tail_pitches) - np.median(voiced_pitches))
        if len(tail_pitches) else 0.0,
    }


def _annotations(archive: ZipFile, meeting: str, agent: str) -> list[tuple[int, int, bool]]:
    words = list(ET.fromstring(archive.read(f"words/{meeting}.{agent}.words.xml")))
    indices = {word.attrib[NITE + "id"]: index for index, word in enumerate(words)}
    segments = ET.fromstring(archive.read(f"segments/{meeting}.{agent}.segments.xml"))
    labeled = []
    for segment in segments:
        contains_words = False
        for child in segment.findall(NITE + "child"):
            match = WORD_RANGE.search(child.attrib["href"])
            if match is None:
                raise ValueError(f"unrecognized AMI word reference in {meeting}")
            first, last = match.groups()
            span = words[indices[first]:indices[last or first] + 1]
            contains_words |= any(word.tag == "w" and word.get("punc") != "true" for word in span)
        start = round(float(segment.attrib["transcriber_start"]) * SAMPLE_RATE)
        end = round(float(segment.attrib["transcriber_end"]) * SAMPLE_RATE)
        if end > start:
            labeled.append((start, end, contains_words))
    return sorted(labeled)


def _meeting_events(path: Path, archive: ZipFile, meeting: str, agent: str) -> list[Event]:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path} must be mono 16 kHz PCM16 WAV")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
        if len(pcm) != wav.getnframes():
            raise ValueError(f"truncated waveform: {path}")
    segments = _annotations(archive, meeting, agent)
    starts = [segment[0] for segment in segments]
    frames = pcm[:len(pcm) // FRAME * FRAME].reshape(-1, FRAME).astype(np.float32)
    power = np.einsum("ij,ij->i", frames, frames) / FRAME
    voiced = power >= (32768 * VOICE_CONFIG.speech_rms) ** 2
    transitions = np.diff(voiced.astype(np.int8), prepend=0, append=1)
    resumes = np.flatnonzero(transitions == 1)
    events = []
    for stop_frame in np.flatnonzero(transitions == -1):
        stop = int(stop_frame)
        restart = int(resumes[np.searchsorted(resumes, stop, side="right")])
        end = stop * FRAME
        gap = (restart - stop) * FRAME
        if end < LOOKBACK or gap < EARLY_SAMPLES or voiced[stop - 20:stop].sum() < 3:
            continue
        index = bisect_right(starts, end) - 1
        if index < 0:
            continue
        start, annotated_end, contains_words = segments[index]
        if not contains_words or end - start < MIN_RUNUP:
            continue
        # Never infer ground truth from interpolated word-level silence timings.
        # Require resumed, sustained energy well inside the same annotated segment.
        margin = 160 * SAMPLE_RATE // 1000
        internal = (restart * FRAME + margin <= annotated_end
                    and voiced[restart:restart + 8].sum() >= 3)
        next_start = starts[index + 1] if index + 1 < len(starts) else len(pcm)
        terminal = (abs(annotated_end - end) <= margin
                    and next_start - annotated_end >= TERMINAL_GAP
                    and gap >= LATE_MS * SAMPLE_RATE // 1000)
        if internal or terminal:
            events.append(Event(meeting, end, gap, terminal, acoustic_features(pcm, end)))
    return events


def _split_events(data: Path) -> tuple[dict[str, list[Event]], dict]:
    archive_path = data / "manual.zip"
    with archive_path.open("rb") as stream:
        sources = {"manual.zip": file_digest(stream, "sha256").hexdigest()}
    events = {}
    speakers = {}
    with ZipFile(archive_path) as archive:
        meetings = ET.fromstring(archive.read("corpusResources/meetings.xml"))
        entries = {item.get("observation"): item for item in meetings.findall("meeting")}
        for split, meeting_names in SPLITS.items():
            events[split] = []
            for meeting in meeting_names:
                # Check all voices in the room, including potential headset bleed.
                for person in entries[meeting].findall("speaker"):
                    identity = person.attrib["global_name"]
                    if speakers.setdefault(identity, split) != split:
                        raise ValueError("AMI speaker overlaps between splits")
                person, = (person for person in entries[meeting].findall("speaker")
                           if person.get("channel") == "0")
                path = data / f"{meeting}.Headset-0.wav"
                rows = _meeting_events(path, archive, meeting, person.attrib["nxt_agent"])
                events[split].extend(rows)
                with path.open("rb") as stream:
                    sources[path.name] = file_digest(stream, "sha256").hexdigest()
            if len({row.terminal for row in events[split]}) != 2:
                raise ValueError(f"{split} requires examples of both labels")
    return events, sources


def _fit(events: list[Event]) -> tuple[float, dict[str, float]]:
    matrix = torch.tensor([list(event.features.values()) for event in events], dtype=torch.float64)
    targets = torch.tensor([event.terminal for event in events], dtype=torch.float64)
    means = matrix.mean(dim=0)
    scales = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
    normalized = (matrix - means) / scales
    model = torch.nn.Linear(matrix.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(normalized).flatten(), targets)
        loss = loss + .01 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        weights = model.weight.flatten() / scales
        bias = (model.bias[0] - weights @ means).item()
    return bias, dict(zip(events[0].features, weights.tolist()))


def _score(events: list[Event], bias: float, weights: dict[str, float]) -> np.ndarray:
    return np.array([bias + sum(weights[name] * value for name, value in event.features.items())
                     for event in events])


def _operating_point(events: list[Event], scores: np.ndarray) -> float | None:
    short = sorted((score for event, score in zip(events, scores)
                    if not event.terminal and event.gap_samples < LATE_MS * SAMPLE_RATE // 1000),
                   reverse=True)
    if not short:
        raise ValueError("calibration split has no short internal pauses")
    budget = int(len(short) * CALIBRATION_BUDGET)
    threshold = float(np.nextafter(short[budget], np.inf))
    return threshold if np.any(scores >= threshold) else None


def _metrics(events: list[Event], timeouts: np.ndarray) -> dict:
    terminals = np.array([event.terminal for event in events], dtype=bool)
    gaps_ms = np.array([event.gap_samples * 1000 / SAMPLE_RATE for event in events])
    short = ~terminals & (gaps_ms < LATE_MS)
    false_cut = ~terminals & (timeouts <= gaps_ms)
    waits = timeouts[terminals]
    return {
        "terminal_events": len(waits),
        "short_internal_pauses": int(short.sum()),
        "internal_pauses": int((~terminals).sum()),
        "early_terminal_events": int((terminals & (timeouts == EARLY_MS)).sum()),
        "short_pause_false_cuts": int((false_cut & short).sum()),
        "all_pause_false_cuts": int(false_cut.sum()),
        "terminal_wait_mean_ms": float(waits.mean()) if len(waits) else None,
        "terminal_wait_p50_ms": int(np.quantile(waits, .5, method="higher")) if len(waits) else None,
        "terminal_wait_p95_ms": int(np.quantile(waits, .95, method="higher")) if len(waits) else None,
    }


def run(data: Path, output: Path) -> dict:
    torch.set_num_threads(1)
    events, sources = _split_events(data)
    bias, weights = _fit(events["train"])
    calibration = _score(events["calibration"], bias, weights)
    threshold = _operating_point(events["calibration"], calibration)
    summary = {
        "splits": SPLITS,
        "calibration_extra_short_pause_budget": CALIBRATION_BUDGET,
        "sample_rate_hz": SAMPLE_RATE,
        "rms_gate": VOICE_CONFIG.speech_rms,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "sha256": sources,
        "results": {},
    }
    output.mkdir(parents=True, exist_ok=True)
    model = {"version": 1, "threshold": threshold, "bias": bias, "weights": weights}
    (output / "acoustic_endpoint_model.json").write_text(json.dumps(model, indent=2) + "\n")
    with (output / "endpoint_events.jsonl").open("w", encoding="utf-8") as trace:
        for split in ("calibration", "test"):
            rows = events[split]
            scores = _score(rows, bias, weights)
            adaptive = (np.full(len(rows), LATE_MS) if threshold is None
                        else np.where(scores >= threshold, EARLY_MS, LATE_MS))
            policies = {"fixed_300": np.full(len(rows), EARLY_MS),
                        "fixed_600": np.full(len(rows), LATE_MS), "acoustic": adaptive}
            by_meeting = {}
            for meeting in SPLITS[split]:
                mask = np.array([row.meeting == meeting for row in rows], dtype=bool)
                subset = [row for row in rows if row.meeting == meeting]
                by_meeting[meeting] = {name: _metrics(subset, delays[mask])
                                       for name, delays in policies.items()}
            summary["results"][split] = {
                "overall": {name: _metrics(rows, delays) for name, delays in policies.items()},
                "by_meeting": by_meeting,
            }
            for event, score, delay in zip(rows, scores, adaptive):
                trace.write(json.dumps({
                    "split": split, "meeting": event.meeting,
                    "boundary_sample": event.boundary_sample, "gap_samples": event.gap_samples,
                    "terminal": event.terminal, "logit": float(score), "wait_ms": int(delay),
                    "contributions": {name: value * weights[name]
                                      for name, value in event.features.items()},
                }) + "\n")
    summary["train_labels"] = dict(Counter("terminal" if row.terminal else "internal"
                                          for row in events["train"]))
    (output / "acoustic_endpoint_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ami-dir", required=True, type=Path, help="manual.zip and AMI headset WAVs")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.ami_dir, args.run_dir)
    for split, metrics in summary["results"].items():
        print(split, json.dumps(metrics["overall"], indent=2))


if __name__ == "__main__":
    main()
