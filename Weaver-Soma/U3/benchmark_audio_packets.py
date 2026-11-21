"""Verify timestamped packet replay against direct SOMA U3 inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from audio_spectral import SAMPLE_RATE
from audio_stream import AudioWindowBuffer
from benchmark_esc50_audio import BATCH_SIZE, read_audio
from model import WeaverSOMAU3
from train_esc50 import CLASSES, prepare_data

PACKET = SAMPLE_RATE // 25
WINDOW = 5 * SAMPLE_RATE


def assemble(wave, *, reorder=False, gap_samples=0):
    buffer = AudioWindowBuffer(window_samples=len(wave))
    offsets = list(range(0, len(wave), PACKET))
    gap_start = ((len(wave) - gap_samples) // 2 // PACKET) * PACKET
    gap_end = gap_start + gap_samples
    skipped = [offset for offset in offsets if gap_start <= offset < gap_end]
    for offset in reversed(offsets) if reorder else offsets:
        if offset in skipped:
            continue
        packet = wave[offset:offset + PACKET]
        buffer.push(offset, packet)
        if reorder and offset % (17 * PACKET) == 0:
            receipt = buffer.push(offset, packet)
            assert receipt.duplicate_samples == len(packet)
    buffer.seal(len(wave) - 1)
    assert buffer.pop_window() is None
    buffer.seal(len(wave))
    window = buffer.pop_window()
    if skipped:
        late = buffer.push(skipped[0], wave[skipped[0]:skipped[0] + PACKET])
        assert late.late_samples == PACKET and late.accepted_samples == 0
    expected_valid = torch.ones_like(wave, dtype=torch.bool)
    expected_valid[gap_start:gap_end] = False
    assert torch.equal(window.valid, expected_valid)
    assert torch.equal(window.received, expected_valid)
    assert torch.equal(window.samples, wave.masked_fill(~expected_valid, 0))
    return window


@torch.inference_mode()
def scores(model, waveforms, validity):
    columns = {"audio_logits": [], "p_correct": [], "scope_score": [],
               "score_available": [], "recommend_abstain": []}
    for audio, valid in zip(waveforms.split(BATCH_SIZE), validity.split(BATCH_SIZE)):
        b = len(audio)
        out = model(torch.zeros(b, 8, 7), torch.ones(b, 8, 3), torch.zeros(b, 8),
                    modalities={"audio": {"values": audio, "valid": valid}})
        recognition = out["modality_logits"].get("audio", torch.zeros(b, len(CLASSES)))
        confidence = out["audio_self_monitor"]
        scope = out["audio_scope_monitor"]
        columns["audio_logits"].append(recognition[:, :len(CLASSES)])
        columns["p_correct"].append(confidence["p_correct"])
        columns["scope_score"].append(scope["score"])
        columns["score_available"].append(scope["score_available"])
        columns["recommend_abstain"].append(confidence["recommend_abstain"] | scope["recommend_abstain"])
    return {name: torch.cat(rows) for name, rows in columns.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--monitor-checkpoint", type=Path, required=True)
    parser.add_argument("--scope-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.manual_seed(921)
    model = WeaverSOMAU3(audio_encoder="logmel", core_dim=64, depth=1, ff_dim=128).eval()
    monitor = model.load_audio_self_monitor(args.monitor_checkpoint)
    scope = model.load_audio_scope_monitor(args.scope_checkpoint)
    rows = [row for row in prepare_data(args.data_dir, 8) if row["fold"] == "5"]
    waveforms, labels, _ = read_audio(args.data_dir, rows)
    assert waveforms.shape[1] == WINDOW
    direct = scores(model, waveforms, torch.ones_like(waveforms, dtype=torch.bool))
    result = {}
    for name, gap, reorder in (("ordered", 0, False), ("reordered_with_retries", 0, True),
                               ("one_second_loss", SAMPLE_RATE, False),
                               ("two_second_loss", 2 * SAMPLE_RATE, False),
                               ("missing_window", WINDOW, False)):
        packets = [assemble(clip, reorder=reorder, gap_samples=gap) for clip in waveforms]
        scoped = scores(model, torch.stack([item.samples for item in packets]),
                        torch.stack([item.valid for item in packets]))
        if gap == 0:
            assert all(torch.equal(value, direct[key]) for key, value in scoped.items())
        available = scoped["score_available"]
        accepted = ~scoped["recommend_abstain"]
        correct = scoped["audio_logits"].argmax(1) == labels
        result[name] = {
            "missing_samples_per_window": gap, "scoreable": int(available.sum()),
            "classifier_correct_when_scoreable": int((correct & available).sum()),
            "accepted_by_both_monitors": int(accepted.sum()),
            "incorrect_among_accepted": int((accepted & ~correct).sum()),
            "packet_and_batch_outputs_identical": gap == 0,
        }
    assert result["missing_window"]["scoreable"] == result["missing_window"]["accepted_by_both_monitors"] == 0
    output = {
        "sample_rate": SAMPLE_RATE, "window_samples": WINDOW, "packet_samples": PACKET,
        "recordings": len(rows), "subset": "previously evaluated ESC-50 fold 5 familiar classes",
        "self_monitor_sha256": monitor.loaded_checkpoint_sha256,
        "scope_monitor_sha256": scope.loaded_checkpoint_sha256,
        "cases": result,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
