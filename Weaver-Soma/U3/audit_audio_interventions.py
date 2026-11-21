"""Offline paired interventions on U3 audio tokens and the fusion path."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from audio_self_monitor import AUDIO_CLASSES
from benchmark_esc50_audio import BATCH_SIZE, read_audio
from model import WeaverSOMAU3
from train_audio_self_monitor import distort
from train_esc50 import prepare_data, sha256

GROUPS = ("mel_0_31", "mel_32_63", "learned_0_31", "learned_32_63")
GROUP_SIZE = 32
TEXT = tuple(b"auditory evidence")
STANDARD_GAIN = 0.50
WEAK_GAIN = 0.80
FUSION_TEST_GAIN = 0.25
OUTPUT_FEATURES = (
    *(f"baseline_probability_{name}" for name in AUDIO_CLASSES),
    *(f"probability_delta_{name}" for name in AUDIO_CLASSES),
)
TOKEN_FEATURES = tuple(f"relative_rms_change_{name}" for name in GROUPS)


@dataclass
class Replay:
    logits: torch.Tensor
    monitor: torch.Tensor
    tokens: torch.Tensor
    valid: torch.Tensor
    attention: torch.Tensor
    state: torch.Tensor

    @property
    def probabilities(self):
        return self.logits.softmax(dim=1)


def unchanged(tokens, valid, _):
    return tokens.clone(), valid.clone()


def attenuate(group, scale):
    def intervene(tokens, valid, _):
        tokens = tokens.clone()
        start = group * GROUP_SIZE
        tokens[..., start:start + GROUP_SIZE].mul_(scale)
        return tokens, valid
    return intervene


def restore(reference, group=None):
    def intervene(tokens, valid, indices):
        if group is None:
            return reference.tokens[indices], reference.valid[indices]
        tokens = tokens.clone()
        start = group * GROUP_SIZE
        target = slice(start, start + GROUP_SIZE)
        tokens[..., target] = reference.tokens[indices, :, target]
        return tokens, valid
    return intervene


@torch.inference_mode()
def replay(model, waveforms, *, edit=None, mask_audio_at_fusion=False):
    batches = []
    for start in range(0, len(waveforms), BATCH_SIZE):
        waveform = waveforms[start:start + BATCH_SIZE]
        count = len(waveform)
        indices = slice(start, start + count)
        captured = {}

        def tap_audio(_module, _args, output):
            tokens, valid = output
            if edit is not None:
                tokens, valid = edit(tokens, valid, indices)
            captured["tokens"], captured["valid"] = tokens, valid
            return tokens, valid

        def mask_fusion(_module, args, kwargs):
            updated = dict(kwargs)
            key_mask = updated["key_padding_mask"].clone()
            key_mask[:, :min(captured["tokens"].shape[1], model.max_modality_tokens)] = True
            if key_mask.all(dim=1).any():
                raise ValueError("fusion intervention requires a valid non-audio reference")
            updated["key_padding_mask"] = key_mask
            return args, updated

        def tap_fusion(_module, _args, output):
            captured["attention"] = output[0][:, 0]

        with ExitStack() as hooks:
            hooks.enter_context(model.spectral_audio.register_forward_hook(tap_audio))
            hooks.enter_context(model.cross_attention.register_forward_hook(tap_fusion))
            if mask_audio_at_fusion:
                hooks.enter_context(model.cross_attention.register_forward_pre_hook(
                    mask_fusion, with_kwargs=True,
                ))
            result = model(
                torch.zeros(count, 8, 7), torch.ones(count, 8, 3), torch.zeros(count, 8),
                modalities={
                    "audio": {"values": waveform},
                    "text": {"values": torch.tensor(TEXT).expand(count, -1)},
                },
            )
        score = result["audio_self_monitor"]
        if not score["score_available"].all():
            raise ValueError("replay requires scoreable clips")
        batches.append(Replay(
            result["modality_logits"]["audio"][:, :len(AUDIO_CLASSES)],
            score["p_correct"], captured["tokens"], captured["valid"],
            captured["attention"], result["state"],
        ))
    return Replay(*(torch.cat([getattr(batch, key) for batch in batches])
                    for key in Replay.__dataclass_fields__))


def probability_shift(reference, actual):
    return (reference.probabilities - actual.probabilities).abs().sum(dim=1) / 2


def token_group_shift(reference, actual):
    delta = actual.tokens - reference.tokens
    groups = delta.reshape(len(delta), delta.shape[1], len(GROUPS), GROUP_SIZE)
    base = reference.tokens.reshape_as(groups)
    return groups.square().mean(dim=(1, 3)).sqrt() / base.square().mean(dim=(1, 3)).sqrt().clamp_min(1e-6)


def effects(reference, actual, labels):
    return {
        "mean_probability_total_variation": float(probability_shift(reference, actual).mean()),
        "class_flips": int((reference.logits.argmax(1) != actual.logits.argmax(1)).sum()),
        "correct": int((actual.logits.argmax(1) == labels).sum()),
        "mean_absolute_confidence_change": float((actual.monitor - reference.monitor).abs().mean()),
        "mean_attention_l2_change": float((actual.attention - reference.attention).norm(dim=1).mean()),
        "mean_state_l2_change": float((actual.state - reference.state).norm(dim=1).mean()),
    }


def output_features(reference, actual):
    return torch.cat((reference.probabilities, actual.probabilities - reference.probabilities), dim=1)


def fit_logistic(features, labels):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500)).fit(features, labels)


def probe_parameters(probe, names, sample):
    scaler, classifier = probe.steps[0][1], probe.steps[1][1]
    scaled = (sample.numpy().astype(np.float64) - scaler.mean_) / scaler.scale_
    score = scaled @ classifier.coef_.T + classifier.intercept_
    expected = probe.decision_function(sample).reshape(score.shape)
    np.testing.assert_allclose(score, expected, rtol=1e-5, atol=2e-6)
    return {
        "feature_names": names, "center": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(), "scores_reconstructed": True,
    }


def site_localization(dev, dev_sites, test, test_sites, weak_sites):
    dev_x = torch.cat([output_features(dev, result) for result in dev_sites])
    fit = fit_logistic(dev_x, np.repeat(np.arange(len(GROUPS)), len(dev.logits)))
    labels = np.repeat(np.arange(len(GROUPS)), len(test.logits))
    results = {}
    for name, sites in (("standard", test_sites), ("weaker_unseen_magnitude", weak_sites)):
        x = torch.cat([output_features(test, result) for result in sites])
        locations = torch.cat([token_group_shift(test, result).argmax(1) for result in sites])
        results[name] = {
            "n": len(labels), "input_only_chance": 1 / len(GROUPS),
            "logits_only_accuracy": float(np.mean(fit.predict(x) == labels)),
            "paired_token_tap_accuracy": float(np.mean(locations.numpy() == labels)),
        }
        if name == "standard":
            results["logits_probe_parameters"] = probe_parameters(fit, OUTPUT_FEATURES, x)
    return results


def waveform_controls(model, waves, seed):
    gen = torch.Generator().manual_seed(seed)
    noise = torch.randn(waves.shape, generator=gen)
    rms = waves.square().mean(dim=1, keepdim=True).sqrt()
    controls = {}
    for amount in (0.025, 0.08, 0.25, 0.80):
        controls[f"noise_{amount}"] = replay(model, (waves + amount * rms * noise).clamp(-1, 1))
    for width in (3, 7, 15, 31):
        filtered = F.avg_pool1d(waves.unsqueeze(1), width, stride=1, padding=width // 2)[:, 0]
        controls[f"lowpass_{width}"] = replay(model, filtered)
    return controls


def match_controls(reference, internal, controls):
    """Pair on the same clip and the nearest magnitude of probability change."""
    control_items = list(controls.items())
    control_effect = torch.stack([probability_shift(reference, case) for _, case in control_items])
    control_logits = torch.stack([output_features(reference, case) for _, case in control_items])
    control_tokens = torch.stack([token_group_shift(reference, case) for _, case in control_items])
    logits_rows, token_rows, pairs = [], [], []
    for group, case in enumerate(internal):
        impact = probability_shift(reference, case)
        error, choice = (control_effect - impact).abs().min(dim=0)
        keep = (error <= .025 + .25 * impact).nonzero().flatten()
        selected = choice[keep]
        if not len(keep):
            continue
        internal_logits = output_features(reference, case)
        internal_tokens = token_group_shift(reference, case)
        logits_rows.extend((internal_logits[keep], control_logits[selected, keep]))
        token_rows.extend((internal_tokens[keep], control_tokens[selected, keep]))
        for row, candidate in zip(keep.tolist(), selected.tolist()):
            pairs.append({
                "row": row, "group": GROUPS[group], "waveform_edit": control_items[candidate][0],
                "encoder_impact": float(impact[row]), "waveform_impact": float(control_effect[candidate, row]),
                "encoder_probe_features": torch.cat((internal_logits[row], internal_tokens[row])).tolist(),
                "waveform_probe_features": torch.cat((control_logits[candidate, row], control_tokens[candidate, row])).tolist(),
            })
    if not pairs:
        raise ValueError("no pairs met the predeclared probability-change tolerance")
    # The alternating blocks hold internal and waveform rows with one of each per pair.
    targets = torch.cat([torch.full((len(block),), 1 - i % 2) for i, block in enumerate(logits_rows)])
    return torch.cat(logits_rows), torch.cat(token_rows), targets, pairs


def distinguish_causes(dev, test):
    dev_logits, dev_tokens, dev_y, dev_pairs = dev
    test_logits, test_tokens, test_y, test_pairs = test
    result = {
        "development_pairs": len(dev_pairs), "test_pairs": len(test_pairs),
        "test_recordings_matched": len({p["row"] for p in test_pairs}),
        "unique_test_waveform_controls": len({(p["row"], p["waveform_edit"]) for p in test_pairs}),
        "test_mean_impact_difference": float(np.mean([
            abs(p["encoder_impact"] - p["waveform_impact"]) for p in test_pairs
        ])),
    }
    for name, names, x, y in (
        ("logits_only", OUTPUT_FEATURES, dev_logits, test_logits),
        ("logits_and_token_tap", OUTPUT_FEATURES + TOKEN_FEATURES,
         torch.cat((dev_logits, dev_tokens), dim=1), torch.cat((test_logits, test_tokens), dim=1)),
    ):
        probe = fit_logistic(x, dev_y)
        result[name] = {
            "balanced_accuracy": float(balanced_accuracy_score(test_y, probe.predict(y))),
            "auroc": float(roc_auc_score(test_y, probe.predict_proba(y)[:, 1])),
            "probe_parameters": probe_parameters(probe, names, y),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--monitor-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    torch.manual_seed(72)
    model = WeaverSOMAU3(audio_encoder="logmel", core_dim=64, depth=1, ff_dim=128).eval()
    monitor = model.load_audio_self_monitor(args.monitor_checkpoint)
    if model.multimodal_gain.item() != 0 or model.quantum_gain.item() != 0:
        raise ValueError("the isolated fusion replay expects both initial gates to be zero")
    rows = prepare_data(args.data_dir, 8)
    development = [row for row in rows if int(row["fold"]) <= 3]
    test_rows = [row for row in rows if row["fold"] == "5"]
    if {r["src_file"] for r in development} & {r["src_file"] for r in test_rows}:
        raise ValueError("original recordings overlap between development and test")
    dev_wave, _, _ = read_audio(args.data_dir, development)
    test_wave, labels, _ = read_audio(args.data_dir, test_rows)

    dev = replay(model, dev_wave)
    dev_sites = [replay(model, dev_wave, edit=attenuate(group, STANDARD_GAIN)) for group in range(len(GROUPS))]
    dev_matches = match_controls(dev, dev_sites, waveform_controls(model, dev_wave, 201))
    original = replay(model, test_wave)
    sham = replay(model, test_wave, edit=unchanged)
    for key in Replay.__dataclass_fields__:
        torch.testing.assert_close(getattr(sham, key), getattr(original, key), atol=0, rtol=0)
    test_sites = [replay(model, test_wave, edit=attenuate(group, STANDARD_GAIN)) for group in range(len(GROUPS))]
    weaker_sites = [replay(model, test_wave, edit=attenuate(group, WEAK_GAIN)) for group in range(len(GROUPS))]
    test_matches = match_controls(original, test_sites, waveform_controls(model, test_wave, 205))

    distorted_wave, _ = distort(test_wave, "echo_bandlimit", noise_seed=207)
    distorted = replay(model, distorted_wave)
    restored = replay(model, distorted_wave, edit=restore(original))
    torch.testing.assert_close(restored.logits, original.logits, atol=0, rtol=0)
    patches = [replay(model, distorted_wave, edit=restore(original, group)) for group in range(len(GROUPS))]

    closed_mask = replay(model, test_wave, mask_audio_at_fusion=True)
    torch.testing.assert_close(closed_mask.state, original.state, atol=0, rtol=0)
    opened = deepcopy(model)
    with torch.no_grad():
        opened.multimodal_gain.fill_(FUSION_TEST_GAIN)
    open_ref = replay(opened, test_wave)
    open_mask = replay(opened, test_wave, mask_audio_at_fusion=True)
    for ref, masked in ((original, closed_mask), (open_ref, open_mask)):
        torch.testing.assert_close(masked.logits, ref.logits, atol=0, rtol=0)
        torch.testing.assert_close(masked.monitor, ref.monitor, atol=0, rtol=0)
    after = replay(model, test_wave)
    torch.testing.assert_close(after.logits, original.logits, atol=0, rtol=0)
    torch.testing.assert_close(after.state, original.state, atol=0, rtol=0)
    cases = {**dict(zip(GROUPS, test_sites)), "encoder_sham": sham, "waveform_echo_bandlimit": distorted,
             "restore_all_audio_tokens": restored,
             **{f"restore_{name}": case for name, case in zip(GROUPS, patches)}}
    report = {
        "monitor_checkpoint_sha256": monitor.loaded_checkpoint_sha256, "torch_version": torch.__version__,
        "interventions": {"groups": GROUPS, "channels_per_group": GROUP_SIZE,
                          "standard_retained_activation": STANDARD_GAIN, "weak_retained_activation": WEAK_GAIN,
                          "matched_absolute_tolerance": 0.025, "matched_relative_tolerance": 0.25},
        "development_recordings": len(development), "test_recordings": len(test_rows),
        "development_folds": [1, 2, 3], "test_fold": 5,
        "fold_5_previously_evaluated": True, "sham_exact": True, "restore_logits_exact": True,
        "baseline_recovered_after_hooks": True,
        "audio_effects": {name: effects(original, case, labels) for name, case in cases.items()},
        "site_localization": site_localization(dev, dev_sites, original, test_sites, weaker_sites),
        "matched_cause_probe": distinguish_causes(dev_matches, test_matches),
        "fusion_mask": {"default_gain": effects(original, closed_mask, labels),
                        "test_gain": FUSION_TEST_GAIN, "manually_enabled": effects(open_ref, open_mask, labels)},
        "manifest": [{"file": row["filename"], "sha256": sha256(args.data_dir / "audio" / row["filename"]),
                      "fold": int(row["fold"])} for row in development + test_rows],
    }
    group_deltas = {name: token_group_shift(original, case) for name, case in cases.items()}
    fusion_deltas = {
        "default_attention_l2": (closed_mask.attention - original.attention).norm(dim=1),
        "default_state_l2": (closed_mask.state - original.state).norm(dim=1),
        "test_gate_attention_l2": (open_mask.attention - open_ref.attention).norm(dim=1),
        "test_gate_state_l2": (open_mask.state - open_ref.state).norm(dim=1),
    }
    evidence = {
        "recordings": [
            {"file": row["filename"], "target": AUDIO_CLASSES[int(labels[i])],
             "baseline_logits": original.logits[i].tolist(), "baseline_confidence": float(original.monitor[i]),
             "conditions": {
                 name: {"logits": case.logits[i].tolist(), "confidence": float(case.monitor[i]),
                        "group_relative_rms_change": group_deltas[name][i].tolist()}
                 for name, case in cases.items()
             },
             "fusion_audio_mask": {key: float(values[i]) for key, values in fusion_deltas.items()}}
            for i, row in enumerate(test_rows)
        ],
        "matched_waveform_pairs": [
            {"file": test_rows[pair["row"]]["filename"],
             **{key: value for key, value in pair.items() if key != "row"}}
            for pair in test_matches[-1]
        ],
    }
    for name, payload in (("intervention_summary.json", report), ("paired_replays.json", evidence)):
        (args.run_dir / name).write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2))


if __name__ == "__main__":
    main()
