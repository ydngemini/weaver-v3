"""Distill a local teacher into a copy of SOMA's existing intent head."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from adaptive_inputs import AdaptiveTextEncoder, masked_mean
from benchmark_adaptive_text import MAX_BYTES, encode, typo_rows

SEED = 2333
SOFT_WEIGHT = .7
TEMPERATURE = 2.


def fit_teacher(training, names):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.svm import LinearSVC

    teacher = make_pipeline(FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)),
    ]), LinearSVC(random_state=SEED))
    teacher.fit(texts(training), [label for _, label in training])
    if list(teacher.classes_) != names:
        raise ValueError("teacher class order differs from the parent checkpoint")
    return teacher


def texts(rows):
    return [text.encode("utf-8")[:MAX_BYTES].decode("utf-8", errors="ignore") for text, _ in rows]


def margins(teacher, rows):
    return torch.tensor(teacher.decision_function(texts(rows)), dtype=torch.float32)


@torch.no_grad()
def features(encoder, packed, budget):
    x, valid, labels = packed
    pieces = []
    for offset in range(0, len(x), 128):
        bx, mask = x[offset:offset + 128], valid[offset:offset + 128]
        length = int(mask.sum(-1).max())
        result = encoder(bx[:, :length], valid=mask[:, :length], limit=budget)
        pieces.append(masked_mean(result.tokens, result.valid))
    return torch.cat(pieces), labels


def fit_head(base, training, validation, soft=None):
    head = deepcopy(base).train()
    opt = torch.optim.AdamW(head.parameters(), lr=.003)
    rng = torch.Generator().manual_seed(SEED)
    tx, ty = training
    vx, vy = validation
    best = (-1, float("-inf"))
    begin = perf_counter()
    for epoch in range(16):
        permutation = torch.randperm(len(tx), generator=rng)
        for ids in permutation.split(128):
            logits = head(tx[ids])
            hard = F.cross_entropy(logits, ty[ids])
            if soft is None:
                loss = hard
            else:
                forecast = F.log_softmax(logits / TEMPERATURE, dim=-1)
                guide = F.kl_div(forecast, soft[ids], reduction="batchmean") * TEMPERATURE ** 2
                loss = (1 - SOFT_WEIGHT) * hard + SOFT_WEIGHT * guide
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = head(vx)
            measure = (int((logits.argmax(-1) == vy).sum()), -float(F.cross_entropy(logits, vy)))
            if measure > best:
                best = measure
                epoch_selected = epoch + 1
                weights = deepcopy(head.state_dict())
    head.load_state_dict(weights)
    return head.eval(), epoch_selected, perf_counter() - begin


@torch.no_grad()
def assess(scores, labels, scale=1.):
    confidence = {name: (logits * scale).softmax(-1).max(-1).values for name, logits in scores.items()}
    threshold = torch.quantile(confidence["calibration"], .05)
    cuts = {"validation_threshold": float(threshold), "slices": {}}
    for name, target in labels.items():
        accept = confidence[name] >= threshold
        correct = scores[name].argmax(-1) == target
        cuts["slices"][name] = {
            "n": len(target), "correct": int(correct.sum()), "accepted": int(accept.sum()),
            "correct_accepted": int((correct & accept).sum()),
        }
    return cuts


def save_candidate(parent, head, path, base_hash, method, stats):
    expected = parent["weights"]
    weights = dict(expected)
    for name, value in head.state_dict().items():
        weights[f"intent.{name}"] = value
    assert set(expected) == set(weights)
    assert all(expected[key].shape == weights[key].shape and expected[key].dtype == weights[key].dtype
               for key in expected)
    changed = sorted(key for key in expected if not torch.equal(expected[key], weights[key]))
    assert changed == ["intent.bias", "intent.weight"]
    payload = {**parent, "weights": weights, "distillation_parent_sha256": base_hash,
               "adaptation_method": method, "confidence_threshold": stats["validation_threshold"]}
    torch.save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=True)
    assert all(torch.equal(reloaded["weights"][k], t) for k, t in weights.items())
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    data = args.dataset.read_bytes()
    source_checkpoint = args.parent.read_bytes()
    digest = sha256(source_checkpoint).hexdigest()
    splits = json.loads(data)
    base = torch.load(args.parent, map_location="cpu", weights_only=True)
    names = list(base["classes"])
    encoder = AdaptiveTextEncoder(base["fusion_dim"]).eval().requires_grad_(False)
    encoder.load_state_dict(base["weights"], strict=True)
    encoder.intent.requires_grad_(True)
    # Stratified original validation split provides disjoint selection and calibration data.
    counters = {name: 0 for name in names}
    selection, calibration = [], []
    for text, name in splits["val"]:
        targets = selection if counters[name] % 2 == 0 else calibration
        targets.append((text, name))
        counters[name] += 1
    parts = {"train": splits["train"], "select": selection, "calibration": calibration,
             "clean": splits["test"], "character_swap": list(typo_rows(splits["test"])),
             "out_of_scope": splits["oos_test"]}
    teacher = fit_teacher(parts["train"], names)
    teacher_scores = {key: margins(teacher, rows) for key, rows in parts.items()}
    all_features = {key: features(encoder, encode(rows, names), base["token_budget"])
                    for key, rows in parts.items()}
    val_y = all_features["calibration"][1]
    class_scales = (.5, 1., 2., 4., 6., 8., 12., 16., 24., 32.)
    scale = min(class_scales, key=lambda s: F.cross_entropy(teacher_scores["calibration"] * s, val_y).item())
    soft = (teacher_scores["train"] * scale / TEMPERATURE).softmax(-1)
    eval_names = ("calibration", "clean", "character_swap", "out_of_scope")
    labels = {k: all_features[k][1] for k in eval_names}
    result = {"teacher": assess({k: teacher_scores[k] for k in eval_names}, labels, scale)}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    changed = {}
    for method in ("parent", "label_only", "distilled"):
        head = encoder.intent
        if method == "parent":
            stats = {}
        else:
            head, epoch, seconds = fit_head(encoder.intent, all_features["train"], all_features["select"],
                                            soft=soft if method == "distilled" else None)
            stats = {"chosen_epoch": epoch, "training_seconds_head_only": seconds}
        with torch.no_grad():
            scores = {name: head(all_features[name][0]) for name in eval_names}
        result[method] = {**stats, **assess(scores, labels)}
        if method != "parent":
            path = args.run_dir / f"soma_intent_{method}.pt"
            changed[method] = save_candidate(base, head, path, digest, method, result[method])
        print(method, result[method]["slices"], flush=True)
    assert args.parent.read_bytes() == source_checkpoint
    report = {"parent_sha256": digest, "data_sha256": sha256(data).hexdigest(), "seed": SEED,
              "trained_parameters": sum(p.numel() for p in encoder.intent.parameters()),
              "total_parameters": sum(p.numel() for p in encoder.parameters()),
              "train_rows": len(parts["train"]), "selection_rows": len(selection),
              "calibration_rows": len(calibration), "teacher": "word and character TF-IDF with linear SVM",
              "teacher_margin_multiplier": scale, "soft_weight": SOFT_WEIGHT, "temperature": TEMPERATURE,
              "candidate_changes": changed, "parent_bytes_preserved": True, "results": result}
    (args.run_dir / "intent_distillation_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
