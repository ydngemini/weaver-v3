"""Train local UTF-8 contextual intent readouts and measure scope and typos."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from adaptive_inputs import AdaptiveTextEncoder

DIM = 48
LIMIT = 48
MAX_BYTES = 128
SEED = 7727


def typo_rows(rows):
    rng = random.Random(SEED)
    for text, label in rows:
        spots = [j for j in range(len(text) - 1) if text[j:j + 2].isalpha() and text[j] != text[j + 1]]
        if spots:
            j = rng.choice(spots)
            text = text[:j] + text[j + 1] + text[j] + text[j + 2:]
        yield text, label


def encode(rows, classes):
    x = torch.zeros(len(rows), MAX_BYTES, dtype=torch.uint8)
    valid = torch.zeros_like(x, dtype=torch.bool)
    labels = torch.empty(len(rows), dtype=torch.long)
    label_lookup = {name: i for i, name in enumerate(classes)}
    for i, (text, label) in enumerate(rows):
        data = list(text.encode("utf-8")[:MAX_BYTES])
        x[i, :len(data)] = torch.tensor(data, dtype=torch.uint8)
        valid[i, :len(data)] = True
        labels[i] = label_lookup.get(label, -1)
    return x, valid, labels


@torch.no_grad()
def infer(model, data):
    x, valid, _ = data
    scores = []
    for i in range(0, len(x), 128):
        # Remove trailing storage padding before the local convolutions.
        selection = slice(i, i + 128)
        width = int(valid[selection].sum(-1).max())
        scores.append(model(x[selection, :width], valid=valid[selection, :width], limit=LIMIT).intent_logits)
    return torch.cat(scores)


def fit(train, validation, *, contextual, epochs):
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    model = AdaptiveTextEncoder(DIM, contextual=contextual)
    opt = torch.optim.AdamW(model.parameters(), lr=.003)
    x, mask, target = train
    best = (-1, float("-inf"))
    for epoch in range(epochs):
        for ids in torch.randperm(len(x), generator=generator).split(128):
            width = int(mask[ids].sum(-1).max())
            y = model(x[ids, :width], valid=mask[ids, :width], limit=LIMIT).intent_logits
            loss = F.cross_entropy(y, target[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val = infer(model, validation)
        score = (int((val.argmax(-1) == validation[2]).sum()), -float(F.cross_entropy(val, validation[2])))
        if score > best:
            best = score
            weights = deepcopy(model.state_dict())
            best_epoch = epoch + 1
    model.load_state_dict(weights)
    return model.eval(), best_epoch


def evaluate(model, validation, test, noisy, oos):
    threshold = torch.quantile(infer(model, validation).softmax(-1).amax(-1), .05)
    result = {"validation_confidence_threshold": float(threshold)}
    for name, data in (("clean", test), ("adjacent_character_swap", noisy), ("out_of_scope", oos)):
        logits = infer(model, data)
        known = logits.softmax(-1).amax(-1) >= threshold
        right = logits.argmax(-1) == data[2]
        result[name] = {"total": len(right), "correct": int(right.sum()),
                        "accepted": int(known.sum()), "correct_and_accepted": int((right & known).sum())}
    return result


def reference(splits):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, make_pipeline
    from sklearn.svm import LinearSVC

    def data(rows):
        return zip(*((text.encode('utf-8')[:MAX_BYTES].decode('utf-8', errors='ignore'), label)
                     for text, label in rows))

    pipeline = make_pipeline(FeatureUnion([
        ("words", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("characters", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2)),
    ]), LinearSVC(random_state=SEED))
    pipeline.fit(*data(splits["train"]))
    val, _ = data(splits["val"])
    threshold = np.quantile(pipeline.decision_function(val).max(-1), .05)
    stats = {"validation_margin_threshold": float(threshold)}
    for name, rows in (("clean", splits["test"]), ("adjacent_character_swap", typo_rows(splits["test"])),
                       ("out_of_scope", splits["oos_test"])):
        x, y = data(rows)
        margins = pipeline.decision_function(x)
        known = margins.max(-1) >= threshold
        correct = pipeline.classes_[margins.argmax(-1)] == np.array(y)
        stats[name] = {"total": len(y), "correct": int(correct.sum()), "accepted": int(known.sum()),
                       "correct_and_accepted": int((correct & known).sum())}
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--reference-only", action="store_true", help="run the optional TF-IDF/SVM baseline")
    args = p.parse_args()
    torch.set_num_threads(2)
    source = args.dataset.read_bytes()
    splits = json.loads(source)
    if args.reference_only:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        stats = reference(splits)
        (args.run_dir / "text_reference_report.json").write_text(json.dumps(stats, indent=2) + "\n")
        print(json.dumps(stats, indent=2))
        return
    classes = sorted(set(label for _, label in splits["train"]))
    if len(classes) != 150:
        raise ValueError("benchmark expects the 150 original CLINC intents")
    training = encode(splits["train"], classes)
    validation = encode(splits["val"], classes)
    test = encode(splits["test"], classes)
    noisy = encode(list(typo_rows(splits["test"])), classes)
    oos = encode(splits["oos_test"], classes)
    result = {}
    args.run_dir.mkdir(parents=True, exist_ok=False)
    for name, contextual in (("byte_bag", False), ("contextual", True)):
        model, epoch = fit(training, validation, contextual=contextual, epochs=args.epochs)
        stats = evaluate(model, validation, test, noisy, oos)
        result[name] = {"chosen_epoch": epoch, **stats}
        if contextual:
            torch.save({"format_version": 1, "fusion_dim": DIM, "token_budget": LIMIT,
                        "classes": classes, "weights": model.state_dict(),
                        "confidence_threshold": stats["validation_confidence_threshold"]},
                       args.run_dir / "adaptive_text_intent.pt")
        print(name, stats, flush=True)

    report = {"source_sha256": sha256(source).hexdigest(), "token_budget": LIMIT, "max_bytes": MAX_BYTES,
              "training_examples": len(training[2]), "validation_examples": len(validation[2]), "seed": SEED,
              "results": result}
    (args.run_dir / "adaptive_text_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
