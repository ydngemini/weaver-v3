"""Train a field ranker and audit extractive responses on controlled splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import itertools
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from grounded_pointer import (
    UNSUPPORTED, FieldRanker, Route, features, load_pointer, matching_fields, parse_prompt, select_evidence,
)
from train_grounded_text import VALUES, UNKNOWN, render

SEED = 4417
ALIASES = {
    "animal": ("animal", "pet", "creature", "species"),
    "color": ("color", "colour", "hue", "shade"),
    "food": ("food", "meal", "dish", "snack"),
    "tool": ("tool", "implement", "device", "instrument"),
    "city": ("city", "town", "municipality", "metro"),
    "shape": ("shape", "outline", "form", "figure"),
}
NOVEL_ALIASES = {"animal": "fauna", "color": "pigment", "food": "cuisine",
                 "tool": "utensil", "city": "borough", "shape": "silhouette"}
TRAIN_TEMPLATES = (
    "What is the {field}?", "Which {field} is listed?", "Find the {field}.",
    "Read the {field} entry.", "Which {field} appears in the record?",
    "The {field}, please.", "The records show which {field}?",
)
CALIBRATION_TEMPLATES = ("Could you name the {field}?", "What is written for the {field}?")
TEST_TEMPLATES = (
    "From these notes, give me the {field}.", "Tell me the recorded {field}.",
    "Can you point to the {field} in the card?", "What did they record as the {field}?",
)
TRAIN_UNKNOWN = ("address", "phone number", "height", "speed", "account", "warranty",
                 "noise", "size", "age", "price", "music", "temperature", "doctor", "country",
                 "personality", "manufacturer")
CALIBRATION_UNKNOWN = ("payment date", "score", "postal code", "hours", "manufacturer name")
TEST_UNKNOWN = ("email", "school", "deadline", "job title", "favourite song", "exchange rate")
LABELS = tuple(VALUES) + (UNSUPPORTED,)


def question_grid(templates, unknown):
    rows = [(template.format(field=alias), key)
            for key, aliases in ALIASES.items() for alias in aliases for template in templates]
    rows.extend((template.format(field=unsupported), UNSUPPORTED)
                for unsupported in unknown for template in templates)
    for first, second in itertools.combinations(VALUES, 2):
        rows.extend((template.format(field=f"{first} and {second}"), UNSUPPORTED) for template in templates)
    return rows


def fit(rows):
    vocabulary = sorted(set().union(*(features(q) for q, _ in rows)))
    network = FieldRanker(LABELS, vocabulary, lexicon=ALIASES)
    data = network.vectors([q for q, _ in rows])
    labels = torch.tensor([LABELS.index(label) for _, label in rows])
    opt = torch.optim.AdamW(network.parameters(), lr=.04)
    for _ in range(220):
        logits = network(data)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return network.eval()


def calibrate(network, rows):
    questions = [q for q, _ in rows]
    labels = [None if field == UNSUPPORTED else field for _, field in rows]
    raw = network.routes(questions, 0., enforce_scope=False)
    cuts = [0.] + sorted({item.margin + 1e-8 for item in raw if item.field is not None})
    candidates = []
    for cut in cuts:
        accepted = [route.field if route.margin >= cut else None for route in raw]
        if not any(choice is not None and choice != gold for choice, gold in zip(accepted, labels)):
            correct_known = sum(choice == gold and gold is not None for choice, gold in zip(accepted, labels))
            candidates.append((correct_known, -cut))
    _, neg_cut = max(candidates)
    return -neg_cut


def replay_previous(path):
    selected = [json.loads(line) for line in path.read_text().splitlines()]
    cases = ("held_out", "unseen_value", "original", "swapped", "removed")
    grouped = {name: [] for name in cases}
    for row in selected:
        if row["model"] == "full_replay" and row["case"] in grouped:
            row["required_status"] = "missing_evidence" if row["expected"] == UNKNOWN else "answered"
            grouped[row["case"]].append(row)
    if any(len(grouped[name]) != count for name, count in zip(cases, (256, 96, 96, 96, 96))):
        raise ValueError("previous prediction trace does not match the recorded benchmark")
    return {f"previous_{name}": rows for name, rows in grouped.items()}


def changed_question(row, question):
    lines = row["prompt"].splitlines()
    return {**row, "prompt": "\n".join(lines[:-2] + [f"Question: {question}", "Answer:"])}


def stress_cards():
    rng = random.Random(SEED + 1)
    suites = {name: [] for name in ("known_synonym", "unseen_synonym", "novel_literal",
                                  "conflict", "agreeing_sources", "unsupported", "ambiguous")}
    for field in VALUES:
        for i in range(16):
            others = rng.sample([name for name in VALUES if name != field], 2)
            fields = [field, *others]
            rng.shuffle(fields)
            text = f"Café–{field}–{i + 97} 東京"
            cards = [(ident, key, text if key == field else rng.choice(VALUES[key]))
                     for ident, key in zip("ABC", fields)]
            template = TEST_TEMPLATES[i % len(TEST_TEMPLATES)]
            row = render(cards, field, template)
            row["required_status"] = "answered"
            suites["novel_literal"].append(row)
            synonym = ALIASES[field][1 + i % (len(ALIASES[field]) - 1)]
            suites["known_synonym"].append(changed_question(row, template.format(field=synonym)))
            suites["unseen_synonym"].append(
                changed_question(row, template.format(field=NOVEL_ALIASES[field])))
            conflict = render(cards + [("D", field, "conflicting entry")], field, template)
            suites["conflict"].append({**conflict, "expected": UNKNOWN, "required_status": "conflicting_evidence"})
            agreeing = render(cards + [("D", field, text)], field, template)
            suites["agreeing_sources"].append({**agreeing, "required_status": "answered"})
            question = template.format(field=TEST_UNKNOWN[i % len(TEST_UNKNOWN)])
            unsupported = changed_question(row, question)
            suites["unsupported"].append({**unsupported, "expected": UNKNOWN,
                                          "required_status": "unsupported_question"})
            question = template.format(field=f"{field} and {others[0]}")
            ambiguous = changed_question(row, question)
            suites["ambiguous"].append({**ambiguous, "expected": UNKNOWN,
                                        "required_status": "unsupported_question"})
    return suites


def dictionary_routes(questions, lexicon):
    routes = []
    for question in questions:
        match = matching_fields(question, lexicon)
        routes.append(Route(match[0], 1., 1., match) if len(match) == 1 else Route(None, 0., 0., match))
    return routes


def assess(rows, routes, parsed, case, traces):
    summary = {"rows": len(rows), "exact": 0, "correct_status": 0,
               "answered": 0, "wrong_answers": 0, "expected_answerable": 0}
    outcomes = []
    for row, route, (_, evidence) in zip(rows, routes, parsed):
        answer = select_evidence(route, evidence)
        equal = answer.text() == row["expected"]
        summary["exact"] += equal
        summary["correct_status"] += answer.status == row["required_status"]
        summary["answered"] += answer.status == "answered"
        summary["wrong_answers"] += answer.status == "answered" and not equal
        summary["expected_answerable"] += row["required_status"] == "answered"
        for source in answer.evidence:
            assert row["prompt"][source.start:source.end] == source.value
        if case == "agreeing_sources" and answer.status == "answered":
            assert len(answer.evidence) == 2
        outcomes.append(equal and answer.status == row["required_status"])
        if traces is not None:
            traces.write(json.dumps({
                "case": case, "prompt": row["prompt"], "expected": row["expected"],
                "expected_status": row["required_status"],
                "response": answer.text(), "decision": asdict(answer),
            }, ensure_ascii=False) + "\n")
    return summary, outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--previous-predictions", type=Path, required=True)
    p.add_argument("--external-oos", type=Path, help="optional CLINC150 data_full.json for an OOS stress slice")
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    training = question_grid(TRAIN_TEMPLATES, TRAIN_UNKNOWN)
    validation = question_grid(CALIBRATION_TEMPLATES, CALIBRATION_UNKNOWN)
    assert not {q for q, _ in training} & {q for q, _ in validation}
    model = fit(training)
    threshold = calibrate(model, validation)
    suites = replay_previous(args.previous_predictions) | stress_cards()
    extra_hash = None
    if args.external_oos:
        raw = args.external_oos.read_bytes()
        extra_hash = sha256(raw).hexdigest()
        unrelated = json.loads(raw)["oos_test"]
        records = [(ident, field, VALUES[field][0]) for ident, field in zip("ABCDEF", VALUES)]
        context = render(records, "city", "What is the {field}?")
        suites["external_unrelated_queries"] = [
            {**changed_question(context, " ".join(query.split())), "expected": UNKNOWN,
             "required_status": "unsupported_question"} for query, _ in unrelated]
    args.run_dir.mkdir(parents=True, exist_ok=False)
    scores = {}
    outcomes = {}
    path = args.run_dir / "soma_grounded_pointer.pt"
    torch.save({"architecture": "soma-u3-structured-evidence-v1", "labels": model.labels,
                "vocabulary": model.vocabulary, "lexicon": model.lexicon, "margin_threshold": threshold,
                "weights": model.state_dict()}, path)
    loaded, threshold = load_pointer(path)
    with (args.run_dir / "exact_evidence_trace.jsonl").open("x", encoding="utf-8") as traces:
        for case, rows in suites.items():
            parsed = [parse_prompt(row["prompt"]) for row in rows]
            questions = [q for q, _ in parsed]
            routes = loaded.routes(questions, threshold)
            assert routes == model.routes(questions, threshold)
            trained, outcomes[case] = assess(rows, routes, parsed, case, traces)
            raw, _ = assess(rows, loaded.routes(questions, threshold, enforce_scope=False), parsed, case, None)
            literal, _ = assess(rows, dictionary_routes(questions, {key: (key,) for key in VALUES}),
                                parsed, case, None)
            dictionary, _ = assess(rows, dictionary_routes(questions, ALIASES), parsed, case, None)
            scores[case] = {"trained_pointer": trained, "unguarded_pointer": raw,
                            "literal_match_pointer": literal, "alias_dictionary_pointer": dictionary}
            if case.startswith("previous_"):
                scores[case]["previous_free_decoder_exact"] = sum(
                    row["generated"] == row["expected"] for row in rows)
    trio = zip(*(outcomes[f"previous_{k}"] for k in ("original", "swapped", "removed")))
    report = {
        "question_fit_examples": len(training), "calibration_examples": len(validation),
        "seed": SEED, "margin_threshold": threshold,
        "previous_prediction_trace_sha256": sha256(args.previous_predictions.read_bytes()).hexdigest(),
        "external_oos_corpus_sha256": extra_hash,
        "scores": scores, "previous_intervention_groups_all_three_correct": sum(all(t) for t in trio),
        "previous_intervention_groups": len(suites["previous_original"]),
    }
    (args.run_dir / "grounded_pointer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
