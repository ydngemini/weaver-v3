"""Compare full SFT and native LoRA on synthetic source-grounded replies."""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import random
from time import perf_counter

import torch
from torch.nn import functional as F

from decoder_lora import add_lora, adapter_targets, adapter_weights, merge_lora
from train_native_text import bits_per_byte, documents, load_native, packed, validation_blocks

SEED = 1867
RANK = 8
ALPHA = 16
VALUES = {
    "animal": ("dog", "cat", "fox", "bear", "bird", "rabbit", "otter"),
    "color": ("red", "blue", "green", "pink", "black", "silver", "gold"),
    "food": ("bread", "cake", "apple", "soup", "rice", "lemon", "beans"),
    "tool": ("bell", "cup", "pen", "book", "box", "rope", "brush"),
    "city": ("Rome", "Paris", "Miami", "London", "Tokyo", "Dover", "Boston"),
    "shape": ("circle", "square", "star", "oval", "heart", "diamond", "triangle"),
}
QUESTIONS = ("What is the {field}?", "Which {field} is listed?", "Find the {field}.")
CITATIONS = "ABCD"
UNKNOWN = "unknown"


def fixture(seed, count, used, *, condition="standard"):
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        query = rng.choice(tuple(VALUES))
        fields = rng.sample(tuple(VALUES), rng.choice((2, 3, 4)))
        answerable = rng.random() < .7
        if answerable and query not in fields:
            fields[rng.randrange(len(fields))] = query
        if not answerable and query in fields:
            fields[fields.index(query)] = rng.choice([field for field in VALUES if field not in fields])
        rng.shuffle(fields)
        ids = rng.sample(CITATIONS, len(fields))
        records = [(ident, field, rng.choice(VALUES[field][:5])) for ident, field in zip(ids, fields)]
        phrasing = rng.choice(QUESTIONS)
        if condition == "unseen_phrase":
            phrasing = "Tell me the recorded {field}."
        if condition == "unseen_value":
            if query not in fields:
                continue
            records = [(ident, field, rng.choice(VALUES[field][5:]) if field == query else value)
                       for ident, field, value in records]
        row = render(records, query, phrasing)
        signature = (tuple(records), query)
        if signature not in used:
            used.add(signature)
            rows.append(row)
    return rows


def render(records, query, wording):
    prompt = "\n".join(f"[{ident}] {field}: {value}." for ident, field, value in records)
    prompt += f"\nQuestion: {wording.format(field=query)}\nAnswer:"
    matches = [(ident, value) for ident, field, value in records if field == query]
    expected = f"{matches[0][1]} [{matches[0][0]}]" if matches else UNKNOWN
    return {"prompt": prompt, "expected": expected, "records": records, "query": query, "wording": wording}


def intervene(rows, used):
    for row in rows:
        query = row["query"]
        if row["expected"] == UNKNOWN:
            continue
        records = row["records"]
        current = next(value for _, field, value in records if field == query)
        alternate = next(value for value in VALUES[query][:5] if value != current)
        changed = [(ident, field, alternate if field == query else value) for ident, field, value in records]
        remaining = [record for record in records if record[1] != query]
        signatures = [(tuple(items), query) for items in (changed, remaining)]
        if any(signature in used for signature in signatures):
            continue
        used.update(signatures)
        yield row, render(changed, query, row["wording"]), render(remaining, query, row["wording"])


def training_rows(rows, codec, context):
    data = []
    for row in rows:
        prefix = [codec.bos, *codec.encode(row["prompt"])]
        answer = [*codec.encode(" " + row["expected"]), codec.eos]
        all_tokens = prefix + answer
        if len(all_tokens) > context:
            raise ValueError("prompt and supervised response must fit the checkpoint context")
        targets = [-100] * (len(prefix) - 1) + answer
        data.append((all_tokens[:-1], targets))
    return data


def minibatch(rows, pad):
    x, y = zip(*rows)
    width = max(map(len, x))
    return (torch.tensor([row + [pad] * (width - len(row)) for row in x]),
            torch.tensor([row + [-100] * (width - len(row)) for row in y]))


@torch.inference_mode()
def decode(model, codec, rows):
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        prompt = [codec.bos, *codec.encode(row["prompt"])]
        buckets[len(prompt)].append((i, prompt))
    answers = [None] * len(rows)
    for length, bucket in buckets.items():
        for start in range(0, len(bucket), 48):
            batch = bucket[start:start + 48]
            current = torch.tensor([prompt for _, prompt in batch])
            limit = min(14, model.config.context - length)
            done = torch.zeros(len(batch), dtype=torch.bool)
            cache = None
            generated = []
            for _ in range(limit):
                logits, cache = model(current, cache=cache, use_cache=True)
                nxt = logits[:, -1].argmax(-1)
                nxt = torch.where(done, codec.eos, nxt)
                done |= nxt == codec.eos
                generated.append(nxt)
                current = nxt[:, None]
                if done.all():
                    break
            if not generated:
                raise ValueError("test prompt leaves no generation slots")
            for (i, _), content in zip(batch, torch.stack(generated, dim=1).tolist()):
                if codec.eos in content:
                    content = content[:content.index(codec.eos)]
                answers[i] = codec.decode(content).strip()
    return answers


def metrics(rows, answers):
    known = [i for i, row in enumerate(rows) if row["expected"] != UNKNOWN]
    unknown = [i for i, row in enumerate(rows) if row["expected"] == UNKNOWN]
    exact = [answer == row["expected"] for row, answer in zip(rows, answers)]
    return {
        "examples": len(rows), "exact": sum(exact), "known": len(known), "known_exact": sum(exact[i] for i in known),
        "unknown": len(unknown), "correct_abstentions": sum(exact[i] for i in unknown),
        "wrong_non_abstaining_answers": sum(not correct and answer != UNKNOWN for correct, answer in zip(exact, answers)),
    }


def fit(model, rows, rng_seed, steps, rate, pad, name, replay=None):
    rng = torch.Generator().manual_seed(rng_seed)
    replay_rng = torch.Generator().manual_seed(rng_seed + 1)
    replay_offsets = torch.arange(model.config.context + 1)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate, weight_decay=.01)
    samples, response_tokens, replay_tokens = 0, 0, 0
    begin = perf_counter()
    for step in range(1, steps + 1):
        ids = torch.randint(len(rows), (24,), generator=rng).tolist()
        x, y = minibatch([rows[i] for i in ids], pad)
        response_tokens += int((y != -100).sum())
        samples += len(ids)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if replay is not None:
            offsets = torch.randint(len(replay) - model.config.context, (2,), generator=replay_rng)
            block = replay[offsets[:, None] + replay_offsets]
            replay_logits, _ = model(block[:, :-1])
            replay_target = block[:, 1:]
            loss = loss + .35 * F.cross_entropy(replay_logits.flatten(0, 1), replay_target.flatten())
            replay_tokens += replay_target.numel()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        if step % 200 == 0 or step == steps:
            print(name, step, round(loss.item(), 4), flush=True)
    return {"elapsed_seconds": perf_counter() - begin, "training_row_draws": samples,
            "supervised_response_tokens": response_tokens, "learning_rate": rate,
            "additional_story_replay_tokens": replay_tokens}


def evaluate(name, model, codec, sets, story_blocks, byte_count, trace):
    model.eval()
    answers = {case: decode(model, codec, rows) for case, rows in sets.items()}
    for case, rows in sets.items():
        for row, answer in zip(rows, answers[case]):
            trace.write(json.dumps({"model": name, "case": case, "prompt": row["prompt"],
                                    "expected": row["expected"], "generated": answer}) + "\n")
    trio = zip(*(zip(sets[case], answers[case]) for case in ("original", "swapped", "removed")))
    sound = sum(all(answer == row["expected"] for row, answer in item) for item in trio)
    return {"slices": {case: metrics(rows, answers[case]) for case, rows in sets.items()},
            "all_three_counterfactuals_correct": sound, "counterfactual_groups": len(sets["original"]),
            "story_validation_bits_per_byte": bits_per_byte(model, story_blocks, byte_count)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--story-validation", type=Path, required=True)
    p.add_argument("--story-replay-train", type=Path, help="also run full SFT with a story replay loss")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=800)
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    base, codec = load_native(args.checkpoint)
    base_hash = sha256(args.checkpoint.read_bytes()).hexdigest()
    used = set()
    training = fixture(SEED, 4096, used)
    canonical = fixture(SEED + 1, 256, used)
    triples = list(intervene(canonical, used))[:96]
    if len(triples) < 32:
        raise ValueError("too few isolated counterfactual groups")
    original, swapped, removed = zip(*triples)
    sets = {"held_out": canonical, "unseen_phrase": fixture(SEED + 2, 128, used, condition="unseen_phrase"),
            "unseen_value": fixture(SEED + 3, 96, used, condition="unseen_value"),
            "original": original, "swapped": swapped, "removed": removed}
    compiled = training_rows(training, codec, base.config.context)
    replay_stories, replay_sha = documents(args.story_replay_train, 1536) if args.story_replay_train else ([], None)
    replay_tokens = packed(replay_stories, codec) if replay_stories else None
    stories, val_hash = documents(args.story_validation, 128, exclude=replay_stories, skip_first=True)
    story_blocks = validation_blocks(stories, codec)
    byte_count = sum(len(s.encode("utf-8")) for s in stories)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    result = {}
    saved = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    methods = [("full", .002), ("lora", .005)]
    if replay_tokens is not None:
        methods.append(("full_replay", .002))
    with (args.run_dir / "predictions.jsonl").open("x", encoding="utf-8") as trace:
        result["base"] = evaluate("base", base, codec, sets, story_blocks, byte_count, trace)
        for name, rate in methods:
            model = deepcopy(base).train()
            if name == "lora":
                model = add_lora(model, rank=RANK, alpha=ALPHA)
            count = sum(x.numel() for x in model.parameters() if x.requires_grad)
            performance = fit(model, compiled, SEED + 4, args.steps, rate, codec.eos, name,
                              replay=replay_tokens if name == "full_replay" else None)
            result[name] = {"training": performance, "trainable_parameters": count,
                            **evaluate(name, model, codec, sets, story_blocks, byte_count, trace)}
            if name == "lora":
                with torch.inference_mode():
                    merged = merge_lora(model).eval()
                    sample, _ = minibatch(compiled[:5], codec.eos)
                    torch.testing.assert_close(merged(sample)[0], model(sample)[0], rtol=1e-4, atol=2e-5)
                for (adapted, attr), (original_block, field) in zip(adapter_targets(model), adapter_targets(base)):
                    original_state = getattr(original_block, field).state_dict()
                    frozen_state = getattr(adapted, attr).base.state_dict()
                    assert all(torch.equal(frozen_state[key], original_state[key]) for key in original_state)
                torch.save({"format_version": 1, "rank": RANK, "alpha": ALPHA, "base_sha256": base_hash,
                            "weights": adapter_weights(model)}, args.run_dir / "soma_grounded_lora_adapter.pt")
                model = merged
            torch.save({**saved, "weights": model.state_dict()}, args.run_dir / f"soma_grounded_{name}.pt")
            print(name, result[name]["slices"]["held_out"], flush=True)
    report = {"base_sha256": base_hash, "story_validation_sha256": val_hash,
              "story_replay_sha256": replay_sha, "seed": SEED,
              "training_unique_prompts": len(training), "optimizer_steps": args.steps, "rank": RANK,
              "alpha": ALPHA, "results": result, "lora_base_unchanged_and_merge_verified": True}
    (args.run_dir / "grounded_text_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
