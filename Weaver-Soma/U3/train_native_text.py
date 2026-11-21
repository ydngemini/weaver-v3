"""Compare byte and byte-level BPE native decoders on frozen TinyStories subsets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
from time import perf_counter, perf_counter_ns

import torch
from torch.nn import functional as F

from native_text_decoder import DecoderConfig, NativeTextDecoder, complete

SEED = 8391
BOS, EOS = "<soma_bos>", "<soma_eos>"
SPLIT_MARKER = b"<|endoftext|>"
CONTEXT = 96
BATCH = 8


class Bytes:
    bos, eos, size = 256, 257, 258
    spec = None

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(token for token in ids if token < 256).decode("utf-8", errors="replace")


class ByteBPE:
    def __init__(self, documents=None, spec=None):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        if spec is None:
            self.backend = Tokenizer(models.BPE())
            self.backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            self.backend.decoder = decoders.ByteLevel()
            self.backend.train_from_iterator(documents, trainers.BpeTrainer(
                vocab_size=512, special_tokens=[BOS, EOS], show_progress=False,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        else:
            self.backend = Tokenizer.from_str(spec)
        self.bos = self.backend.token_to_id(BOS)
        self.eos = self.backend.token_to_id(EOS)
        self.size = self.backend.get_vocab_size()
        self.spec = self.backend.to_str()

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(ids)


def documents(path, limit, *, exclude=(), skip_first=False):
    source = path.read_bytes()
    complete_records = source.split(SPLIT_MARKER)[int(skip_first):-1]
    seen = set(exclude)
    selected = []
    for raw in complete_records:
        text = raw.decode("utf-8").strip()
        if text and text not in seen:
            selected.append(text)
            seen.add(text)
            if len(selected) == limit:
                break
    if len(selected) != limit:
        raise ValueError(f"{path} contains only {len(selected)} of {limit} required complete distinct stories")
    return selected, sha256(source).hexdigest()


def packed(corpus, codec):
    tokens = []
    for story in corpus:
        tokens.extend([codec.bos, *codec.encode(story), codec.eos])
    return torch.tensor(tokens, dtype=torch.long)


def validation_blocks(corpus, codec):
    pairs = []
    for story in corpus:
        tokens = [codec.bos, *codec.encode(story)]
        for i in range(0, len(tokens) - 1, CONTEXT):
            span = tokens[i:i + CONTEXT + 1]
            pad = CONTEXT + 1 - len(span)
            pairs.append((span[:-1] + [codec.eos] * pad, span[1:] + [-100] * pad))
    inputs, targets = zip(*pairs)
    return torch.tensor(inputs), torch.tensor(targets)


@torch.inference_mode()
def bits_per_byte(model, blocks, byte_count):
    total = 0.
    for x, y in zip(blocks[0].split(24), blocks[1].split(24)):
        logits, _ = model(x)
        total += float(F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="sum"))
    return total / (math.log(2) * byte_count)


def unigram_bits(tokens, vocab, y, byte_count):
    counts = torch.bincount(tokens, minlength=vocab).double() + 1
    probabilities = counts / counts.sum()
    return float(-probabilities[y[y != -100]].log2().sum() / byte_count)


@torch.inference_mode()
def measure_decode(model, row):
    prefix = row[:40].unsqueeze(0)
    continuation = row[40:64].split(1)

    def uncached():
        context = prefix
        scores = []
        for part in continuation:
            context = torch.cat((context, part[None]), dim=-1)
            logits, _ = model(context)
            scores.append(logits[:, -1])
        return torch.stack(scores, dim=1)

    def cached():
        _, cache = model(prefix, use_cache=True)
        scores = []
        for part in continuation:
            logits, cache = model(part[None], cache=cache, use_cache=True)
            scores.append(logits[:, -1])
        return torch.stack(scores, dim=1)

    reference = uncached()
    optimized = cached()
    torch.testing.assert_close(optimized, reference, atol=2e-5, rtol=1e-4)
    worst_difference = (optimized - reference).abs().max().item()
    methods = (("uncached", uncached), ("cached", cached))
    for _ in range(5):
        for _, method in methods:
            method()
    elapsed = {name: [] for name, _ in methods}
    repetitions = 80
    for i in range(repetitions):
        for name, method in methods[::1 if i % 2 == 0 else -1]:
            begin = perf_counter_ns()
            method()
            elapsed[name].append((perf_counter_ns() - begin) / 1e6)
    result = {name: {"median_ms": median(times), "p95_ms": sorted(times)[math.ceil(.95 * repetitions) - 1]}
              for name, times in elapsed.items()}
    return {"prefill_tokens": 40, "decode_tokens": len(continuation), "includes_prefill": True,
            "repetitions": repetitions, "max_logit_absolute_difference": worst_difference, **result}


def run(codec, train, validation, steps, output, name):
    torch.manual_seed(SEED)
    rng = torch.Generator().manual_seed(SEED)
    all_tokens = packed(train, codec)
    held_out = validation_blocks(validation, codec)
    byte_count = sum(len(text.encode("utf-8")) for text in validation)
    config = DecoderConfig(codec.size, context=CONTEXT)
    model = NativeTextDecoder(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.02)
    offset = torch.arange(CONTEXT + 1)
    checkpoints = [{"training_tokens": 0, "bits_per_byte": bits_per_byte(model.eval(), held_out, byte_count)}]
    seconds = 0.
    for step in range(1, steps[-1] + 1):
        start = perf_counter()
        model.train()
        starts = torch.randint(len(all_tokens) - CONTEXT, (BATCH,), generator=rng)
        block = all_tokens[starts[:, None] + offset]
        logits, _ = model(block[:, :-1])
        loss = F.cross_entropy(logits.flatten(0, 1), block[:, 1:].flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        seconds += perf_counter() - start
        if step in steps:
            bpb = bits_per_byte(model.eval(), held_out, byte_count)
            print(name, step, round(bpb, 4), round(seconds, 2), flush=True)
            checkpoints.append({"training_tokens": step * BATCH * CONTEXT, "bits_per_byte": bpb,
                                "training_seconds": seconds})
    saved = {"format_version": 1, "architecture": "soma-native-causal-v1", "tokenizer": name,
             "tokenizer_json": codec.spec, "config": asdict(config), "weights": model.state_dict()}
    path = output / f"soma_native_{name}.pt"
    torch.save(saved, path)
    restore, loaded_codec = load_native(path)
    prompt = "Once upon a time, a little robot"
    pieces = complete(restore, [loaded_codec.bos, *loaded_codec.encode(prompt)], max_new=48,
                      stop=loaded_codec.eos, generator=torch.Generator().manual_seed(SEED))
    return {
        "parameter_count": sum(t.numel() for t in model.parameters()),
        "vocab_size": codec.size, "encoded_training_corpus_tokens": len(all_tokens),
        "validation_text_bytes": byte_count,
        "unigram_bits_per_byte": unigram_bits(all_tokens, codec.size, held_out[1], byte_count),
        "learning_curve": checkpoints,
        "decode_one_cpu_thread": speed_one_thread(restore, held_out[0][0]),
        "sample_prompt": prompt, "sample_continuation": loaded_codec.decode(pieces),
        "checkpoint_sha256": sha256(path.read_bytes()).hexdigest(),
    }


def speed_one_thread(model, row):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return measure_decode(model, row)
    finally:
        torch.set_num_threads(threads)


def load_native(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["architecture"] != "soma-native-causal-v1" or saved["format_version"] != 1:
        raise ValueError("unsupported SOMA text checkpoint")
    tokenizer = saved["tokenizer"]
    if tokenizer not in ("bytes", "bpe"):
        raise ValueError("unsupported tokenizer")
    codec = Bytes() if tokenizer == "bytes" else ByteBPE(spec=saved["tokenizer_json"])
    model = NativeTextDecoder(DecoderConfig(**saved["config"])).eval()
    if model.config.vocab_size != codec.size:
        raise ValueError("checkpoint and tokenizer vocabulary sizes differ")
    model.load_state_dict(saved["weights"], strict=True)
    return model, codec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    train, train_hash = documents(args.train, 1536)
    # The published validation text starts inside a story.
    valid, val_hash = documents(args.validation, 128, exclude=train, skip_first=True)
    assert not set(train) & set(valid)
    args.run_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, codec in (("bytes", Bytes()), ("bpe", ByteBPE(train))):
        results[name] = run(codec, train, valid, (60, 180, 420), args.run_dir, name)
    report = {"source_revision": args.source_revision, "source_sha256": {"train": train_hash, "validation": val_hash},
              "training_documents": len(train), "validation_documents": len(valid),
              "training_text_bytes": sum(len(story.encode("utf-8")) for story in train),
              "context_tokens": CONTEXT, "batch_size": BATCH, "seed": SEED, "training_cpu_threads": 2,
              "torch_version": str(torch.__version__), "results": results}
    (args.run_dir / "native_text_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
