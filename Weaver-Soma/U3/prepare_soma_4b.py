"""Build a bounded, revision-pinned FineWeb-Edu pilot corpus."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from hashlib import file_digest, sha256
from itertools import islice
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unicodedata

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


DATASET = "HuggingFaceFW/fineweb-edu"
SUBSET = "sample-10BT"
REVISION = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"
SPECIAL = ["<eos>"]


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def prepare(rows, output: Path, *, doc_limit: int, vocab_size: int, source: dict):
    if doc_limit < 64 or not 257 <= vocab_size <= 65_536:
        raise ValueError("use at least 64 documents and a vocabulary in [257, 65536]")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".soma-corpus-") as temporary:
        staging = Path(temporary)
        files = {split: staging / f"{split}.jsonl" for split in ("train", "validation")}
        seen = set()
        counts = Counter()
        with files["train"].open("w", encoding="utf-8") as train, \
                files["validation"].open("w", encoding="utf-8") as validation, \
                (staging / "documents.jsonl").open("w", encoding="utf-8") as index:
            for row in islice(rows, doc_limit):
                text = row.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue
                cleaned = unicodedata.normalize("NFKC", " ".join(text.split()))
                digest = sha256(cleaned.encode()).hexdigest()
                if digest in seen:
                    counts["duplicate"] += 1
                    continue
                seen.add(digest)
                split = "validation" if int(digest[:8], 16) % 100 < 5 else "train"
                counts[split] += 1
                record = {"sha256": digest, "split": split, "id": row.get("id"), "url": row.get("url")}
                index.write(json.dumps(record) + "\n")
                handle = validation if split == "validation" else train
                handle.write(json.dumps(text) + "\n")
        if min(counts["train"], counts["validation"]) < 1:
            raise ValueError("both splits need documents, increase the document sample")

        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
            special_tokens=SPECIAL, show_progress=False,
        )
        with files["train"].open(encoding="utf-8") as train:
            tokenizer.train_from_iterator((json.loads(line) for line in train), trainer=trainer)
        tokenizer.save(str(staging / "tokenizer.json"))

        eos = tokenizer.token_to_id(SPECIAL[0])
        splits = {}
        for split, raw in files.items():
            target = staging / f"{split}.bin"
            with raw.open(encoding="utf-8") as source_rows, target.open("wb") as tokens:
                for line in source_rows:
                    ids = tokenizer.encode(json.loads(line), add_special_tokens=False).ids
                    np.asarray([*ids, eos], dtype="<u2").tofile(tokens)
            raw.unlink()
            splits[split] = {
                "documents": counts[split], "tokens": target.stat().st_size // 2,
                "sha256": checksum(target),
            }
        manifest = {
            "format": 1, "source": source, "document_limit": doc_limit,
            "exact_duplicates": counts["duplicate"], "split_method": "normalized_sha256_mod_100_lt_5",
            "vocab_size": tokenizer.get_vocab_size(), "eos_id": eos, "splits": splits,
            "tokenizer_sha256": checksum(staging / "tokenizer.json"),
            "documents_sha256": checksum(staging / "documents.jsonl"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return manifest


def fineweb_rows():
    from huggingface_hub import HfFileSystem
    from pyarrow.parquet import ParquetFile

    fs = HfFileSystem()
    shards = sorted(fs.glob(f"datasets/{DATASET}@{REVISION}/sample/10BT/*.parquet"))
    if not shards:
        raise FileNotFoundError("pinned FineWeb-Edu parquet shards were not found")
    for shard in shards:
        with fs.open(shard, "rb", block_size=4 * 2**20) as remote:
            with ParquetFile(remote, pre_buffer=False) as reader:
                for batch in reader.iter_batches(
                    batch_size=128, columns=["text", "id", "url"], use_threads=False
                ):
                    yield from batch.to_pylist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=8192)
    parser.add_argument("--vocab-size", type=int, default=32_768)
    args = parser.parse_args()
    source = {
        "dataset": DATASET, "config": SUBSET, "revision": REVISION,
        "license": "ODC-BY-1.0", "attribution": "Hugging Face FineData / Penedo et al.",
        "url": f"https://huggingface.co/datasets/{DATASET}",
    }
    with closing(fineweb_rows()) as rows:
        manifest = prepare(rows, args.output, doc_limit=args.documents,
                           vocab_size=args.vocab_size, source=source)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
