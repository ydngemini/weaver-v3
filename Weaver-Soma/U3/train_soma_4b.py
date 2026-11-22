"""Train and resume the SOMA pretraining pilot on CPU or one CUDA GPU."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from hashlib import file_digest
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch.nn import functional as F

from soma_4b import PretrainConfig, SomaPretrainLM


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


class TokenCorpus:
    def __init__(self, path):
        manifest_file = path / "manifest.json"
        self.manifest = json.loads(manifest_file.read_text())
        if self.manifest["format"] != 1:
            raise ValueError("unsupported corpus format")
        with (path / "tokenizer.json").open("rb") as handle:
            if file_digest(handle, "sha256").hexdigest() != self.manifest["tokenizer_sha256"]:
                raise ValueError("corrupt tokenizer")
        self.streams = {}
        for split in ("train", "validation"):
            file = path / f"{split}.bin"
            spec = self.manifest["splits"][split]
            with file.open("rb") as handle:
                digest = file_digest(handle, "sha256").hexdigest()
            if digest != spec["sha256"] or file.stat().st_size != 2 * spec["tokens"]:
                raise ValueError(f"corrupt {split} token stream")
            self.streams[split] = np.memmap(file, dtype="<u2", mode="r")

    def batches(self, split, phase, count, context, microbatch, device):
        stream = self.streams[split]
        for batch in range(phase, phase + count):
            left = batch * context * microbatch
            right = left + context * microbatch
            if right >= len(stream):
                raise ValueError(f"insufficient {split} tokens for batch {batch}")
            tokens = torch.from_numpy(stream[left:right + 1].astype(np.int64)).to(device)
            yield tokens[:-1].reshape(microbatch, context), tokens[1:].reshape(microbatch, context)


def forward_loss(model, inputs, targets, device):
    precision = torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False) \
        if device.type == "cuda" else nullcontext()
    with precision:
        logits = model(inputs)
        return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())


@torch.inference_mode()
def validate(model, corpus, args, device):
    model.eval()
    losses = [float(forward_loss(model, x, y, device))
              for x, y in corpus.batches("validation", 0, args.validation_batches,
                                        args.context, args.microbatch, device)]
    model.train()
    value = sum(losses) / len(losses)
    if not math.isfinite(value):
        raise FloatingPointError("nonfinite validation loss")
    return value


def save_checkpoint(path, state):
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def train(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("require a CUDA device with BF16 support")
    corpus = TokenCorpus(args.data)
    config = (PretrainConfig.smoke(corpus.manifest["vocab_size"])
              if args.profile == "smoke" else PretrainConfig())
    if corpus.manifest["vocab_size"] != config.vocab_size:
        raise ValueError(f"expected a {config.vocab_size}-token vocabulary")
    if args.context > config.context:
        raise ValueError(f"maximum context is {config.context}")
    if args.validation_batches * args.context * args.microbatch >= len(corpus.streams["validation"]):
        raise ValueError("validation budget exceeds available validation tokens")

    deadline = time.monotonic() + args.max_seconds
    path = args.output / "latest.pt"
    signature = {
        "model": asdict(config), "manifest": corpus.manifest, "context": args.context,
        "microbatch": args.microbatch, "accumulate": args.accumulate, "device_type": device.type,
        "lr": args.lr, "warmup": args.warmup, "seed": args.seed,
        "validation_batches": args.validation_batches,
    }
    if path.exists() != args.resume:
        raise ValueError("use --resume with an existing checkpoint, or choose a fresh output path")
    state = torch.load(path, map_location="cpu", weights_only=True) if args.resume else None
    if state is not None and state["signature"] != signature:
        raise ValueError("checkpoint and model, corpus or training settings differ")
    step = 0 if state is None else state["step"]
    if step > args.steps:
        raise ValueError("step limit is lower than the checkpoint step")
    if args.steps * args.accumulate * args.context * args.microbatch >= len(corpus.streams["train"]):
        raise ValueError("requested steps exceed unique available training tokens")

    torch.manual_seed(args.seed)
    with torch.device(device):
        model = SomaPretrainLM(config, activation_checkpointing=device.type == "cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.1, fused=device.type == "cuda")
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["gpu_rng"], device)
        del state

    parameters = sum(parameter.numel() for parameter in model.parameters())
    tokens_per_step = args.context * args.microbatch * args.accumulate
    baseline = validate(model, corpus, args, device)
    emit("baseline", step=step, validation_loss=baseline, parameters=parameters,
         device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    model.train()
    stopped = False

    def request_stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def persist(current_step):
        optimizer.zero_grad(set_to_none=True)
        loss = validate(model, corpus, args, device)
        save_checkpoint(path, {
            "signature": signature, "step": current_step, "validation_loss": loss,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "gpu_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        })
        emit("checkpoint", step=current_step, validation_loss=loss, path=str(path))

    saved_step = step
    while step < args.steps and time.monotonic() < deadline and not stopped:
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_total = torch.zeros((), device=device)
        for x, y in corpus.batches("train", step * args.accumulate, args.accumulate,
                                   args.context, args.microbatch, device):
            loss = forward_loss(model, x, y, device)
            (loss / args.accumulate).backward()
            loss_total += loss.detach() / args.accumulate
        mean_loss = float(loss_total)
        if not math.isfinite(mean_loss):
            raise FloatingPointError("nonfinite training loss")
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                             error_if_nonfinite=True))
        rate = args.lr * min(1.0, (step + 1) / max(1, args.warmup))
        optimizer.param_groups[0]["lr"] = rate
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step += 1
        emit("step", step=step, tokens_seen=step * tokens_per_step, loss=mean_loss,
             gradient_norm=gradient_norm, learning_rate=rate,
             tokens_per_second=tokens_per_step / (time.monotonic() - started),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
             if device.type == "cuda" else None,
             peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
             if device.type == "cuda" else None)

        if step == args.steps or step % args.checkpoint_every == 0:
            persist(step)
            saved_step = step

    if saved_step != step:
        persist(step)
    reason = "step_limit" if step == args.steps else "interrupt" if stopped else "time_limit"
    emit("finished", step=step, tokens_seen=step * tokens_per_step, reason=reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("4b", "smoke"), default="4b")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--steps", type=int, required=True, help="absolute optimizer step limit")
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.steps, args.context, args.microbatch, args.accumulate,
           args.validation_batches, args.checkpoint_every) <= 0:
        parser.error("step, context, microbatch, accumulation, validation and checkpoint sizes must be positive")
    if args.warmup < 0 or not (math.isfinite(args.lr) and args.lr > 0
                               and math.isfinite(args.max_seconds) and args.max_seconds > 0):
        parser.error("require a nonnegative warmup and finite positive learning rate and time limit")
    if args.profile == "4b" and args.device != "cuda":
        parser.error("the full profile requires CUDA, use --profile smoke for CPU")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args)


if __name__ == "__main__":
    main()
