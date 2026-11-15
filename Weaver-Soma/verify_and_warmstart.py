"""Verify the local recovered U1 checkpoint and warm-start the matching U2 layers.

This script DOES NOT train U2, authorize any robot action, or deploy software.
Restore a trusted U1 checkpoint to U1/results/weaver_soma_unified.pt first.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
U2_PATH = ROOT / "U2"
CHECKPOINT = ROOT / "U1" / "results" / "weaver_soma_unified.pt"
EXPECTED_SHA256 = "3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f"

def main() -> None:
    if not CHECKPOINT.is_file():
        raise SystemExit(
            "Missing U1/results/weaver_soma_unified.pt. Restore the original "
            "checkpoint from the separate trusted recovery archive. "
            "Never download random weights from an untrusted source."
        )

    actual_sha = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise SystemExit(f"Checkpoint SHA-256 mismatch: {actual_sha}")

    # U2 includes the preserved U1 model under the same import name.
    sys.path.insert(0, str(U2_PATH))
    from weaver_soma_u2 import WeaverSOMA100M, count_parameters

    torch.set_num_threads(2)
    original = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if original.get("architecture") != "WeaverSOMAUnified-v1":
        raise ValueError("Wrong checkpoint architecture")
    state = original.get("weights")
    if not isinstance(state, dict):
        raise ValueError("Missing original tensor mapping")

    torch.manual_seed(20261115)
    model = WeaverSOMA100M().eval()
    matched = model.load_u1_weights(str(CHECKPOINT))
    assert matched == len(state) == 91
    assert count_parameters(model) == 100204067
    for name, weights in state.items():
        assert torch.equal(model.state_dict()[name], weights), name

    obs = torch.randn(1, 8, 7)
    quality = torch.ones(1, 8, 3)
    context = torch.zeros(1, 8)
    action = torch.tensor([[0., 1., 0.]])
    with torch.inference_mode():
        result = model(obs, quality, context, action)
    assert result["risk_logits"].shape == (1, 3)
    assert result["future"].shape == (1, 3, 6, 3)

    print(f"Recovered U1 epoch: {original.get('best_epoch')}")
    print(f"Verified U1 tensor count: {matched}")
    print(f"U2 total trainable parameters: {count_parameters(model):,}")
    print("U2 forward pass and exact U1 warm start: PASS")
    print("WARNING: the new U2 deep transformer layers remain UNTRAINED.")

if __name__ == "__main__":
    main()
