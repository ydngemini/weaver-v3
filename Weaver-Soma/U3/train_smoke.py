"""Tiny SYNTHETIC olfactory/audio training smoke run; not a real sensory benchmark.

Usage: python U3/train_smoke.py [trusted_U1_checkpoint.pt]
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "U2")]
from model import WeaverSOMAU3


def batch(n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    odor_label = torch.randint(0, 2, (n,), generator=g)
    audio_label = torch.randint(0, 2, (n,), generator=g)
    gas = torch.randn(n, 12, 8, generator=g) * 0.25
    gas[:, :, 0] += (odor_label * 2 - 1).float().unsqueeze(-1) * 0.6
    gas[:, :, 3] -= (odor_label * 2 - 1).float().unsqueeze(-1) * 0.4
    noise = torch.randn(n, 128, generator=g) * 0.08
    t = torch.arange(128).float().unsqueeze(0) / 128
    waveform = (2 * torch.pi * torch.where(audio_label[:, None] == 0, 3., 13.) * t).sin() + noise
    return {"olfaction": {"values": gas, "quality": torch.ones_like(gas)},
            "audio": {"values": waveform}}, odor_label, audio_label


def evaluate(model):
    model.eval()
    sensors, y_odor, y_audio = batch(256, 900001)
    body = (torch.zeros(256, 8, 7), torch.ones(256, 8, 3), torch.zeros(256, 8))
    with torch.no_grad():
        out = model(*body, modalities=sensors)["modality_logits"]
    return {"odor_accuracy": float((out["olfaction"][:, :2].argmax(-1) == y_odor).float().mean()),
            "audio_accuracy": float((out["audio"][:, :2].argmax(-1) == y_audio).float().mean())}


def main():
    if len(sys.argv) > 2:
        raise SystemExit("usage: python U3/train_smoke.py [trusted_U1.pt]")
    torch.set_num_threads(2)
    torch.manual_seed(22)
    model = WeaverSOMAU3(core_dim=64, depth=2, ff_dim=128, fusion_dim=64,
                          quantum_qubits=3)
    if len(sys.argv) == 2:
        assert model.load_verified_u1(sys.argv[1]) == 91
    # Preserve ALL inherited U1/U2 weights, including the random deep core.
    for name, param in model.named_parameters():
        if name in model._u2_keys:
            param.requires_grad_(False)
    before = evaluate(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    for step in range(80):
        sensors, odor_y, audio_y = batch(32, 10000 + step)
        body = (torch.zeros(32, 8, 7), torch.ones(32, 8, 3), torch.zeros(32, 8))
        model.train()
        logits = model(*body, modalities=sensors)["modality_logits"]
        loss = F.cross_entropy(logits["olfaction"][:, :2], odor_y) + F.cross_entropy(
            logits["audio"][:, :2], audio_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = evaluate(model)
    print({"steps": 80, "initial": before, "final": after, "synthetic_only": True,
           "trained_old_U2_weights": False, "checkpoint_saved": False})
    assert after["odor_accuracy"] > 0.80 and after["audio_accuracy"] > 0.80
    print("PASS: internal sensory stems learn simple synthetic tasks; no real sensing skill")


if __name__ == "__main__":
    main()
