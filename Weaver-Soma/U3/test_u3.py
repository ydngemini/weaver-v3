"""CPU wiring tests, not evidence of learned multimodal recognition.

python U3/test_u3.py
python U3/test_u3.py /path/to/weaver_soma_unified_RECOVERED.pt
"""
import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "U2"))
from model import WeaverSOMAU3, LEGACY_OUTPUTS, U1_SHA256
from weaver_soma_u2 import WeaverSOMA100M


def samples(batch=2):
    torch.manual_seed(19)
    return {
        "text": {"values": torch.randint(0, 255, (batch, 12)),
                 "valid": torch.ones(batch, 12, dtype=torch.bool)},
        "vision": {"values": torch.rand(batch, 2, 3, 8, 8),
                   "valid": torch.ones(batch, 2, dtype=torch.bool)},
        "audio": {"values": torch.randn(batch, 128),
                  "valid": torch.ones(batch, 128, dtype=torch.bool)},
        "touch": {"values": torch.randn(batch, 8, 6),
                  "quality": torch.ones(batch, 8, 6)},
        "olfaction": {"values": torch.randn(batch, 10, 8),
                      "quality": torch.ones(batch, 10, 8),
                      "timestamps": torch.arange(10).expand(batch, -1).float()},
        "rf": {"values": torch.randn(batch, 7, 8)},
        "spatial": {"values": torch.randn(batch, 9, 6)},
    }


def test_exact_legacy(path=None):
    torch.manual_seed(101)
    old = WeaverSOMA100M(core_dim=64, depth=2, ff_dim=128).eval()
    new = WeaverSOMAU3(core_dim=64, depth=2, ff_dim=128,
                        fusion_dim=64, quantum_qubits=3).eval()
    if path:
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == U1_SHA256
        assert old.load_u1_weights(path) == 91
        assert new.load_verified_u1(path) == 91
        status = "verified U1 epoch-24 weights"
    else:
        # Structural test only. Random base tensors must not be called trained.
        r = new.load_state_dict(old.state_dict(), strict=False)
        assert not r.unexpected_keys
        status = "random U2 base"
    new.deep_cognitive_core.load_state_dict(old.deep_cognitive_core.state_dict(), strict=True)
    obs = torch.randn(2, 8, 7)
    quality = torch.ones(2, 8, 3)
    context = torch.randn(2, 8)
    action = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    with torch.no_grad():
        original = old(obs, quality, context, action)
        plain = new(obs, quality, context, action)
        enriched = new(obs, quality, context, action, modalities=samples())
    assert set(original) == LEGACY_OUTPUTS and set(plain) == LEGACY_OUTPUTS
    for key in LEGACY_OUTPUTS:
        assert torch.equal(original[key], plain[key]), ("base regression", key)
        assert torch.equal(original[key], enriched[key]), ("zero-gated regression", key)
    assert enriched["quantum_expectations"].shape == (2, 3)
    assert (enriched["quantum_expectations"].abs() <= 1.00001).all()
    assert len([k for k in enriched["modality_logits"] if k != "text_token_logits"]) == 7
    print("PASS: all 15 legacy outputs exactly equal, with seven sensory stems;", status)
    return new, (obs, quality, context, action)


def test_trainability(model, inputs):
    model.train()
    obs, quality, context, action = inputs
    data = samples()
    pred = model(obs, quality, context, action, modalities=data)
    loss = (pred["modality_logits"]["olfaction"].square().mean()
            + pred["modality_logits"]["audio"].square().mean()
            + pred["modality_logits"]["vision"].square().mean()
            + pred["quantum_expectations"].square().mean()
            + pred["risk_logits"].square().mean())
    loss.backward()
    assert model.temporal_stems["olfaction"].project[0].weight.grad is not None
    assert model.audio_conv[0].weight.grad is not None
    assert model.vision_patch.weight.grad is not None
    assert model.quantum.theta_y.grad is not None
    assert model.quantum.encode.weight.grad is not None
    assert model.multimodal_gain.grad is not None

    model.eval()
    with torch.no_grad():
        model.multimodal_gain.fill_(0.2)
        first = model(obs, quality, context, action, modalities=data)
        data["olfaction"]["values"] *= -10
        second = model(obs, quality, context, action, modalities=data)
    assert not torch.equal(first["state"], second["state"])
    print("PASS: sensory gradients and quantum simulator gradients; inputs affect fused state")


def test_invalid_input(model, inputs):
    model.eval()
    obs, quality, context, action = inputs
    with torch.no_grad():
        model.multimodal_gain.fill_(0.2)
        samples_ = torch.randn(2, 6, 8)
        channel_quality = torch.ones_like(samples_)
        channel_quality[:, :, 2] = 0
        data = {"olfaction": {"values": samples_, "quality": channel_quality}}
        original = model(obs, quality, context, action, modalities=data)
        corrupted = samples_.clone()
        corrupted[:, :, 2] = 1e8
        data["olfaction"]["values"] = corrupted
        changed = model(obs, quality, context, action, modalities=data)
        assert torch.equal(original["state"], changed["state"])
        all_absent = {"olfaction": {"values": samples_,
                      "valid": torch.zeros(2, 6, dtype=torch.bool)}}
        muted = model(obs, quality, context, action, modalities=all_absent)
        base = model(obs, quality, context, action)
        for key in LEGACY_OUTPUTS:
            assert torch.equal(muted[key], base[key]), ("missing sensor regression", key)
    print("PASS: masked chemical channel and missing sensory stream preserve original model")


def test_spectral_audio():
    config = dict(core_dim=64, depth=1, ff_dim=128, fusion_dim=64, quantum_qubits=2)
    torch.manual_seed(23)
    plain = WeaverSOMAU3(**config).eval()
    torch.manual_seed(23)
    spectral = WeaverSOMAU3(audio_encoder="logmel", **config).eval()
    waveform = torch.randn(2, 4096)
    valid = torch.ones_like(waveform, dtype=torch.bool)
    valid[0, 1800:] = False
    valid[1] = False
    audio = {"values": waveform, "valid": valid}
    corrupted = {"values": waveform.masked_fill(~valid, float("nan")), "valid": valid}
    obs, quality, context = torch.zeros(2, 8, 7), torch.ones(2, 8, 3), torch.zeros(2, 8)
    with torch.no_grad():
        unrelated = {"text": {"values": torch.zeros(2, 5, dtype=torch.long)}}
        original = plain(obs, quality, context, modalities=unrelated)
        output = spectral(obs, quality, context, modalities=unrelated)
        assert all(torch.equal(original[key], output[key]) for key in LEGACY_OUTPUTS)
        assert torch.equal(original["modality_logits"]["text"], output["modality_logits"]["text"])

        output = spectral(obs, quality, context, modalities={"audio": audio})
        changed = spectral(obs, quality, context, modalities={"audio": corrupted})
        assert torch.equal(output["modality_logits"]["audio"], changed["modality_logits"]["audio"])
        assert output["sensory_token_mask"][0].any() and not output["sensory_token_mask"][1].any()
        stem = spectral.spectral_audio
        direct = stem.classify(*stem(waveform, valid))
        assert torch.equal(output["modality_logits"]["audio"], direct)
        assert "audio" not in spectral(obs, quality, context, modalities={
            "audio": {"values": waveform, "quality": torch.zeros_like(waveform)}
        })["modality_logits"]
        assert spectral(obs, quality, context, modalities={
            "audio": {"values": waveform[:, :1]}
        })["modality_logits"]["audio"].shape == (2, 16)
    print("PASS: spectral audio integration, masked values, short inputs, and other modality parity")


def test_audio_monitor(tmp_dir):
    from audio_self_monitor import AUDIO_CLASSES, AudioDecisionReason, AudioSelfMonitor

    cfg = dict(core_dim=64, depth=1, ff_dim=128, fusion_dim=64, quantum_qubits=2)
    model = WeaverSOMAU3(audio_encoder="logmel", **cfg).eval()
    waveform = torch.stack((torch.randn(1600), torch.full((1600,), float("nan"))))
    obs, quality, context = torch.zeros(2, 8, 7), torch.ones(2, 8, 3), torch.zeros(2, 8)
    audio = {"audio": {"values": waveform}}
    with torch.inference_mode():
        before = model(obs, quality, context, modalities=audio)
    monitor = AudioSelfMonitor()
    checkpoint = tmp_dir / "monitor.pt"
    payload = {"architecture": "SOMA-U3-audio-self-monitor-v1",
               "classes": AUDIO_CLASSES, "sample_rate": 16000, "fusion_dim": 64,
               "audio_weights": model.spectral_audio.state_dict(),
               "monitor_weights": monitor.state_dict()}
    torch.save(payload, checkpoint)
    model.load_audio_self_monitor(checkpoint)
    with torch.inference_mode():
        after = model(obs, quality, context, modalities=audio)
        assert all(torch.equal(before[k], after[k]) for k in LEGACY_OUTPUTS)
        assert torch.equal(before["modality_logits"]["audio"], after["modality_logits"]["audio"])
        readout = after["audio_self_monitor"]
        assert readout["score_available"].tolist() == [True, False]
        assert readout["recommend_abstain"][1] and readout["p_correct"][1] == 0
        assert readout["decision_reason"][1] == AudioDecisionReason.MISSING_SAMPLES
        audit = readout["audit"]
        contribution = audit["log_odds_contributions"]
        assert contribution.shape[1] == len(audit["feature_names"])
        torch.testing.assert_close(audit["intercept"] + contribution.sum(1), audit["log_odds"])
        assert audit["loaded_checkpoint_sha256"] == model._file_hash(checkpoint)
        assert not contribution[1].any()
        assert "audio_self_monitor" not in model(obs, quality, context)
        empty = model(obs, quality, context, modalities={"audio": {
            "values": torch.zeros_like(waveform), "valid": torch.zeros_like(waveform, dtype=torch.bool)
        }})
        assert not empty["audio_self_monitor"]["score_available"].any()
        assert empty["audio_self_monitor"]["recommend_abstain"].all()
        assert "audio" not in empty["modality_logits"]
        silent = model(obs, quality, context, modalities={"audio": {
            "values": torch.zeros_like(waveform)
        }})["audio_self_monitor"]
        assert silent["samples_observed"].all() and not silent["score_available"].any()
        assert (silent["decision_reason"] == AudioDecisionReason.ZERO_SIGNAL).all()
    active_monitor = model.audio_self_monitor
    payload["monitor_weights"]["threshold"] = torch.tensor(float("nan"))
    torch.save(payload, checkpoint)
    try:
        model.load_audio_self_monitor(checkpoint)
    except ValueError:
        assert model.audio_self_monitor is active_monitor
    else:
        raise AssertionError("nonfinite monitor checkpoint should be rejected")
    print("PASS: monitor reload, unchanged logits, missing evidence, and invalid checkpoint rejection")


def main():
    if len(sys.argv) > 2:
        raise SystemExit("usage: python U3/test_u3.py [trusted_U1_checkpoint.pt]")
    torch.set_num_threads(2)
    model, inputs = test_exact_legacy(sys.argv[1] if len(sys.argv) == 2 else None)
    test_trainability(model, inputs)
    test_invalid_input(model, inputs)
    test_spectral_audio()
    with TemporaryDirectory() as work:
        test_audio_monitor(Path(work))
    print("SOMA U3 engineering smoke tests: PASS; NO recognition or training claim")


if __name__ == "__main__":
    main()
