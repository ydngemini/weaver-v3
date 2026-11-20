"""Run: python test_evolution.py; research-only structural operator tests."""
import copy
import torch
from torch import nn
from evolution import widen_feedforward, EvolutionProposalHead


def test_growth():
    torch.set_num_threads(2)
    for act in (nn.GELU(), nn.SiLU(), nn.ReLU()):
        for seed in range(5):
            torch.manual_seed(seed)
            base = nn.Sequential(nn.Linear(9, 13), act, nn.Linear(13, 4))
            untouched = copy.deepcopy(base)
            grown, manifest = widen_feedforward(base, [0, 1, 1, 8, 12])
            x = torch.randn(16, 9)
            assert torch.allclose(base(x), grown(x), atol=2e-6, rtol=2e-6)
            assert manifest.old_hidden == 13 and manifest.new_hidden == 18
            assert manifest.parameter_delta == 5 * (9+1+4)
            for a,b in zip(base.parameters(), untouched.parameters()):
                assert torch.equal(a,b), "source network was altered"
    print("PASS: function-preserving neuron growth, 15 configurations")


def test_invalid():
    base = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    for sources in ([], [9], [-1], [True], [0]*5):
        try:
            widen_feedforward(base, sources)
        except (TypeError, ValueError):
            continue
        raise AssertionError(f"invalid growth should fail: {sources}")
    print("PASS: bounded-growth and operator input guards")


def test_proposal():
    torch.manual_seed(3)
    head = EvolutionProposalHead()
    diagnostics = torch.randn(6, 8)
    out = head(diagnostics)
    assert out["action_logits"].shape == (6, 4)
    assert (out["growth_fraction"] > 0).all() and (out["growth_fraction"] < .25).all()
    assert (out["lr_multiplier"] >= .5).all() and (out["lr_multiplier"] <= 2.).all()
    loss = out["action_logits"].square().mean() + out["growth_fraction"].mean() + out["lr_multiplier"].mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    try:
        head(torch.full((1,8),float("nan")))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite diagnostics should fail")
    print("PASS: neural proposal head differentiable; numeric and shape contracts")


def test_unified_model_integration():
    from model import WeaverSOMAU3, LEGACY_OUTPUTS
    torch.manual_seed(14)
    model = WeaverSOMAU3(core_dim=64, depth=1, ff_dim=128,
                         fusion_dim=64, quantum_qubits=2).eval()
    obs = torch.zeros(2, 8, 7)
    quality = torch.ones(2, 8, 3)
    context = torch.zeros(2, 8)
    with torch.no_grad():
        legacy = model(obs, quality, context)
        proposed = model(obs, quality, context, evolution_diagnostics=torch.zeros(2, 8))
    assert set(legacy) == LEGACY_OUTPUTS
    assert set(proposed) == LEGACY_OUTPUTS | {"evolution_proposals"}
    assert proposed["evolution_proposals"]["action_logits"].shape == (2, 4)
    for key in LEGACY_OUTPUTS:
        assert torch.equal(legacy[key], proposed[key]), ("inherited output changed", key)
    print("PASS: proposal head lives inside one U3 model and preserves 15 legacy outputs")


if __name__ == "__main__":
    test_growth()
    test_invalid()
    test_proposal()
    test_unified_model_integration()
