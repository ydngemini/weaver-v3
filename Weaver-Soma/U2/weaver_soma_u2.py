"""Weaver-SOMA U2: CPU-oriented 100M-parameter extension of U1.
Research only. Neural outputs never authorize motors or bypass signed intents.
"""
import torch
from torch import nn
import torch.nn.functional as F
from unified_model import WeaverSOMAUnified


class CognitiveBlock(nn.Module):
    """Workspace attention and feed-forward module, jointly trained with U1 heads."""
    def __init__(self, dim=1024, ff_dim=2816, heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim))

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class DeepCognitiveCore(nn.Module):
    """Joint workspace with body, memory, expert broadcast, and decision tokens."""
    def __init__(self, latent_dim=64, context_dim=8, dim=1024, depth=10, ff_dim=2816):
        super().__init__()
        self.in_proj = nn.ModuleList([nn.Linear(latent_dim, dim) for _ in range(4)])
        self.context_proj = nn.Linear(context_dim, dim)
        self.pos = nn.Parameter(torch.randn(4, dim) * 0.01)
        self.blocks = nn.ModuleList([CognitiveBlock(dim, ff_dim) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, latent_dim)
        # Exact U1 at step zero; learned gates control experimental additions.
        self.residual_gain = nn.Parameter(torch.zeros(()))
        self.adapter = nn.Sequential(nn.Linear(latent_dim, 16), nn.SiLU(), nn.Linear(16, latent_dim))
        self.adapter_gain = nn.Parameter(torch.zeros(()))
        self.train_mode = "deep"

    def forward(self, common, z, mem, broadcast, context):
        if self.train_mode == "adapter":
            # Skip expensive 100M forward for CPU adapter-only training.
            return common + self.adapter_gain * self.adapter(common)
        x = torch.stack([layer(t) for layer, t in zip(self.in_proj, (z, mem, broadcast, common))], dim=1)
        x = x + self.pos.unsqueeze(0) + self.context_proj(context).unsqueeze(1)
        for block in self.blocks:
            x = block(x)
        refined = self.out_proj(self.final_norm(x[:, -1]))
        return common + self.residual_gain * refined + self.adapter_gain * self.adapter(common)


class WeaverSOMA100M(WeaverSOMAUnified):
    """U1-compatible input/output signatures; 100,204,067 trainable parameters."""
    def __init__(self, width=64, latent=64, core_dim=1024, depth=10, ff_dim=2816):
        super().__init__(width=width, latent=latent)
        self.deep_cognitive_core = DeepCognitiveCore(latent, 8, core_dim, depth, ff_dim)

    def refine_shared_state(self, common, z, internal_memory, broadcast, context, token_z):
        return self.deep_cognitive_core(common, z, internal_memory, broadcast, context)

    def load_u1_weights(self, checkpoint_path):
        """Load exactly matching trained U1 tensors from a TRUSTED checkpoint.

        The newly added deep_cognitive_core remains UNTRAINED.
        Use weights_only=True and verify checkpoint provenance before calling.
        """
        obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or obj.get("architecture") != "WeaverSOMAUnified-v1":
            raise ValueError("Expected verified WeaverSOMAUnified-v1 checkpoint")
        state = obj.get("weights")
        if not isinstance(state, dict):
            raise ValueError("Original U1 checkpoint has no weights mapping")
        target = self.state_dict()
        for key, value in state.items():
            if key not in target or value.shape != target[key].shape:
                raise ValueError(f"U1 tensor incompatible with U2: {key}")
        original_keys = {k for k in target if not k.startswith("deep_cognitive_core.")}
        if set(state) != original_keys:
            raise ValueError("U1 keys do not exactly match the preserved U2 base")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(
            not k.startswith("deep_cognitive_core.") for k in result.missing_keys
        ):
            raise ValueError("Unexpected mismatch restoring U1 into U2")
        return len(state)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import argparse
    import time
    from unified_model import joint_objective

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--ff", type=int, default=2816)
    args = parser.parse_args()

    torch.set_num_threads(4)
    model = WeaverSOMA100M(core_dim=args.width, depth=args.depth, ff_dim=args.ff)
    print("trainable_parameters", count_parameters(model), flush=True)
    if args.smoke_test or args.train_step:
        obs = torch.randn(1, 8, 7)
        quality = torch.ones(1, 8, 3)
        context = torch.randn(1, 8)
        action = F.one_hot(torch.tensor([1]), 3).float()
        start = time.perf_counter()
        out = model(obs, quality, context, action)
        print("forward_sec", round(time.perf_counter() - start, 2), flush=True)
        if args.train_step:
            # Dummy-label test of gradients only, NOT meaningful model training.
            targets = {
                "risk": torch.zeros(1, 3),
                "future": torch.zeros(1, 3, 6, 3),
                "best_action": torch.zeros(1, dtype=torch.long),
                "external": torch.zeros(1),
                "summary": torch.zeros(1, 3),
                "body": torch.zeros(1, 3),
                "attention": torch.zeros(1, dtype=torch.long),
                "route": torch.zeros(1, dtype=torch.long),
                "regulation": torch.zeros(1, 7),
                "reliability": torch.zeros(1),
            }
            loss, _ = joint_objective(out, targets)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-1e-5)
            print("gradient_update_passed", True, "dummy_loss", float(loss), flush=True)
