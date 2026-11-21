"""Test action-conditioned self-prediction in a synthetic two-axis body."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from train_esc50 import sha256

HISTORY = 5
SEED = 72


@dataclass
class Body:
    pose: torch.Tensor
    velocity: torch.Tensor
    gain: torch.Tensor
    damping: torch.Tensor
    history: torch.Tensor


def next_velocity(pose, velocity, gain, damping, command):
    cross_axis = torch.stack((command[:, 1], -command[:, 0]), dim=1)
    return (damping * velocity + gain * command + .09 * cross_axis
            + .10 * command.pow(3) - .08 * pose.sin())


def simulated_bodies(seed, size, gain_range):
    gen = torch.Generator().manual_seed(seed)
    pose = .5 * torch.randn(size, 2, generator=gen)
    velocity = .3 * torch.randn(size, 2, generator=gen)
    low, high = gain_range
    gain = low + (high - low) * torch.rand(size, 2, generator=gen)
    damping = .62 + .24 * torch.rand(size, 2, generator=gen)
    history = []
    for _ in range(HISTORY):
        command = torch.empty(size, 2).uniform_(-1, 1, generator=gen)
        changed = next_velocity(pose, velocity, gain, damping, command)
        measured = changed - velocity + .025 * torch.randn(size, 2, generator=gen)
        history.append(torch.cat((command, measured), dim=1))
        pose = pose + .18 * changed
        velocity = changed
    return Body(pose, velocity, gain, damping, torch.cat(history, dim=1))


def predictors(body, command):
    state = torch.cat((body.pose, body.velocity), dim=1)
    action = torch.cat((state, command), dim=1)
    return {"state_only": state, "state_action": action,
            "state_action_history": torch.cat((action, body.history), dim=1)}


def target(body, command):
    return next_velocity(body.pose, body.velocity, body.gain, body.damping, command)


def make_model(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32),
                         nn.SiLU(), nn.Linear(32, 2))


def train(x, y):
    torch.manual_seed(SEED)
    model = make_model(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.001)
    for _ in range(35):
        for indices in torch.randperm(len(x)).split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    return model.eval()


def rmse(estimate, reference):
    return float((estimate - reference).square().mean().sqrt())


@torch.inference_mode()
def evaluate(models, body, command):
    plus = predictors(body, command)
    minus = predictors(body, -command)
    expected = target(body, command)
    expected_flipped = target(body, -command)
    effect = expected - expected_flipped
    output = {}
    for name, model in models.items():
        key = "state_action" if name == "linear_state_action" else name
        prediction = model(plus[key])
        difference = prediction - model(minus[key])
        output[name] = {
            "next_velocity_rmse": rmse(prediction, expected),
            "command_effect_rmse": rmse(difference, effect),
            "command_effect_direction_agreement": float(
                (F.cosine_similarity(difference, effect) > 0).float().mean()
            ),
        }
    return output


def command_batch(seed, size):
    gen = torch.Generator().manual_seed(seed)
    return torch.empty(size, 2).uniform_(-1, 1, generator=gen)


@torch.inference_mode()
def evaluate_residual(model, calibration, standard, shifted):
    def residuals(body, seed):
        command = command_batch(seed, len(body.pose))
        inputs = predictors(body, command)["state_action_history"]
        estimate = model(inputs)
        gen = torch.Generator().manual_seed(seed)
        observed = target(body, command) + .025 * torch.randn(estimate.shape, generator=gen)
        direction = F.normalize(torch.randn(estimate.shape, generator=gen), dim=1)
        nominal = (observed - estimate).norm(dim=1)
        perturbed = (observed + .55 * direction - estimate).norm(dim=1)
        return nominal, perturbed

    calibration_residuals, _ = residuals(calibration, 44)
    threshold = torch.quantile(calibration_residuals, .95)
    nominal, perturbed = residuals(standard, 45)
    shifted_nominal, shifted_perturbed = residuals(shifted, 46)
    return {
        "threshold": float(threshold), "in_range_false_alert_rate": float((nominal > threshold).float().mean()),
        "in_range_impulse_detection_rate": float((perturbed > threshold).float().mean()),
        "changed_gain_false_alert_rate": float((shifted_nominal > threshold).float().mean()),
        "changed_gain_impulse_detection_rate": float((shifted_perturbed > threshold).float().mean()),
        "simulated_impulse_magnitude": .55,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    standard_gain = (.62, 1.28)
    changed_gain = (.30, .58)
    training = simulated_bodies(1, 6000, standard_gain)
    calibration = simulated_bodies(2, 1200, standard_gain)
    standard = simulated_bodies(3, 1600, standard_gain)
    shifted = simulated_bodies(4, 1600, changed_gain)
    command = command_batch(43, len(training.pose))
    noisy_target = target(training, command) + .025 * torch.randn(
        command.shape, generator=torch.Generator().manual_seed(80),
    )
    designs = predictors(training, command)
    models = {name: train(x, noisy_target) for name, x in designs.items()}
    action_x = designs["state_action"]
    augmented = torch.cat((action_x, torch.ones_like(action_x[:, :1])), dim=1)
    coefficients = torch.linalg.lstsq(augmented, noisy_target).solution
    linear = nn.Linear(action_x.shape[1], 2)
    with torch.no_grad():
        linear.weight.copy_(coefficients[:-1].T)
        linear.bias.copy_(coefficients[-1])
    # The linear baseline sees the same state and command as the action-only MLP.
    models["linear_state_action"] = linear
    history_model = models["state_action_history"]
    checkpoint = args.run_dir / "soma_u3_body_self_model_sim.pt"
    torch.save({"architecture": "SOMA-U3-simulated-body-forward-model-v1",
                "history": HISTORY, "weights": history_model.state_dict()}, checkpoint)
    report = {
        "synthetic_only": True, "training_episodes": len(training.pose),
        "calibration_episodes": len(calibration.pose),
        "standard_test_episodes": len(standard.pose), "changed_gain_episodes": len(shifted.pose),
        "standard_gain_range": standard_gain, "changed_gain_range": changed_gain,
        "in_range": evaluate(models, standard, command_batch(47, len(standard.pose))),
        "changed_gain": evaluate(models, shifted, command_batch(48, len(shifted.pose))),
        "residual_alerts": evaluate_residual(history_model, calibration, standard, shifted),
        "checkpoint_sha256": sha256(checkpoint),
    }
    (args.run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
