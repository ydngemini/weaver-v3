"""Train a native paired-action head on official UTD-MHAD MAT archives."""
from __future__ import annotations

import argparse
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import re
from zipfile import ZipFile

import torch
from torch.nn import functional as F
from scipy.io import loadmat

from model import LEGACY_OUTPUTS, WeaverSOMAU3
from paired_actions import ACTIONS, PairedActionHead, skeleton_phases, wearable_phases

TRIAL = re.compile(r"a(\d+)_s(\d+)_t(\d+)_(skeleton|inertial)\.mat")
SEED = 2251


def read(path, modality, field):
    values = {}
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            match = TRIAL.fullmatch(Path(member).name)
            if match is None or match[4] != modality:
                continue
            key = tuple(map(int, match.group(1, 2, 3)))
            if key in values:
                raise ValueError(f"duplicate trial {key}")
            raw = loadmat(BytesIO(archive.read(member)), variable_names=[field])[field]
            values[key] = torch.as_tensor(raw, dtype=torch.float32)
    return values


def process(items, keys, fn):
    rows = [fn(items[key].unsqueeze(0)) for key in keys]
    return tuple(torch.cat(parts) for parts in zip(*rows))


def select(item, idx):
    return tuple(a[idx] for a in item)


def normalize(model, pose, imu, idx):
    with torch.no_grad():
        for view, center, scale in (
                (pose, model.skeleton_center, model.skeleton_scale),
                (imu, model.wearable_center, model.wearable_scale)):
            data, valid = select(view, idx)
            mask = valid > 0
            total = mask.sum(0).clamp_min(1)
            center.copy_((data * mask).sum(0) / total)
            scale.copy_((((data - center).square() * mask).sum(0) / total).sqrt().clamp_min(.001))


def fit(pose, imu, labels, train, val):
    model = PairedActionHead()
    normalize(model, pose, imu, train)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.03)
    best = (0, float("-inf"))
    generator = torch.Generator().manual_seed(SEED)
    for epoch in range(160):
        idx = train[torch.randperm(len(train), generator=generator)]
        for group in idx.split(64):
            left, right = select(pose, group), select(imu, group)
            out = model.score_phases(left, right)
            truth = labels[group]
            # Train on both streams and each stream alone.
            loss = F.cross_entropy(out["logits"], truth)
            loss += .4 * (
                F.cross_entropy(2 * out["skeleton_contribution"], truth)
                + F.cross_entropy(2 * out["wearable_contribution"], truth)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            out = model.score_phases(select(pose, val), select(imu, val))
            score = (int((out["logits"].argmax(1) == labels[val]).sum()),
                     -float(F.cross_entropy(out["logits"], labels[val])))
            if score > best:
                best = score
                weights = deepcopy(model.state_dict())
                selected = epoch + 1
    model.load_state_dict(weights)
    return model.eval(), selected


def partial(item):
    x, mask = item
    mask = mask.clone()
    # Remove three middle bins while leaving the total trial duration supplied.
    mask[:, 2:5] = 0
    return torch.where(mask > 0, x, 0), mask


def report_case(model, pose, imu, target, *, coverage_weighting=True):
    out = model.score_phases(pose, imu, coverage_weighting=coverage_weighting)
    predicted = out["logits"].argmax(-1)
    scored = out["score_available"]
    return {"trials": len(target), "scoreable": int(scored.sum()),
            "correct": int((scored & (predicted == target)).sum()),
            "stream_disagreement": int(out["streams_disagree"].sum())}


@torch.inference_mode()
def verify(model, skeleton, wearable):
    body = WeaverSOMAU3(core_dim=64, ff_dim=128, depth=2, fusion_dim=32, enable_paired_actions=True).eval()
    body.paired_action_head.load_state_dict(model.state_dict(), strict=True)
    args = torch.zeros(1, 8, 7), torch.ones(1, 8, 3), torch.zeros(1, 8)
    baseline = body(*args)
    props = {"skeleton": skeleton.unsqueeze(0), "wearable": wearable.unsqueeze(0)}
    result = body(*args, paired_action_inputs=props)
    out = result["paired_actions"]
    torch.testing.assert_close(out["logits"], out["skeleton_contribution"] + out["wearable_contribution"])
    torch.testing.assert_close(out["logits"], model(**props)["logits"])
    dropped = model(skeleton=props["skeleton"], wearable=torch.full_like(props["wearable"], float("nan")))
    torch.testing.assert_close(dropped["logits"], model(skeleton=props["skeleton"])["logits"], rtol=0, atol=0)
    assert not model(skeleton=torch.full_like(props["skeleton"], float("nan")))["score_available"].any()
    assert all(torch.equal(baseline[k], result[k]) for k in LEGACY_OUTPUTS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skeleton", type=Path, required=True)
    p.add_argument("--inertial", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    stereo = read(args.skeleton, "skeleton", "d_skel")
    skeleton = {k: x.permute(2, 0, 1) for k, x in stereo.items()}
    wearable = read(args.inertial, "inertial", "d_iner")
    keys = sorted(skeleton.keys() & wearable.keys())
    if not keys:
        raise ValueError("no paired trials found")
    targets = torch.tensor([k[0] - 1 for k in keys])
    people = torch.tensor([k[1] for k in keys])
    pose = process(skeleton, keys, skeleton_phases)
    imu = process(wearable, keys, wearable_phases)
    training = torch.isin(people, torch.tensor([1, 3, 5])).nonzero().flatten()
    validation = (people == 7).nonzero().flatten()
    test = (people % 2 == 0).nonzero().flatten()
    model, epoch = fit(pose, imu, targets, training, validation)
    p_test, i_test, truth = select(pose, test), select(imu, test), targets[test]
    wrong = torch.arange(len(truth))
    while True:
        bad = truth[wrong] == truth
        if not bad.any():
            break
        wrong[bad] = (wrong[bad] + 1) % len(truth)
    cases = {"paired": (p_test, i_test), "skeleton_only": (p_test, None), "wearable_only": (None, i_test),
             "skeleton_partial": (partial(p_test), i_test), "wearable_partial": (p_test, partial(i_test)),
             "wrong_action_wearable": (p_test, select(i_test, wrong))}
    with torch.inference_mode():
        results = {name: report_case(model, *inputs, truth) for name, inputs in cases.items()}
        results["equal_weight_ablation"] = {
            name: report_case(model, *cases[name], truth, coverage_weighting=False)
            for name in ("paired", "skeleton_partial", "wearable_partial")}
        results["paired_by_subject"] = {
            int(s): report_case(model, select(p_test, people[test] == s),
                                select(i_test, people[test] == s), truth[people[test] == s])
            for s in people[test].unique()}
    example = keys[int(test[0])]
    verify(model, skeleton[example], wearable[example])
    args.run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"seed": SEED, "classes": ACTIONS, "train_subjects": [1, 3, 5], "validation_subjects": [7],
               "test_subjects": [2, 4, 6, 8], "training_trials": len(training), "validation_trials": len(validation),
               "chosen_epoch": epoch, "paired_records": len(keys), "native_integration_verified": True,
               "source_sha256": {path.name: WeaverSOMAU3._file_hash(path) for path in (args.skeleton, args.inertial)},
               "results": results}
    (args.run_dir / "paired_action_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"format_version": 1, "classes": ACTIONS, "weights": model.state_dict()},
               args.run_dir / "paired_action_head.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
