# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.4,<3", "numpy>=1.26,<3", "scikit-learn>=1.3,<2", "huggingface_hub>=1.0,<3"]
# ///
"""SOMA U3 experimental Hugging Face Jobs runner.

This is not a trained AGI model. Uses an exact pinned GitHub commit as source,
not an arbitrary agent/third-party foundation model.
Requires permission/credits for Hugging Face Jobs. Do not submit paid Jobs
without explicit user approval.

Supported tasks:
  verify: run existing U3 regression/growth tests (random base; no trained U2 claim)
  audio5: train a small-core native U3 5-class ESC-50 audio stem using real licensed
          recordings, with independent official fold 5 held out until final.
Audio training is research-only; not evidence of full U3 multimodal skill.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from huggingface_hub import HfApi

# Exact tested design branch snapshot; deliberate updates require a code change.
REPO_OWNER = "ydngemini"
REPO_NAME = "weaver-v3"
SOURCE_COMMIT = "332d2cb33af4327403161f4cc54aa203ef18408f"
ARCHIVE = f"https://codeload.github.com/{REPO_OWNER}/{REPO_NAME}/tar.gz/{SOURCE_COMMIT}"

def fetch_source(folder: Path) -> Path:
    archive_path = folder / "pinned-source.tar.gz"
    with urllib.request.urlopen(ARCHIVE, timeout=180) as stream, archive_path.open("wb") as dest:
        while block := stream.read(1024 * 1024):
            dest.write(block)
    with tarfile.open(archive_path, "r:gz") as tar:
        # Python 3.12+ filtered extraction; never blindly trust an archive.
        tar.extractall(folder, filter="data")
    archive_path.unlink(missing_ok=True)
    matches = list(folder.glob(f"{REPO_NAME}-*/Weaver-Soma/U3/test_evolution.py"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one SOMA U3 checkout, got {len(matches)}")
    return matches[0].parents[1]

def run(args: list[str], cwd: Path):
    print("RUN:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)

def upload_results(folder: Path, *, repo_id: str, task: str):
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is needed for private artifact upload; none supplied")
    api = HfApi(token=os.environ["HF_TOKEN"])
    info = api.create_repo(repo_id=repo_id, private=True, repo_type="model", exist_ok=True)
    # Confirm existing repository is not public; do not upload private checkpoints publicly.
    actual = api.repo_info(repo_id=info.repo_id, repo_type="model")
    if not actual.private:
        raise RuntimeError("Refusing artifact upload: repository must be private")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=info.repo_id,
        repo_type="model",
        path_in_repo=f"experiments/{task}/{SOURCE_COMMIT[:12]}",
        commit_message=f"SOMA U3 {task} experimental artifacts: pinned source {SOURCE_COMMIT[:12]}",
    )
    print(f"PRIVATE MODEL ARTIFACTS UPLOADED: https://huggingface.co/{info.repo_id}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("verify", "audio5"), default="verify")
    parser.add_argument("--repo-id", default=None,
                        help="Private HF model repo like username/SOMA-U3-research")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples", type=int, default=12000)
    args = parser.parse_args()
    if args.task == "audio5" and args.epochs < 1:
        parser.error("epochs must be positive")
    if args.repo_id and args.task != "audio5":
        parser.error("repo-id only supported for audio5 artifacts")
    with tempfile.TemporaryDirectory(prefix="soma-u3-hf-") as tmp:
        workspace = Path(tmp)
        soma_root = fetch_source(workspace)
        print(json.dumps({"task":args.task,"commit":SOURCE_COMMIT,
                          "python":sys.version.split()[0],"torch":__import__("torch").__version__,
                          "gpu":__import__("torch").cuda.is_available(),
                          "trained_U2_Stage4_checkpoint_loaded":False}),flush=True)
        if args.task == "verify":
            run([sys.executable,"U3/test_evolution.py"],soma_root)
            run([sys.executable,"U3/test_u3.py"],soma_root)
            print("STATUS: CPU/GPU ENGINEERING WIRING TESTS PASSED; NO TRAINED MODEL CLAIM",flush=True)
            return
        artifacts = workspace / "results"
        artifacts.mkdir()
        args_ = [sys.executable,"U3/train_esc50.py",
                 "--data-dir",str(workspace/"esc50"),
                 "--run-dir",str(artifacts),
                 "--epochs",str(args.epochs),
                 "--samples",str(args.samples),
                 "--batch-size","8",
                 "--threads","2",
                 "--seed","2501"]
        run(args_,soma_root)
        metrics = artifacts / "metrics.json"
        if not metrics.is_file():
            raise RuntimeError("training exited without a metrics report")
        obj = json.loads(metrics.read_text())
        print(json.dumps({"status":"AUDIO5_COMPLETE","heldout":obj.get("heldout_test"),
                          "classes":obj.get("classes"),"path":str(artifacts)},default=str),flush=True)
        if args.repo_id:
            upload_results(artifacts,repo_id=args.repo_id,task=args.task)
        else:
            print("NO PERMANENT ARTIFACT UPLOAD requested; Job storage is ephemeral",flush=True)

if __name__ == "__main__":
    main()
