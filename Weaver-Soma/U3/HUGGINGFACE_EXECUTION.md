# SOMA U3: Run native model experiments on Hugging Face

**Prepared:** November 20, 2026. **Status:** runnable job script committed to GitHub; no Hugging Face Jobs started and no Hub repositories created. The connected Linux device has the Hugging Face CLI installed, but is **not authenticated** to an account.

## Platform choice

Use Hugging Face **Jobs** for bounded CPU/GPU experiments and a **private Model Hub repository** for checkpoints, results, and model cards. Use Spaces only later for an optional demo dashboard. Jobs require prepaid credits and charge for compute (including CPU). Never launch a billable job without user approval.

## Step 1: Sign in securely on Linux

    hf auth login
    hf auth whoami

Never paste tokens into chat or store tokens in GitHub. An account with Job access and pre-paid credits is required.

## Step 2: Run U3 validation (billable CPU)

    hf jobs uv run --flavor cpu-basic --timeout 30m \
      https://raw.githubusercontent.com/ydngemini/weaver-v3/22e17f72b7ed04a15bd5513c8213ef54d258d5ca/Weaver-Soma/U3/hf_jobs_runner.py --task verify

This downloads and runs pinned SOMA source at commit 332d2cb33af4327403161f4cc54aa203ef18408f, including existing U3 multimodal tests and new neural-growth tests. PASS means source mechanics pass; it does **not** mean U2 Stage 4 trained checkpoint was restored.

## Step 3: Real environmental sound research experiment (billable CPU)

The code uses a native untrained audio encoder and 200 original licensed ESC-50 recordings: dog, rain, siren, coughing, sneezing. Folds 1-3 train, fold 4 validation, fold 5 held out. This is not a 50-class benchmark or 100M full-core pretraining. Dataset license CC BY-NC 3.0, noncommercial.

Non-persistent trial:

    hf jobs uv run --flavor cpu-upgrade --timeout 2h \
      https://raw.githubusercontent.com/ydngemini/weaver-v3/22e17f72b7ed04a15bd5513c8213ef54d258d5ca/Weaver-Soma/U3/hf_jobs_runner.py --task audio5 --epochs 8

If you explicitly approve preserving results in your own **private** Hub model repo:

    hf jobs uv run --flavor cpu-upgrade --timeout 2h -s HF_TOKEN \
      https://raw.githubusercontent.com/ydngemini/weaver-v3/22e17f72b7ed04a15bd5513c8213ef54d258d5ca/Weaver-Soma/U3/hf_jobs_runner.py --task audio5 --epochs 8 \
      --repo-id YOUR_HF_USERNAME/SOMA-U3-research

The upload option asks Hugging Face to create the repo as private and refuses an existing public repository. HF_TOKEN is passed by the CLI as an encrypted secret from the local authentication context; never put a token in the command text itself.

## Monitor and cancel

    hf jobs list
    hf jobs inspect JOB_ID
    hf jobs logs JOB_ID
    hf jobs cancel JOB_ID

## Pricing and limitations

Published rates at plan preparation: CPU-basic Jobs about $0.01/hr, CPU-upgrade about $0.03/hr, T4-small GPU $0.40/hr. These are estimates, not an authenticated quote. The current audio training code is CPU-only and will not benefit from a paid GPU flavor without making device-awareness changes first. Run the CPU experiments before buying GPU time. See https://huggingface.co/pricing and https://huggingface.co/docs/hub/en/jobs-quickstart .

## Next gates

1. Authenticate account and explicitly approve any intended Job cost.
2. Run cloud U3 native tests; capture detailed logs and fail cases.
3. Run held-out ESC-50 and save metrics in private model repository.
4. Independently recover and hash-validate missing ~401 MB Stage 4 U2 weights before claiming learned U2-to-U3 transfer.
5. Build real olfactory sensor experiments and multi-seed, compute-matched neural evolution comparisons.
6. Only then make GPU training device-aware and evaluate cost / speed benefits.

GitHub runner: https://raw.githubusercontent.com/ydngemini/weaver-v3/22e17f72b7ed04a15bd5513c8213ef54d258d5ca/Weaver-Soma/U3/hf_jobs_runner.py

Source branch: https://github.com/ydngemini/weaver-v3/tree/feature/weaver-soma-u3-multimodal-evolution/Weaver-Soma/U3
