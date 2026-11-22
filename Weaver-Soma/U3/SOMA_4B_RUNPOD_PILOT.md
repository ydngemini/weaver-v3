# SOMA 4B Runpod pilot

SOMA's language core starts with random weights and its own 32,768-token byte-level BPE tokenizer. It has 40 causal transformer blocks, 3,072 hidden dimensions, 24 query heads, 8 key/value heads and a 7,936-wide SwiGLU layer. It uses rotary positions, RMSNorm and a tied input and output matrix. The default profile has **4,033,072,128 trainable parameters** and supports sequences of up to 2,048 tokens. This pilot covers the text core. Speech and other encoders have their own training runs and checkpoints.

## Training data

| Dataset | Place in the plan |
| --- | --- |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Start pretraining on the pinned `sample-10BT` subset, published under ODC-BY. The script preserves the dataset attribution, revision, source URLs and hashes. The card's token count uses a different tokenizer. |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | Evaluate as a later base-corpus expansion for papers, code and general text. The card lists ODC-BY. Measure and remove overlap before blending it with FineWeb-Edu. |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Test as a small synthetic-text mixture against a matched-token FineWeb-Edu baseline after the first pilot. The dataset card lists Apache-2.0. |
| [UltraChat 200K](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | Consider for supervised dialogue tuning after base-model training. |
| [UltraFeedback Binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) | Consider for preference tuning after supervised tuning. The card lists MIT. |
| [Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset) | Candidate for later multilingual instruction tuning. Track the gain by language on held-out prompts. The card lists Apache-2.0. |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | Candidate for a code-only pretraining mixture after access and per-repository license/attribution checks. Reserve code benchmarks before ingesting. |
| [OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | Candidate for a later math fine-tuning experiment. Review the dataset terms and remove overlaps with the math benchmarks. |
| [Paloma](https://huggingface.co/datasets/allenai/paloma) and [BigCodeBench](https://huggingface.co/datasets/bigcode/bigcodebench) | Reserve for evaluation. Decontaminate the production corpus against selected benchmarks before measuring them. |

The pilot contains 4,096 FineWeb-Edu documents, including 3,903 training and 193 validation documents. Its tokenizer produced **4,001,396 training tokens** and **227,737 validation tokens**. A hash of each normalized document determines its split and removes exact duplicates. Only training documents contribute to the vocabulary. The split checks the pipeline and short-run loss. Broader evaluations will require near-duplicate and benchmark-overlap filtering across sources, source sample reviews and separate out-of-domain sets.

## Run configuration

Use one Secure Cloud RTX PRO 6000 Blackwell Server Edition 96 GB with the official Runpod PyTorch `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` image and CUDA host floor 12.8. Runpod listed **$2.69 per GPU hour** on November 22, 2026, with low stock in US-MO-2, US-NC-1 and US-NE-1. Each region also listed Standard network volumes. Recheck both before deployment. Use a 30 GB container disk and a 120 GB Standard network volume mounted at `/workspace` in the same region. The network volume holds the corpus, logs and full resume state. The workload requires no inbound application ports.

The FP32 weights, gradients and AdamW moment tensors total about 64.5 GB in GPU memory. BF16 compute, activation checkpointing and microbatch 1 leave the rest for transient buffers and activations. The full GPU peak must be measured. A resume checkpoint needs about 48.4 GB for weights and moments. Atomic replacement can briefly require two files.

At the listed rates, 90 minutes is about **$4.06** for GPU compute and storage during the run, before tax. Standard 120 GB network storage costs about **$8.40 per month** for as long as it is retained. Verify the selected region's storage tier and rate before provisioning. Runpod [requires an hour of credit for on-demand deployment](https://docs.runpod.io/pods/pricing). Set an external stop deadline since the trainer's time limit controls training and cannot stop GPU billing. Keep checkpoints on the network volume through the GPU shutdown and account for storage retention separately.

From the checked-out `Weaver-Soma/U3` directory:

```bash
python -m pip install 'numpy==2.2.6' 'tokenizers==0.21.4' 'huggingface-hub==0.36.2' 'pyarrow==26.0.1'
python prepare_soma_4b.py --documents 4096 --output /workspace/soma/data
set -o pipefail
python -u train_soma_4b.py --data /workspace/soma/data --output /workspace/soma/run \
  --context 512 --microbatch 1 --accumulate 8 --steps 4 --max-seconds 900 \
  --checkpoint-every 128 | tee /workspace/soma/pilot.jsonl
python -u train_soma_4b.py --data /workspace/soma/data --output /workspace/soma/run \
  --context 512 --microbatch 1 --accumulate 8 --steps 128 --max-seconds 3000 \
  --checkpoint-every 128 --resume | tee /workspace/soma/resume.jsonl
```

First confirm four finite-loss steps, a durable checkpoint and peak GPU memory. Resume the same corpus and settings to a maximum of 128 steps total, or 524,288 next-token targets. Confirm that validation loss improves from step zero and use measured tokens per second and cost per million tokens to price the next stage. Keep the corpus manifest and logs with the model. A sustained full pretraining run needs a separate budget calibrated from those measurements. The [Chinchilla study](https://arxiv.org/abs/2203.15556) provides a scaling reference. At roughly 20 tokens per parameter, this model's reference point is about 81 billion tokens.

## Local verification

The full profile's parameter count and output shape were verified on PyTorch's meta device. A 106,816-parameter CPU configuration trained on 16,384 targets from the same FineWeb-Edu source, with validation loss falling from 6.2486 to 5.9806. Resuming from step 16 gave bit-identical model weights, optimizer state and CPU RNG at step 32 compared with continuous training. Causality and gradient equivalence with activation checkpointing were checked separately. A second CPU rehearsal used the full 32,768-token pilot vocabulary with a 2,171,200-parameter model. After 256 steps and 131,072 next-token targets, validation cross-entropy fell from 10.4092 to 7.8216 nats per token. Step 128 and the reload both reported 8.7072 on the same validation slice. A fresh materialization of the 4,096 documents reproduced the prior tokenizer, corpus and document-index SHA-256 hashes. Full-size CUDA training and peak VRAM remain to be measured on Runpod.

The A100 SXM4 80 GB was listed at $2.19/hour, with in-stock U.S. regions that showed no Standard or High Performance network volumes at the time of the check. Prefer the 96 GB setup for its extra memory and compatible durable storage. Runpod [global volumes](https://docs.runpod.io/storage/globalvolume/overview) lack file locks and atomic renames. The trainer uses both for checkpoint safety.
