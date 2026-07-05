# SFT training job for the Weaver MoE base — packaged as an oncompute.ai
# Compute-to-Data container (also runs on any NVIDIA GPU box, unchanged).
#
# BUILD (context = pretrain_moe/ so the COPYs resolve):
#   cd CascadeProjects/windsurf-project/pretrain_moe
#   docker build -f distill/oncompute_job/Dockerfile -t weaver-sft:latest .
#
# The base checkpoint (2.3GB, gitignored) is NOT baked in — it's provided at
# run time: on oncompute publish it as the input dataset (mounted under
# /data/inputs/); on a plain GPU box pass BASE_CKPT=/path/ckpt_latest.pt.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app
COPY distill/oncompute_job/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Preserve the repo layout so sft_train.py's `from model import ...` (it adds the
# parent dir to sys.path) resolves: model.py must sit one level above distill/.
COPY model.py                                   /app/pretrain_moe/model.py
COPY distill/sft_train.py                       /app/pretrain_moe/distill/sft_train.py
COPY distill/agentic_traces.jsonl               /app/pretrain_moe/distill/agentic_traces.jsonl
COPY kaggle/code_dataset/tokenizer/tokenizer.json /app/pretrain_moe/tokenizer.json
COPY distill/oncompute_job/run_job.py           /app/run_job.py

CMD ["python3", "/app/run_job.py"]
