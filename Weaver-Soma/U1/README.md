# WEAVER-SOMA U1: single trained synthetic body-cognition model

This is the **original U1 source** recovered from the research archive. The original pretrained checkpoint is intentionally excluded from this public GitHub folder.

U1 jointly learns from three masked numerical sensor streams (pressure, temperature, joint load), an eight-step recurrent body state, five trainable experts, a global workspace, abstract episodic summaries, external-event likelihood, action-conditioned futures and risk predictions. Its interface is implemented in unified_model.py; Weaver adapters in weaver_adapter.py are advisory-only.

## Run on CPU

~~~bash
python -m pip install -r requirements.txt
python train_unified.py --epochs 30 --train-n 8400 --val-n 2100
python -m pytest -q test_unified.py
python demo_inference.py
python evaluate_policy.py
python evaluate_ablation.py
~~~

The training script creates its own local checkpoint in results/ from synthetic simulator episodes. To use the **recovered checkpoint from epoch 24** instead, restore a trusted copy into results/weaver_soma_unified.pt and verify its SHA-256:
3eb0746806f3842ec2d8121a7bcf30f5351b1b26b70e2c53debdac2ea5030c8f.

Original checkpoint: 121,827 model parameters, 91 named tensors. Weights are not retrained or re-created by merely adding source code to GitHub.

## Scientific boundaries

Prior synthetic results are in MODEL_CARD.md. Training labels and physics are simulated; measurements are not calibrated for humans or real hardware. The model is not an LLM, has no trained camera/voice inputs, and does not exhibit verified subjective experience. Weaver's signed intent permissions, privacy boundaries, and independent reflex controls must remain separate.

This source is committed for reproducible research, **not deployed to live Weaver v3**.
