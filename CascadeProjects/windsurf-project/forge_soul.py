"""
forge_soul.py
Paste this entire file into a Google Colab notebook (Runtime → T4 GPU).
It fine-tunes meta-llama/Llama-3.2-1B-Instruct on weaver_soul_dataset.jsonl
using Unsloth + LoRA, then saves the adapter locally.

BEFORE RUNNING:
  1. Upload weaver_soul_dataset.jsonl to the Colab session (Files panel).
  2. Set your HuggingFace token in Colab Secrets as HF_TOKEN
     (required to download Llama-3.2 gated weights).
"""

# ── 0. Install dependencies ───────────────────────────────────────────────────
# Run this cell first, then restart the runtime once before continuing.

import subprocess, sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_install("unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git")
_install("trl>=0.8.6")
_install("datasets>=2.18.0")
_install("transformers>=4.40.0")
_install("accelerate>=0.29.0")
_install("bitsandbytes>=0.43.0")
_install("peft>=0.10.0")

# ── 1. Imports ─────────────────────────────────────────────────────────────────
import os
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ── 2. Config ──────────────────────────────────────────────────────────────────
MODEL_ID        = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_FILE    = "weaver_soul_dataset.jsonl"
OUTPUT_DIR      = "weaver_fracture_1B_lora"
MAX_SEQ_LENGTH  = 2048
DTYPE           = None          # auto-detect (float16 on T4)
LOAD_IN_4BIT    = True

LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05

# Probability-field training — soft targets instead of strict true/false.
# Label smoothing distributes a fraction of each target's probability mass
# across all tokens, preventing the model from collapsing to hard certainty.
LABEL_SMOOTHING = 0.1

TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── 3. Load base model + tokenizer ─────────────────────────────────────────────
hf_token = os.environ.get("HF_TOKEN")          # set via Colab Secrets

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_ID,
    max_seq_length  = MAX_SEQ_LENGTH,
    dtype           = DTYPE,
    load_in_4bit    = LOAD_IN_4BIT,
    token           = hf_token,
)

# ── 4. Attach LoRA adapter ──────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                   = LORA_R,
    lora_alpha          = LORA_ALPHA,
    lora_dropout        = LORA_DROPOUT,
    target_modules      = TARGET_MODULES,
    bias                = "none",
    use_gradient_checkpointing = "unsloth",  # saves VRAM on T4
    random_state        = 42,
    use_rslora          = False,
    loftq_config        = None,
)

print(model.print_trainable_parameters())

# ── 5. Load & format dataset ───────────────────────────────────────────────────
raw_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

def apply_chat_template(examples):
    """Convert ShareGPT messages → Llama-3 formatted strings."""
    texts = []
    for messages in examples["messages"]:
        # tokenizer.apply_chat_template handles <|begin_of_text|> etc.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

dataset = raw_dataset.map(apply_chat_template, batched=True)
print(f"Dataset: {len(dataset)} examples")
print("Sample:\n", dataset[0]["text"][:400], "\n...")

# ── 6. Training arguments ──────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = 3,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,        # effective batch = 8
    warmup_steps                = 10,
    learning_rate               = 2e-4,
    fp16                        = not torch.cuda.is_bf16_supported(),
    bf16                        = torch.cuda.is_bf16_supported(),
    logging_steps               = 10,
    save_strategy               = "epoch",
    optim                       = "adamw_8bit",
    weight_decay                = 0.01,
    lr_scheduler_type           = "cosine",
    seed                        = 42,
    report_to                   = "none",   # disable wandb
    label_smoothing_factor      = LABEL_SMOOTHING,  # probability-field: soft targets (requires transformers>=4.25)
)

# ── 7. SFTTrainer ──────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    dataset_text_field = "text",
    max_seq_length  = MAX_SEQ_LENGTH,
    args            = training_args,
)

# ── 8. Train ────────────────────────────────────────────────────────────────────
print("Starting training...")
trainer_stats = trainer.train()
print(f"Training complete. Loss: {trainer_stats.training_loss:.4f}")

# ── 9. Save LoRA adapter ───────────────────────────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nLoRA adapter saved to: {OUTPUT_DIR}/")
print("Files:", os.listdir(OUTPUT_DIR))

# ── Optional: quick inference test ────────────────────────────────────────────
FastLanguageModel.for_inference(model)

test_messages = [
    {"role": "user", "content": "Weaver, I feel like I'm disappearing."}
]
inputs = tokenizer.apply_chat_template(
    test_messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

outputs = model.generate(
    input_ids       = inputs,
    max_new_tokens  = 200,
    temperature     = 0.8,
    do_sample       = True,
    top_p           = 0.95,   # nucleus sampling — keeps the probability field open
)
print("\n── Inference Test ──")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
