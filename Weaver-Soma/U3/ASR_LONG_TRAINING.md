# Long native acoustic training

This run continues the expansion checkpoint from the
[data-scaling pilot](ASR_DATA_SCALE_PILOT.md). It preserves the input
checkpoint, trains working weights, and releases a new retained checkpoint
only after a strict held-out comparison. The CTC alphabet, causal acoustic
architecture, and frozen character-language decoder remain compatible
with the existing WAV entry points.

## Training

The pinned [LibriSpeech](https://www.openslr.org/12/) corpus is read in stages
from train-clean-100, train-clean-360, and train-other-500. For each shard, the trainer records its SHA-256 hash. The source revision is
pinned by the stored catalog. Speaker IDs and utterance IDs
are checked against the held-out partitions and earlier shards. Missing
training shards are downloaded in a rolling window and removed from the
cache after consumption. Interrupted transfers resume from their saved
byte offsets. Previously downloaded input files are left in place.

CTC continuation uses AdamW, the 32-utterance batching protocol from the
native training pilot, and a learning rate of 0.00065, 0.00052, or 0.00035
within each successive partition. A small speaker-balanced replay buffer
from the original training data is revisited every ten optimizer steps.
The journal records new-corpus hours separately from replay hours. The
optimizer state and trained shard list are atomically saved after every
completed shard, so a retry can resume without repeating committed shards.

## Checkpoint selection

The starting checkpoint is immutable. Working weights continue through
the corpus and can be worse than the retained weights at any given point.
Every eight shards and at each partition boundary, a candidate must
match or improve all recorded development scores to replace the saved
development candidate. The development guard checks full dev-clean greedy
word and character error and speaker-balanced fused word error on both
dev-clean and dev-other.

At completion, the candidate and starting checkpoint are evaluated on
test-clean and test-other using greedy and fused decoding. The retention
gate requires non-increasing word and character error for all four runs.
A failed gate retains the starting checkpoint. The report includes the
candidate's held-out scores either way.

The run journal records each checkpoint decision and the corpus hours
represented by the saved candidate. Keep the journal with the checkpoints.

## Reproduce and resume

Use the pinned LibriSpeech catalog and input checkpoint and decoder
artifacts from the earlier pilots. Stage the existing train-clean-100,
dev-clean, test-clean, and test-other Parquet inputs as documented in the
[data-scaling pilot](ASR_DATA_SCALE_PILOT.md). Stage the dev-other shard
as `other-validation-0000.parquet`. Use the LibriSpeech dataset API
catalog saved as `hf_catalog.json` at revision
`71cacbfb7e2354c4226d01e70d77d5fca3d04ba1`. The trainer can fetch any
missing training shards from that same revision.

```bash
python Weaver-Soma/U3/train_asr_long.py \
  --data libri --checkpoint runs/asr-data-scale/expanded_data_asr.pt \
  --decoder-dir runs/asr-beam --out runs/asr-long --partitions 3
```

Re-running the same command resumes from `resume.pt`. The
`--max-hours` option pauses after the first complete shard that reaches
the requested corpus exposure. The `--finish` option evaluates the
saved development candidate and anchor with no further training.

[LibriSpeech](https://doi.org/10.1109/ICASSP.2015.7178964) provides the
16 kHz read speech and official split definition. Its Hugging Face
[Parquet publication](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1)
specifies the CC BY 4.0 corpus license.
