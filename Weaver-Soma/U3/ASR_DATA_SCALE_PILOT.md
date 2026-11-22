# Native acoustic training with more speakers

The earlier [native transcription pilot](NATIVE_ASR_PILOT.md) trained a
122,525-parameter causal CTC recognizer. The [decoder pilot](ASR_BEAM_DECODING_PILOT.md)
reduced word error with a fixed transcript language model. This experiment
continues acoustic training while keeping the model architecture, starting
weights, optimizer settings, and evaluation decoders fixed.

## Training and evaluation

Both arms receive 384 updates of 32 utterances with the same AdamW learning
rate and seed. The replay arm samples from the original two training shards.
The expanded arm adds four disjoint training shards from train-clean-100.
Each expanded minibatch is paired with a replay minibatch by nearest padded
acoustic frame count. An expanded minibatch is reused after its pool has
been exhausted. The report distinguishes total clip exposures from the
number of unique utterances seen.

Both arms select a checkpoint by dev-clean greedy word error at updates
0, 128, 256, and 384. Test evaluation runs after selection. The clean
partition repeats the prior benchmark. Test-other is a new holdout for this
project. The two official test partitions, development speakers, and
training speakers are disjoint. Both greedy and fused decoders use the
same configurations across all checkpoints. The fused character language
model remains the one fitted in the earlier decoder experiment.

## Results

| Selected checkpoint | Clean greedy WER | Clean fused WER | Other greedy WER | Other fused WER |
|---|---:|---:|---:|---:|
| Published starting checkpoint | 95.22% | 82.41% | 98.36% | 88.24% |
| Replay original data | 95.22% | 82.41% | 98.36% | 88.24% |
| Expanded data | 93.94% | 82.40% | 96.83% | 88.32% |

The original training set contains 14.51 hours from 38 speakers. The
expanded set contains 42.82 hours from 111 speakers. Both runs made
12,288 clip exposures, with 3,945,696 padded acoustic frames for replay
and 3,871,072 for expanded training. Replay retained the starting weights
after validation. Expanded training selected update 384, at 94.33%
dev-clean WER compared with 95.19% at the starting checkpoint.

The expanded model reduced greedy test-other WER by 1.53 percentage
points, with a paired 95% interval from 0.91 to 2.28 points lower. Fused
test-other WER changed by +0.08 points, with an interval from −0.27 to
+0.42 points. On test-clean, fused WER changed by −0.01 points, with an
interval from −0.40 to +0.39 points. This run found consistent improvement
in acoustic-only decoding and no consistent improvement through the
existing language decoder.

Training and development evaluation took about 107 seconds per arm on
four CPU threads. The expanded checkpoint decoded 36 replay utterances
with matching streamed and batched greedy text. CPU packet processing
was 0.39 ms p95 on one thread with preloaded audio. The checkpoints
remain offline experiments.

The interval for the difference in word error resamples held-out speaker
clusters within each paired evaluation, so clips from the same speaker
remain grouped. Both test partitions contain read speech.

## Reproduce

Use the original [LibriSpeech inputs and checkpoint](NATIVE_ASR_PILOT.md)
and the [selected decoder and character model](ASR_BEAM_DECODING_PILOT.md).
Download the four extra training shards and the test-other shard from the
same pinned [LibriSpeech revision](https://huggingface.co/datasets/openslr/librispeech_asr/tree/71cacbfb7e2354c4226d01e70d77d5fca3d04ba1).

```bash
mkdir -p libri
revision=71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
for shard in clean/train.100/0002.parquet clean/train.100/0005.parquet clean/train.100/0009.parquet clean/train.100/0012.parquet other/test/0000.parquet
do
  relative="${shard#clean/}"
  output="${relative//\//-}"
  curl -fL "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/$revision/$shard?download=true" \
    -o "libri/$output"
done

python Weaver-Soma/U3/train_asr_data_scale.py \
  --data libri --checkpoint runs/native-asr/soma_ctc.pt \
  --decoder-dir runs/asr-beam --out runs/asr-data-scale

python -m unittest discover -s Weaver-Soma/U3 -q
```

The export includes both selected acoustic checkpoints, the full update
histories and exposure counts, frozen decoder and dataset hashes, paired
intervals, and per-utterance predictions. The
[LibriSpeech paper](https://doi.org/10.1109/ICASSP.2015.7178964) and
[OpenSLR corpus page](https://www.openslr.org/12/) document its source,
license, and partitions.
