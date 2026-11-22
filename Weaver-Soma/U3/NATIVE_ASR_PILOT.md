# Native streaming transcription pilot

SOMA's first open-vocabulary speech-to-text model learns English characters
directly from audio and whole-utterance transcripts. Two log-mel windows
feed each 40 ms model step, with 45 ms of audio needed for the first step.
A two-layer unidirectional GRU carries acoustic state between packets.
Greedy CTC decoding preserves repeated letters separated by a blank and
reports character emission offsets in the input sample clock.

## Corpus and selection

The experiment uses the CC BY 4.0
[LibriSpeech corpus](https://www.openslr.org/12/) by Panayotov and colleagues.
Two published shards of train-clean-100 provide 4,077 utterances from
38 speakers, totaling 14.51 hours. The full official dev-clean and
test-clean partitions each contain 40 speakers and about 5.4 hours.
The trainer verifies speaker and utterance separation. Checkpoint selection
uses the lowest dev-clean word error rate across twelve CPU epochs.

The model uses randomly initialized SOMA weights, greedy decoding, and
only transcript characters. In a second test condition, each clean test
recording is mixed with deterministic white noise at 10 dB SNR. This
stress case represents synthetic noise on read English speech.

## Results

| Evaluation | Word error rate | Character error rate |
|---|---:|---:|
| Official LibriSpeech test-clean, 2,620 utterances | 95.2% | 49.3% |
| Same clips with 10 dB white noise | 100.7% | 79.3% |

Character error excludes spaces. Insertions can raise word error above
100%. Clean speech produced 42,011 substitutions, 5,377 deletions, and
2,677 insertions across 52,576 reference words.

On the same fixed 80 clean utterances across 40 speakers, SOMA measured
96.7% word error and PocketSphinx 5.0.4 measured 30.1%. PocketSphinx uses
its bundled acoustic and language models with full-utterance normalization.
SOMA uses streaming acoustic evidence and greedy character decoding.
The selected SOMA checkpoint is from epoch 11 and remains an offline
accuracy baseline.

The 122,525-parameter model took 0.30 ms p95 per scored 20 ms packet
on one warm CPU thread, including greedy decoding. The benchmark supplies
already-captured audio in memory. Across 36 utterances, all streamed
hypotheses matched batched decoding and the maximum logit difference
was below 0.00001. Twelve epochs plus validation took about 402 seconds
on four CPU threads after feature preparation.

## Run the pilot

Download the exact LibriSpeech Parquet shards into a local directory.
The training command requires Python 3.11 or later, CPU PyTorch, NumPy,
PyArrow, SoundFile, and RapidFuzz. The streaming inference command uses
Python, CPU PyTorch, and NumPy.

```bash
mkdir -p libri
revision=71cacbfb7e2354c4226d01e70d77d5fca3d04ba1
for shard in train.100/0000.parquet train.100/0007.parquet validation/0000.parquet test/0000.parquet
do
  out="${shard//\//-}"
  curl -fL "https://huggingface.co/datasets/openslr/librispeech_asr/resolve/$revision/clean/$shard?download=true" \
    -o "libri/$out"
done

python Weaver-Soma/U3/train_native_asr.py --data libri --out runs/native-asr
python Weaver-Soma/U3/native_asr.py \
  --checkpoint runs/native-asr/soma_ctc.pt --wav input-16k.wav --events
python -m unittest discover -s Weaver-Soma/U3 -q
```

The optional paired reference uses PocketSphinx's bundled English acoustic
and language models. The same two utterances per test speaker are selected
by ranking SHA-256 hashes of utterance IDs. Both recognizers use the same
transcript normalization and word edit-distance scorer.

```bash
python Weaver-Soma/U3/benchmark_asr_reference.py \
  --test-parquet libri/test-0000.parquet \
  --native-predictions runs/native-asr/librispeech_test_clean.jsonl \
  --out runs/native-asr/pocketsphinx_reference.json
```

The export includes the selected checkpoint, corpus SHA-256 values, training
curves, and per-utterance references, hypotheses, and error counts.
[CTC](https://doi.org/10.1145/1143844.1143891) supplies the alignment objective.
The [LibriSpeech paper](https://doi.org/10.1109/ICASSP.2015.7178964) describes the
speaker partitions and read English material.
