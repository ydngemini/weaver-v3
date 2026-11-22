# Native streaming speech command pilot

SOMA learned ten spoken commands from its own initialization on CPU. The
recurrent head consumes 16 kHz mono PCM16 and carries a compact GRU state
and a running mean and maximum of its acoustic activations. Each complete
25 ms window contributes a log-mel frame. Scores update every 20 ms after
the first full frame. For this pilot, the CLI reports one decision per
complete WAV. Partial scores are useful for replay diagnostics.

## Corpus and controls

The pilot uses [Speech Commands v0.02](https://www.tensorflow.org/datasets/catalog/speech_commands)
by Pete Warden, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The training and calibration subsets come from the official lists. The
evaluation uses the separate official test archive. Speaker IDs were checked
across all three partitions. The 18,844 training and 5,327 validation clips
cover 1,970 and 250 identified speakers, respectively. The 4,890 test clips
include 239 identified speakers and 408 real background examples.

Both heads saw the same data, 18 optimizer epochs, and the same clean and
10 dB white-noise features. The pooling baseline pools the log-mel frame
sequence directly. The SOMA head learns 64-dimensional recurrent features
before pooling. It has 29,388 parameters. Checkpoint selection uses
validation macro accuracy. The acceptance cutoff is set using validation to
allow at most 10% false accepts across unsupported words and silence.

Training and validation quiet examples were generated noise. The official
test background recordings were held out entirely. The additive white-noise
test perturbs spoken clips and preserves the background examples. Room,
microphone, overlap, wake-word, and continuous false-trigger rates require
separate evaluations.

## Held-out scores

| Condition | Head | Known commands correctly accepted | Unsupported audio accepted |
|---|---|---:|---:|
| Clean | Pooling baseline | 18.5% | 19.1% |
| Clean | SOMA recurrent | 92.9% | 22.4% |
| Added white noise at 10 dB | Pooling baseline | 31.4% | 26.2% |
| Added white noise at 10 dB | SOMA recurrent | 87.0% | 24.9% |

On the clean recurrent run, 37 of 408 unsupported words and 146 of 408
background clips were accepted as commands. The background slice explains
most of the gap with the 10% validation target. The recurrent candidate
stays in offline evaluation. The 250 ms prefix accepted 49.6% of unsupported
clips under the full-clip decision rule. Keep prefix scores provisional.

The recurrent head took 0.36 ms p95 per scored 20 ms packet on one warm CPU
thread. This measurement covers processing an already-arrived packet.
The packetized and batched recurrent logits differed by at most
0.0000041 across the latency sample. CPU training and validation for that
head took about 64 seconds after acoustic features were prepared.

## Reproduce

From the repository root, use Python 3.11 or later, NumPy and CPU PyTorch.

```bash
mkdir -p speech-data
curl -fL https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz \
  -o speech-data/speech_commands_v0.02.tar.gz
curl -fL https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz \
  -o speech-data/speech_commands_test_set_v0.02.tar.gz
python Weaver-Soma/U3/train_speech_commands.py \
  --train-archive speech-data/speech_commands_v0.02.tar.gz \
  --test-archive speech-data/speech_commands_test_set_v0.02.tar.gz \
  --out runs/speech-commands
python Weaver-Soma/U3/native_speech_commands.py \
  --checkpoint runs/speech-commands/soma_recurrent.pt --wav sample-16k.wav
python -m unittest discover -s Weaver-Soma/U3 -q
```

The run exports both model weights, per-epoch curves, corpus SHA-256 hashes,
calibration values, and per-clip hypotheses. The inference command reports
the top label, confidence, threshold, and all twelve scores, including the
unknown and silence labels.

[Speech Commands](https://doi.org/10.48550/arXiv.1804.03209) informed corpus
partitioning and the unknown/background slices.
[Streaming keyword spotting](https://doi.org/10.48550/arXiv.2005.06720)
informed the bounded-state PCM implementation and packetized-versus-batched
parity check.
