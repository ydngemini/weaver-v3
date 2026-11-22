# Native speech command noise rejection

The recurrent SOMA command classifier previously accepted 146 of 408
Speech Commands background examples. This experiment compares keeping that
checkpoint and changing its decision threshold with fine-tuning it on
additional synthetic or recorded background. The streaming model and
checkpoint format are shared by all four arms.

## Data and controls

The speech partition and original checkpoint come from the
[native speech command pilot](NATIVE_SPEECH_COMMANDS_PILOT.md). The real
negatives use only the 400 [ESC-10](https://github.com/karolpiczak/ESC-50#license)
clips distributed under CC BY. The original sound-recording IDs are checked
for overlap across train folds 1–3, validation fold 4, and test fold 5.
ESC-10 credits are listed in the upstream
[LICENSE](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE).

Both fine-tuning arms start from the identical recurrent checkpoint and
see 20,044 examples per epoch for six epochs. This includes 1,200 extra
non-command windows. The synthetic arm matches each recorded window's
RMS using generated white noise. The recorded arm uses five one-second
windows per training clip and varies the gain of alternating windows.
The same shuffle, optimizer, and speech augmentation run in both arms.

Threshold calibration allows at most 10% false accepts within each
validation slice of other spoken words, generated quiet, and recorded
environmental sounds. Each fine-tuning arm selects the epoch with the
most correct accepted commands under its calibrated threshold. The
Speech Commands test is a frozen baseline replay from the earlier study.
ESC-10 fold 5 supplies a fresh holdout of 400 windows from 57 original
sound recordings. The test-only audio is excluded from threshold and
checkpoint selection.

## Held-out results

| Policy | Clean commands correctly accepted | Speech Commands background accepted | ESC-10 noise-only windows accepted | Commands correctly accepted with ESC-10 noise at 10 dB |
|---|---:|---:|---:|---:|
| Original | 92.9% | 146/408 | 304/400 | 83.7% |
| Original weights, recalibrated threshold | 87.6% | 3/408 | 42/400 | 71.3% |
| Fine-tuned with synthetic negatives | 90.4% | 10/408 | 35/400 | 76.5% |
| Fine-tuned with recorded negatives | 94.5% | 3/408 | 20/400 | 86.4% |

The recorded-negative checkpoint reduced the number of ESC-10 source
recordings with any false accept from 55/57 to 9/57. Unsupported spoken
words were accepted 43/408 times, up from 37/408 with the original
checkpoint. Background gains therefore coexist with a small regression
on the spoken-word reject slice. The final checkpoint stays in offline
evaluation. The ESC-10 windows from the same recording are correlated,
and neither benchmark measures hourly false triggers in continuous audio.

One warm CPU thread processed each scored 20 ms packet in 0.28 ms p95
for the recorded-negative checkpoint. The maximum logit difference
between packetized and whole-clip inference was 0.0000053 across the
latency sample.

## Reproduce

Use the speech archives and recurrent checkpoint from the first pilot.
The trainer uses CPU PyTorch, NumPy, and SciPy. From the repository root:

```bash
curl -fL https://codeload.github.com/karolpiczak/ESC-50/zip/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6 \
  -o esc50.zip
unzip -q esc50.zip -d data
python Weaver-Soma/U3/train_speech_noise_rejection.py \
  --train-archive speech-data/speech_commands_v0.02.tar.gz \
  --test-archive speech-data/speech_commands_test_set_v0.02.tar.gz \
  --esc data/ESC-50-33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6 \
  --starting-checkpoint runs/speech-commands/soma_recurrent.pt \
  --out runs/speech-noise-rejection
python Weaver-Soma/U3/native_speech_commands.py \
  --checkpoint runs/speech-noise-rejection/recorded_negatives.pt \
  --wav sample-16k.wav
python -m unittest discover -s Weaver-Soma/U3 -q
```

The output includes source hashes, per-epoch calibration, per-command
decisions, and per-category ambient false-accept counts. Prior work on
[negative examples in keyword spotting](https://doi.org/10.1109/ICASSP40776.2020.9053009)
motivated the targeted comparison. The
[ESC paper](https://doi.org/10.1145/2733373.2806390) provides the provenance
for the original sound recordings.
