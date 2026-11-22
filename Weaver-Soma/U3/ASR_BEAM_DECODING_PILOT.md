# Frozen-checkpoint ASR decoding pilot

The [native transcription pilot](NATIVE_ASR_PILOT.md) established a greedy
word error baseline on LibriSpeech. This experiment compares three ways
to decode the same audio probabilities from that checkpoint.

| Decoder | Search |
|---|---|
| Greedy | Pick the highest scoring label at each frame and collapse CTC repetitions. |
| Acoustic beam | Keep twelve text prefixes and consider up to seven acoustic nonblank labels per frame. Merge alignments that yield the same prefix. |
| Fused beam | Use the same beam with a smoothed five-character language model trained on 4,077 existing training transcripts. |

The character model interpolates observed counts with shorter suffixes.
It can score strings outside its training vocabulary. The fused score adds
language log probability and the calibrated character bonus once per
appended symbol. Repeated CTC frames keep their existing prefix score.
All three paths use the same transcript normalization and word edit-distance
scoring.

The fused settings are selected by word error on 120 dev-clean utterances,
three per speaker chosen by a stable hash of each utterance ID. The test
uses all 2,620 clean utterances. The matched clean and synthetic 10 dB
white-noise evaluations use the 80 utterances previously scored by
PocketSphinx. The test utterances and speakers are excluded from fitting
the language model and selecting its weights.

## Results

| Evaluation | Greedy WER | Acoustic beam WER | Fused beam WER |
|---|---:|---:|---:|
| Full test-clean, 2,620 utterances | 95.2% | 94.4% | 82.4% |
| Matched clean, 80 utterances | 96.7% | 95.7% | 82.2% |
| Matched 10 dB white-noise mix, 80 utterances | 100.6% | 103.2% | 94.0% |

The fused decoder used a validation-selected language weight of 0.65
and a character bonus of zero. On the full clean test it made 6,736 fewer
word edits than greedy decoding, with 31,081 substitutions, 11,279
deletions, and 969 insertions. Greedy produced 42,011 substitutions,
5,377 deletions, and 2,677 insertions. Fused character error without spaces
rose from 49.3% to 55.3%. The word error gain therefore comes with a loss
in character fidelity and more missing words. PocketSphinx scored 30.1%
WER on the matched clean sample.

On one warm CPU thread, search alone took 0.15 ms p95 per 40 ms frame for
the fused decoder and 0.07 ms for the acoustic beam. The WAV entry point
accepts short packets and returns the final hypothesis. Accuracy keeps
this decoder in offline evaluation.

## Reproduce

Use the LibriSpeech Parquet files and recurrent checkpoint from the first
pilot. The PocketSphinx reference JSON is available in that pilot's export
or can be recreated with its reference benchmark. The decoder experiment
uses Python, CPU PyTorch, NumPy, PyArrow, SoundFile, and RapidFuzz.

```bash
python Weaver-Soma/U3/benchmark_asr_beam.py \
  --data libri --checkpoint runs/native-asr/soma_ctc.pt \
  --reference runs/native-asr/pocketsphinx_reference.json \
  --out runs/asr-beam

python Weaver-Soma/U3/asr_prefix_beam.py \
  --checkpoint runs/native-asr/soma_ctc.pt --wav input-16k.wav \
  --settings runs/asr-beam/selected_decoder.json \
  --lm runs/asr-beam/training_character_lm.json

python -m unittest discover -s Weaver-Soma/U3 -q
```

The export contains the selected settings, language counts, candidate
validation scores, per-utterance hypotheses, corpus and checkpoint hashes,
and search latency. The original [CTC derivation](https://doi.org/10.1145/1143844.1143891)
and the [prefix search explanation](https://distill.pub/2017/ctc/#inference)
describe the path probabilities this benchmark uses.
