# Acoustic endpoint pilot

A CPU-trained SOMA head evaluated when the system could stop waiting after a
quiet interval. The control uses the same 20 ms, RMS 0.012 energy gate in the
offline candidate extractor. The head sees the one second of mono waveform
ending just before each quiet interval. Its ten features cover energy, zero
crossings, periodicity, and a pitch proxy. A regularized one-layer PyTorch model
logs the individual feature contributions to its score.

## Recorded speech evaluation

The [AMI corpus](https://groups.inf.ed.ac.uk/ami/corpus/) supplied 12 public
headset recordings and [manual v1.6.2 annotations](https://groups.inf.ed.ac.uk/ami/download/).
They are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The benchmark uses six meetings for training, three for calibration, and
three for testing. It checks every room participant's AMI speaker ID for
cross-split overlap, including voices that could bleed into another headset.

The benchmark counts a quiet run as internal when sustained energy returns
at least 160 ms before the end of the same transcribed speech segment. It
counts a segment end when a quiet run of at least 600 ms starts within
160 ms of that boundary and the next segment starts at least 900 ms after
the boundary. These labels test pause handling against annotation segments.
They are proxies for conversational intent. The energy gate conditions which
events reach the evaluation. The score excludes captured playback, echo,
device clocks, and end-to-end response time.

The acoustic policy selects 300 or 600 ms. Calibration allows at most 2%
additional early cuts among pauses from 300 ms up to 600 ms. Calibration
selected an early cutoff for 1 of 27 segment ends and 5 of 275 short pauses.
The cutoff was then fixed for the held-out speakers.

| Held-out policy | Ends handled at 300 ms | Cuts on pauses from 300 to under 600 ms | Cuts on all internal pauses | Wait at segment ends, p95 |
|---|---:|---:|---:|---:|
| Fixed 300 ms | 53/53 | 125/125 | 243/243 | 300 ms |
| Fixed 600 ms | 0/53 | 0/125 | 118/243 | 600 ms |
| Acoustic classifier | 0/53 | 0/125 | 118/243 | 600 ms |

The trained head provided zero latency gain at its calibrated operating
point, so it stays outside the live voice path. The 118 long-pause cuts at
600 ms were also present with the classifier. Eligibility rules and three test
meetings limit generalization. Test meeting ES2012a has internal pauses only
after those rules are applied. The short-pause cut calculation is independent
of the live pilot's false interruption target.

## Reproduce

Use Python 3.11 or later with CPU PyTorch and NumPy. From the repository root:

```bash
mkdir -p ami-endpoint
curl -fL https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip \
  -o ami-endpoint/manual.zip
for meeting in ES2002a ES2003a ES2005a ES2006a ES2007a ES2008a ES2009a ES2010a ES2011a ES2012a IS1000a IS1001a
do
  stem="$meeting.Headset-0.wav"
  curl -fL "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$meeting/audio/$stem" \
    -o "ami-endpoint/$stem"
done
python Weaver-Soma/U3/benchmark_acoustic_endpoint.py --ami-dir ami-endpoint --run-dir runs/acoustic
python -m unittest discover -s Weaver-Soma/U3 -q
```

The run directory contains weights and threshold, source SHA-256 hashes,
scores per meeting, and a JSON Lines trace. Each trace row gives the
headset sample index, quiet length, label, logit, selected wait, and
contributions. Adding the model bias to the contributions reconstructs the
logit. Two independent CPU reruns produced byte-identical output files.

The design uses [Voice Activity Projection](https://doi.org/10.48550/arXiv.2205.09812)
as a reference for the plan's future-activity training comparison.
[Full-Duplex-Bench](https://doi.org/10.1109/ASRU65441.2025.11433838) informed
scoring pause handling separately from audible interruption and playback.
The planned device pilot still needs consented recordings and physical
microphone-to-speaker timing, including echo and true interruption controls.
