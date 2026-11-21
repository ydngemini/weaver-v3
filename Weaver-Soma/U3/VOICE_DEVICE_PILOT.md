# Voice device pilot

The laptop preflight identified Linux Mint 22.2, built-in audio, PipeWire
tools, PortAudio, and FFmpeg with Flite. A check on the laptop's isolated
ALSA Loopback card sent a 3,200-sample generated signal at 16 kHz mono
PCM16 through playback and recovered the exact samples on the paired virtual
capture device. A separate PortAudio duplex probe on that virtual route
recovered a second signal exactly across 70 20 ms callbacks, with usable
stream timestamps and zero reported xruns. Physical microphone latency,
speaker cutoff, and echo cancellation need independent observation.

## Collect one trace per trial

Use consented speakers and an independent room recorder for interruption
timing. Put the recorder where both the speaker and assistant's playback
are distinguishable. If a synchronized two-channel recorder is available,
use one channel close to the speaker and another close to SOMA's output.
Keep the audio route, output level, noise, distance, and echo configuration
fixed within each pair. Record those settings alongside the manifest.
Pair the same script for the same speaker with 300 and 600 ms endpoints,
and alternate which policy runs first. A complete pilot can use ten paired
scripts per speaker across ten speakers, with all four scenarios covered.

| Scenario | What the speaker and assistant do | What to mark |
|---|---|---|
| `true_interruption` | Start speaking during an active assistant reply. | User speech onset and last audible assistant output, judged from the same room recording. Mark the first sample after old playback ceases. Use a null mute mark if the response continues, ends naturally, or the cutoff cannot be distinguished. |
| `within_turn_pause` | Say “what is,” pause, then finish “the status” as the same request. Include pauses on both sides of the controls where practical. | Pause start and resumed speech on the exact WAV captured by SOMA. |
| `echo_only` | Say an initial request, then stay quiet while the assistant plays through the tested speaker route. | Check that playback occurred. Any assistant-response interruption counts as a false interruption for this trial. |
| `clean_control` | Give an uninterrupted request and let the assistant finish on a clean route. | Check that playback occurred. Any assistant-response interruption counts as a false interruption for this trial. |

From the repository root, use a Python environment with CPU PyTorch and
NumPy and the deployed PortAudio library. On the laptop, add the existing
local voice pilot packages inside that environment.

```bash
python -m pip install pocketsphinx==5.0.4 sounddevice==0.5.6
python Weaver-Soma/U3/live_voice.py --devices
python Weaver-Soma/U3/live_voice.py --live --endpoint-ms 300 \
  --input-device INPUT --output-device OUTPUT --trace pilot/p01-pause-300.jsonl \
  --capture-input pilot/p01-pause-300-input.wav
```

Substitute actual device IDs and create the pilot directory before starting.
Use Ctrl+C after the trial completes to close the trace and finalize the WAV.
Use `--capture-input` for pause annotation. Keep input WAVs, room audio, and
trace transcripts with the consented pilot recordings. For a public summary,
share aggregated metrics and the device configuration.

## Score

Create one manifest per hardware and echo configuration. The policy comes
from each trace's header. Trial paths are relative to the manifest directory.
Use a pseudonymous speaker ID, a unique trial ID, and the same pair ID and
scenario for the two policy trials. Sample marks are zero-based integers.

```json
{
  "evidence": "physical",
  "trials": [
    {
      "trial_id": "p01-interrupt-300",
      "speaker_id": "p01",
      "pair_id": "interrupt-1",
      "scenario": "true_interruption",
      "trace": "p01-interrupt-300.jsonl",
      "reference_wav": "p01-interrupt-300-room.wav",
      "speech_onset_ref_sample": 48000,
      "audible_mute_ref_sample": 53760
    },
    {
      "trial_id": "p01-pause-600",
      "speaker_id": "p01",
      "pair_id": "pause-1",
      "scenario": "within_turn_pause",
      "trace": "p01-pause-600.jsonl",
      "input_wav": "p01-pause-600-input.wav",
      "pause_start_input_sample": 12800,
      "resume_input_sample": 19520
    }
  ]
}
```

For echo and clean control trials, supply the trial ID, speaker, pair,
scenario, and trace. The example shows the two sets of sample fields.
Collect the matching control for each example pair in the actual pilot.

```bash
python Weaver-Soma/U3/score_voice_device_pilot.py --manifest pilot/trials.json \
  --out pilot/scorecard.json
python Weaver-Soma/U3/benchmark_live_loopback.py --run-dir runs/pilot-replay
```

The scorer reports p50 and p95 audible cutoff, missed or unmeasured
interruptions, false interruptions, early endpoints during pauses, pairing,
and invalid trials. A missed cutoff counts as an unbounded latency in the
percentile calculation. The false-interruption interval is trial-wise
Wilson and should be read with the repeated-speaker design in mind.
Trial coverage and policy-level thresholds are reported independently.
The deterministic replay checks old-response output after the scheduled mute.

The [PortAudio callback timing specification](https://portaudio.com/docs/v19-doxydocs/structPaStreamCallbackTimeInfo.html)
defines the ADC, DAC, and callback stamps on its own stream clock. Use
those stamps to diagnose scheduling and use the independent room recorder
clock for audible latency. [Full-Duplex-Bench](https://doi.org/10.1109/ASRU65441.2025.11433838)
separates pause handling, turn shifts, and interruption. The pilot keeps
those behaviors in separate trial groups. Match the source and sink
when evaluating a configured [PipeWire echo-cancel route](https://docs.pipewire.org/page_module_echo_cancel.html).
