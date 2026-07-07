#!/bin/bash
# One-time setup of the Weaver TTS stack on the weaver box.
#
# Default: lightweight AWS Polly provider (realistic AWS voice, no OpenAI).
#
# Optional external-code path: set TTS_INSTALL_OPENVOICE=1 to install:
#   * OpenVoice v2   (github.com/myshell-ai/OpenVoice, MIT)
#   * MeloTTS        (github.com/myshell-ai/MeloTTS, MIT)
#   * model weights  (huggingface.co/myshell-ai/OpenVoiceV2)
# plus torch/fastapi/uvicorn from PyPI. ~3.5GB disk. Review before running.
#
# Prereq: this repo's deploy/tts/tts_server.py copied to ~/tts/tts_server.py.
# OpenVoice fallback also needs ~/tts/weaver_voice_ref.wav.
#
# Run:  bash setup_tts.sh 2>&1 | tee ~/tts/install.log
set -euxo pipefail
cd ~/tts
INSTALL_OPENVOICE="${TTS_INSTALL_OPENVOICE:-0}"

sudo apt-get update -q && sudo apt-get install -y -q software-properties-common build-essential libsndfile1 ffmpeg

if [ "$INSTALL_OPENVOICE" = "1" ]; then
    # OpenVoice pins numpy==1.22.0 (and similarly old friends) which have no
    # py3.12 wheels and don't compile there — the stack targets py3.10. Give it
    # its own 3.10 from deadsnakes instead of fighting the pins.
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get install -y -q python3.10 python3.10-venv python3.10-dev
    python3.10 -m venv venv
else
    sudo apt-get install -y -q python3-venv python3-dev
    python3 -m venv venv
fi

./venv/bin/pip install -U pip wheel setuptools
./venv/bin/pip install fastapi uvicorn soundfile boto3 "huggingface_hub[cli]"

if [ "$INSTALL_OPENVOICE" = "1" ]; then
    ./venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    # MeloTTS first with full deps; then OpenVoice WITHOUT deps — its requirements
    # drag in faster-whisper -> PyAV (no aarch64 wheel for the old pin, won't build
    # against noble's ffmpeg 6). We never use that path: clone-once extracts the
    # embedding via ToneColorConverter.extract_se(), no whisper/VAD needed.
    ./venv/bin/pip install git+https://github.com/myshell-ai/MeloTTS.git
    ./venv/bin/pip install --no-deps git+https://github.com/myshell-ai/OpenVoice.git
    ./venv/bin/pip install pydub wavmark
    ./venv/bin/python -m unidic download || true     # MeloTTS imports its ja stack even for EN
    ./venv/bin/huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ~/tts/OpenVoiceV2
    ./venv/bin/python - <<'PYEOF'
import nltk
for pkg in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]:
    try: nltk.download(pkg)
    except Exception as e: print("nltk", pkg, e)
PYEOF
else
    echo "Polly-only install complete. Set TTS_INSTALL_OPENVOICE=1 to add local OpenVoice fallback."
fi

# install + start the service (unit file comes from the repo's deploy/tts/)
sudo cp ~/tts/weaver-tts.service /etc/systemd/system/weaver-tts.service
sudo systemctl daemon-reload
sudo systemctl enable --now weaver-tts
sleep 5
systemctl is-active weaver-tts
echo INSTALL_DONE
