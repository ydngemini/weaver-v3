#!/bin/bash
# One-time setup of the cloned-voice TTS stack on the weaver box.
#
# EXTERNAL CODE NOTICE — this installs and executes, on the box:
#   * OpenVoice v2   (github.com/myshell-ai/OpenVoice, MIT)
#   * MeloTTS        (github.com/myshell-ai/MeloTTS, MIT)
#   * model weights  (huggingface.co/myshell-ai/OpenVoiceV2)
# plus torch/fastapi/uvicorn from PyPI. ~3.5GB disk. Review before running.
#
# Prereqs already on the box: ~/tts/weaver_voice_ref.wav (the reference sample)
# and this repo's deploy/tts/tts_server.py copied to ~/tts/tts_server.py.
#
# Run:  bash setup_tts.sh 2>&1 | tee ~/tts/install.log
set -euxo pipefail
cd ~/tts

sudo apt-get update -q && sudo apt-get install -y -q python3-venv python3-dev build-essential libsndfile1 ffmpeg
python3 -m venv venv
./venv/bin/pip install -U pip wheel setuptools
./venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install git+https://github.com/myshell-ai/OpenVoice.git
./venv/bin/pip install git+https://github.com/myshell-ai/MeloTTS.git
./venv/bin/python -m unidic download || true     # MeloTTS imports its ja stack even for EN
./venv/bin/pip install fastapi uvicorn soundfile "huggingface_hub[cli]"
./venv/bin/huggingface-cli download myshell-ai/OpenVoiceV2 --local-dir ~/tts/OpenVoiceV2
./venv/bin/python - <<'PYEOF'
import nltk
for pkg in ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict"]:
    try: nltk.download(pkg)
    except Exception as e: print("nltk", pkg, e)
PYEOF

# install + start the service (unit file comes from the repo's deploy/tts/)
sudo cp ~/tts/weaver-tts.service /etc/systemd/system/weaver-tts.service
sudo systemctl daemon-reload
sudo systemctl enable --now weaver-tts
sleep 5
systemctl is-active weaver-tts
echo INSTALL_DONE
