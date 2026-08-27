# Weaver v3 — Azure Native Deployment

Replace AWS (Bedrock, Mantle, Polly, S3, Nova Sonic) with Azure-native services
while keeping the same systemd + Docker operational pattern as the current EC2 deployment.

## Architecture

```
Browser ──wss──► Caddy (TLS) ──http──► Weaver Brain API (Azure OpenAI)
                                          │
                                          ├── SLM Experts (Azure OpenAI)
                                          ├── Azure Speech (STT + TTS)
                                          └── Azure Blob Storage (avatars/assets)
```

| Service | AWS (old) | Azure (new) |
|---------|-----------|-------------|
| LLM API | Bedrock / Mantle | Azure OpenAI (`AZURE_OPENAI_KEY`) |
| SLM Backend | Mantle | Azure OpenAI |
| Realtime Voice | Nova Sonic | Azure Speech SDK |
| TTS | AWS Polly | Azure Speech Services |
| Object Storage | S3 | Azure Blob Storage |
| Compute | EC2 | Azure VM |
| Domain | weaverv3.com | weaverv3.com (same DNS) |

## VM Spec

- **SKU:** Standard_E4s_v5 (4 vCPU, 32 GB RAM) — ~$170/mo reserved
- **Region:** eastus
- **OS:** Ubuntu 22.04 LTS
- **Storage:** 64 GB Premium SSD + temp disk (for model cache)
- **Network:** Static public IP, NSG locked to Cloudflare IPs only, HTTPS via Caddy auto-TLS

## Provisioning Steps

### 0. Prerequisites

```bash
# Local machine
az login
terraform -chdir=deploy/azure-terraform init
```

### 1. Deploy Infrastructure

```bash
export TF_VAR_ssh_public_key="$(cat ~/.ssh/id_ed25519.pub)"
terraform -chdir=deploy/azure-terraform apply
```

Outputs the VM public IP, SSH command, and Blob Storage container URL.

### 2. Configure Environment

Copy the example env file and fill in your Azure OpenAI keys:

```bash
cp .env.example .env
# Edit .env with your AZURE_OPENAI_KEY, AZURE_SPEECH_KEY, etc.
```

### 3. Provision VM

```bash
scp -i ~/.ssh/weaver .env deploy/azure-terraform/ azure-user@<PUBLIC_IP>:~
ssh -i ~/.ssh/weaver azure-user@<PUBLIC_IP> 'sudo bash setup_azure.sh'
```

### 4. Upload Assets to Blob Storage

```bash
export AZURE_STORAGE_CONNECTION_STRING="..."
az storage blob upload-batch \
  --destination weaver-assets/avatar \
  --source images/avatar
```

### 5. Verify

```bash
curl https://weaverv3.com/health
# Should show mode: azure, azure_deployment: gpt-4.1-mini, voice: azure
```

## Environment Variables

### Azure OpenAI

```env
WEAVER_LLM_BACKEND=azure
AZURE_OPENAI_KEY=sk-...
AZURE_OPENAI_ENDPOINT=https://neoh.openai.azure.com
AZURE_DEPLOYMENT=gpt-4.1-mini
AZURE_NANO_DEPLOYMENT=gpt-5-mini
```

### Azure Speech

```env
TTS_PROVIDER=azure
AZURE_SPEECH_KEY=...    # falls back to AZURE_OPENAI_KEY
AZURE_SPEECH_REGION=eastus
AZURE_SPEECH_VOICE=en-US-AriaNeural
WEAVER_VOICE_REALTIME_MODE=azure
WEAVER_VOICE_ID=en-US-AriaNeural
```

### Blob Storage

```env
AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;..."
AZURE_STORAGE_CONTAINER_NAME=weaver-assets
```

## Deployed Models (eastus)

| Model | Deployment | TPM | Use |
|-------|-----------|-----|-----|
| GPT-4.1-mini | gpt-4.1-mini | 200K | Primary brain + cortex |
| GPT-5-mini | gpt-5-mini | 500K | Nanomodel (SLM) |
| Phi-4 | phi-4 | 20K | Experimental |
| Llama-3.3-70B | llama-3.3-70b | 20K | Heavy lift |

## Troubleshooting

**Brain API won't start:**
```bash
systemctl status weaver-brain
journalctl -u weaver-brain -n 50 --no-pager
```

**TTS silent:** Check `AZURE_SPEECH_KEY` and that the voice region matches the
endpoint. Azure Speech and Azure OpenAI can use the same key if the AI Services
resource is multi-modal.

**Voice bridge not connecting:** Ensure `WEAVER_VOICE_REALTIME_MODE=azure` and
that the Speech SDK is installed (`pip install azure-cognitiveservices-speech`).
