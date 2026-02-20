# Qwen3-TTS OpenClaw Skill — Complete Implementation Plan

> **Author:** [daMustermann](https://github.com/daMustermann) · **Version:** 1.0 · **License:** MIT · **ClawHub:** `daMustermann/qwen3-tts`

## Overview

This document describes a **complete OpenClaw skill** that gives any OpenClaw agent the ability to **generate human-quality speech** using [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base). The skill auto-detects the user's GPU hardware (NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU-only), creates an isolated Python virtual environment, installs all dependencies, and exposes a rich set of TTS capabilities: basic synthesis, voice design, voice cloning, audio file management, and direct delivery of voice messages via Telegram and WhatsApp.

All designed and cloned voices are **persisted with a user-chosen name** so they can be reused on-the-fly in any future request (e.g. *"say this with the voice Angie"*).

---

## 1. Skill Directory Layout

```
~/clawd/skills/qwen3-tts/
├── SKILL.md                      # Main skill definition (YAML frontmatter + instructions)
├── README.md                     # GitHub + ClawHub README
├── LICENSE                       # MIT License
├── .gitignore                    # Ignore .venv, models, output, config secrets
├── scripts/
│   ├── detect_gpu.sh             # Detects GPU vendor & prints accelerator type
│   ├── setup_env.sh              # Creates venv, installs PyTorch + qwen-tts + deps
│   ├── start_server.sh           # Launches the FastAPI TTS server
│   ├── stop_server.sh            # Gracefully stops the server
│   ├── health_check.sh           # Checks if the server is alive
│   └── convert_to_ogg_opus.sh    # Converts any audio to OGG/Opus via ffmpeg
├── server/
│   ├── tts_server.py             # FastAPI server (OpenAI-compatible /v1/audio/speech)
│   ├── voice_manager.py          # Voice profile CRUD (save/load/list/delete/rename)
│   ├── voice_cloner.py           # Voice cloning pipeline
│   ├── voice_designer.py         # Natural-language voice design pipeline
│   ├── audio_converter.py        # WAV → OGG/Opus, WAV → MP3, etc.
│   └── messaging/
│       ├── telegram_sender.py    # Send OGG voice message via Telegram Bot API
│       └── whatsapp_sender.py    # Send OGG voice message via WhatsApp Business API
├── voices/                       # ★ Persisted named voice profiles (embeddings + metadata)
│   └── .gitkeep                  #   Each voice = <name>.json + <name>.pt
├── output/                       # Generated audio files (transient)
│   └── .gitkeep
├── models/                       # Cached model weights (auto-downloaded, git-ignored)
│   └── .gitkeep
└── requirements/
    ├── base.txt                  # Core Python deps (fastapi, uvicorn, soundfile, etc.)
    ├── cuda.txt                  # NVIDIA-specific (torch+cu128, etc.)
    ├── rocm.txt                  # AMD-specific (torch+rocm6.3, etc.)
    ├── intel.txt                 # Intel-specific (torch for XPU, intel-extension-for-pytorch)
    └── cpu.txt                   # CPU-only PyTorch
```

---

## 2. SKILL.md — Skill Definition

The `SKILL.md` is the file OpenClaw reads to understand the skill. It contains YAML frontmatter for metadata and markdown instructions for the agent.

### 2.1 YAML Frontmatter

```yaml
---
name: qwen3-tts
description: >
  High-quality text-to-speech using Qwen3-TTS. Supports voice cloning (3s of
  audio), natural-language voice design, 10+ languages, persistent named voices,
  and delivering audio via Telegram/WhatsApp as native voice messages.
  Auto-detects GPU hardware (CUDA, ROCm, Intel XPU, CPU).
version: "1.0"
author: daMustermann
repository: https://github.com/daMustermann/qwen3-tts
license: MIT
requires:
  - python>=3.10
  - ffmpeg
  - git
tags:
  - tts
  - audio
  - voice
  - speech
  - voice-cloning
  - voice-design
  - telegram
  - whatsapp
  - clawhub
---
```

### 2.2 Instruction Body (Markdown)

The instruction body tells OpenClaw **when** and **how** to use the skill. It covers:

1. **First-time setup** — detect hardware → create venv → install → download models
2. **Starting/stopping** the TTS server
3. **Generating speech** from text
4. **Designing a new voice** from a natural-language description
5. **Cloning a voice** from a reference audio clip
6. **Managing voices** (list, save, delete, export)
7. **Converting and exporting audio** (WAV, MP3, OGG/Opus, FLAC)
8. **Sending audio to the user** (file attachment, Telegram PTT, WhatsApp PTT)

Full structure detailed below in Section 5.

---

## 3. Hardware Detection & Environment Setup

### 3.1 `detect_gpu.sh`

This script runs on the user's Linux machine and outputs exactly one of: `cuda`, `rocm`, `intel`, `cpu`.

**Detection Logic:**

```
1.  Check if `nvidia-smi` exists and returns GPU info   → "cuda"
2.  Check if `/opt/rocm/bin/rocminfo` or `rocminfo` exists
    and reports an AMD GPU agent                        → "rocm"
3.  Check if `xpu-smi` or `intel_gpu_top` exists,
    or `/dev/dri/renderD*` + `lspci | grep -i intel`
    reports an Intel discrete/integrated GPU             → "intel"
4.  Fallback                                            → "cpu"
```

### 3.2 `setup_env.sh`

1. **Check system dependencies** — ensure `python3`, `pip`, `ffmpeg`, `git` are installed. If not, print clear instructions for the distro (CachyOS/Arch: `pacman -S`, Fedora: `dnf install`, Ubuntu/Debian: `apt install`).
2. **Run `detect_gpu.sh`** to determine the accelerator type.
3. **Create a Python venv** at `~/clawd/skills/qwen3-tts/.venv/`.
4. **Install PyTorch** with the correct index URL:

   | Accelerator | pip install command |
   |-------------|---------------------|
   | `cuda`      | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` |
   | `rocm`      | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.3` |
   | `intel`     | `pip install torch torchaudio intel-extension-for-pytorch --index-url https://download.pytorch.org/whl/xpu` |
   | `cpu`       | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu` |

5. **Install base requirements** — `pip install -r requirements/base.txt`
6. **Install `qwen-tts`** library — `pip install qwen-tts`
7. **Pre-download models** (optional, can be deferred to first use):
   - `Qwen/Qwen3-TTS-12Hz-0.6B-Base` — lightweight, fast model
   - `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — high-quality model
   - `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` — for voice design
   - `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` — for voice cloning
   - `Qwen/Qwen3-Voice-Embedding-12Hz-0.6B` — speaker embedding encoder

### 3.3 `requirements/base.txt`

```
qwen-tts
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
soundfile>=0.13.0
numpy
scipy
pydub
python-multipart
httpx
aiofiles
python-telegram-bot>=21.0
pydantic>=2.0
huggingface-hub[hf_xet]
```

### 3.4 CachyOS / Arch-Specific Notes

CachyOS is Arch-based, so system packages come from `pacman` or `yay`:

```bash
# System deps
sudo pacman -S python python-pip ffmpeg git base-devel

# For CUDA (CachyOS has nvidia in repos)
sudo pacman -S nvidia nvidia-utils cuda cudnn

# For ROCm
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk

# For Intel (ARC GPUs)
# Install intel-compute-runtime and level-zero from AUR or repos
yay -S intel-compute-runtime level-zero-loader
```

Other distros follow their own package manager conventions — the setup script will detect the distro using `/etc/os-release` and advise accordingly.

---

## 4. TTS Server Architecture

### 4.1 `tts_server.py` — FastAPI Server

The server provides an **OpenAI-compatible** API at `http://localhost:8880`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Generate speech (supports saved voice names) |
| `/v1/audio/voice-design` | POST | Create speech with a natural-language voice description |
| `/v1/audio/voice-clone` | POST | Clone a voice from reference audio + text |
| `/v1/voices` | GET | List all saved voice profiles |
| `/v1/voices` | POST | Save/persist a voice profile (name + embedding) |
| `/v1/voices/{name}` | GET | Get a specific voice profile by name |
| `/v1/voices/{name}` | PATCH | Rename or update a voice profile |
| `/v1/voices/{name}` | DELETE | Delete a voice profile |
| `/v1/audio/convert` | POST | Convert audio between formats (WAV/MP3/OGG/FLAC) |
| `/v1/audio/send/telegram` | POST | Send generated audio as Telegram PTT voice message |
| `/v1/audio/send/whatsapp` | POST | Send generated audio as WhatsApp PTT voice message |
| `/health` | GET | Health check |

### 4.2 Model Loading Strategy

```python
# Lazy-load models on first request to save memory
# The agent can specify which model to use per request:

MODELS = {
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "base-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voice-embed": "Qwen/Qwen3-Voice-Embedding-12Hz-0.6B",
}

# On CPU-only or low-VRAM systems, default to the 0.6B model
# On systems with ≥12GB VRAM, default to the 1.7B model
```

### 4.3 Device Detection at Runtime

```python
import torch

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    elif hasattr(torch, "hip") and torch.hip.is_available():
        return "cuda"  # ROCm uses CUDA API via HIP
    else:
        return "cpu"
```

### 4.4 Audio Generation Flow

```
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌─────────────┐
│  Text Input  │────▶│  Qwen3-TTS    │────▶│  WAV Output  │────▶│  Post-proc  │
│  + Voice     │     │  Model        │     │  (24kHz)     │     │  (convert)  │
│  Settings    │     │  Inference    │     │              │     │             │
└──────────────┘     └───────────────┘     └──────────────┘     └─────┬───────┘
                                                                       │
                                            ┌──────────────────────────┤
                                            ▼                          ▼
                                     ┌─────────────┐          ┌──────────────┐
                                     │  Save File  │          │  Send via    │
                                     │  (WAV/MP3/  │          │  Telegram /  │
                                     │   OGG/FLAC) │          │  WhatsApp    │
                                     └─────────────┘          └──────────────┘
```

---

## 5. Skill Capabilities (SKILL.md Instruction Details)

### 5.1 First-Time Setup

When the agent detects the skill has not been set up (no `.venv/` directory exists), it should:

```
1. Run: bash ~/clawd/skills/qwen3-tts/scripts/setup_env.sh
2. Wait for completion (may take 5-15 minutes depending on download speed)
3. Verify: bash ~/clawd/skills/qwen3-tts/scripts/health_check.sh
4. Report success/failure to the user
```

### 5.2 Basic Text-to-Speech

**When to use:** User asks to speak text, read something aloud, generate audio, create a voiceover, narrate something.

**How to use:**
```bash
# Ensure server is running
bash ~/clawd/skills/qwen3-tts/scripts/start_server.sh

# Generate speech via API
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base-1.7b",
    "input": "Hello, this is a test of the Qwen3 text-to-speech system.",
    "voice": "default",
    "language": "en",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output ~/clawd/skills/qwen3-tts/output/speech.wav
```

**Supported languages:** `en`, `zh`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it`

**Supported formats:** `wav`, `mp3`, `ogg`, `flac`

### 5.3 Voice Design (Create New Voices)

**When to use:** User wants to create a custom voice, describe a voice character, design a persona's voice.

**How to use:**
```bash
curl -X POST http://localhost:8880/v1/audio/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voice-design",
    "input": "Welcome to our podcast! Today we discuss AI.",
    "voice_description": "A warm, deep male voice with a slight British accent, calm and authoritative, like a BBC news presenter in his 40s",
    "language": "en",
    "response_format": "wav"
  }' \
  --output ~/clawd/skills/qwen3-tts/output/designed_voice.wav
```

> [!IMPORTANT]
> **After every voice-design or voice-clone request, the agent MUST ask the user:**
> 1. *"Would you like to save this voice for future use?"*
> 2. If yes: *"What name would you like to give this voice?"*
> 3. Save with the chosen name via `POST /v1/voices` (see §5.5)
>
> This ensures every voice the user likes is persisted and reusable by name.

### 5.4 Voice Cloning

**When to use:** User provides a reference audio clip and wants to generate new speech in that voice.

**How to use:**
```bash
# Upload reference audio + generate clone
curl -X POST http://localhost:8880/v1/audio/voice-clone \
  -F "reference_audio=@/path/to/reference.wav" \
  -F "reference_text=This is the transcript of the reference audio." \
  -F "input=Now say this new sentence in the cloned voice." \
  -F "language=en" \
  -F "response_format=wav" \
  --output ~/clawd/skills/qwen3-tts/output/cloned_speech.wav
```

> After cloning, the agent asks the user to name and save the voice (same flow as §5.3).

- Minimum 3 seconds of reference audio
- Recommended 10-30 seconds for best quality
- Accurate transcription of reference audio improves results
- Supports cross-language cloning (clone from English, speak in Japanese)

### 5.5 Voice Persistence & Management

Voices are the **core reusable asset** of this skill. Every designed or cloned voice can be saved with a human-friendly name chosen by the user. Once saved, the user can reference it by name in any future request.

#### 5.5.1 Saving a Voice

After generating audio via voice-design or voice-clone, the server returns a temporary `voice_id` in the response headers. The agent saves it:

```bash
# Save the voice from the last generation with a user-chosen name
curl -X POST http://localhost:8880/v1/voices \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Angie",
    "source_voice_id": "<temporary voice_id from last generation>",
    "description": "Warm, deep male voice with British accent",
    "tags": ["male", "british", "calm"]
  }'
```

This persists two files in `~/clawd/skills/qwen3-tts/voices/`:
- `Angie.json` — metadata (name, description, source, timestamps, tags)
- `Angie.pt` — PyTorch speaker embedding tensor

#### 5.5.2 Using a Saved Voice

Once saved, the user can say *"say this with voice Angie"* and the agent passes the name:

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base-1.7b",
    "input": "Hello, this is Angie speaking!",
    "voice": "Angie",
    "language": "en",
    "response_format": "wav"
  }' \
  --output ~/clawd/skills/qwen3-tts/output/angie_speech.wav
```

The server loads `Angie.pt` and conditions the TTS model on that embedding.

#### 5.5.3 Listing, Inspecting, Renaming & Deleting Voices

```bash
# List all saved voices (returns name, description, source, created_at)
curl http://localhost:8880/v1/voices

# Get details for a specific voice
curl http://localhost:8880/v1/voices/Angie

# Rename a voice
curl -X PATCH http://localhost:8880/v1/voices/Angie \
  -H "Content-Type: application/json" \
  -d '{"name": "Angela"}'

# Delete a voice
curl -X DELETE http://localhost:8880/v1/voices/Angie
```

#### 5.5.4 Voice Profile Format

Each voice is stored as a JSON file in `~/clawd/skills/qwen3-tts/voices/`:
```json
{
  "name": "Angie",
  "description": "Warm, deep male voice with slight British accent, calm and authoritative",
  "created_at": "2026-02-20T19:00:00Z",
  "updated_at": "2026-02-20T19:00:00Z",
  "source": "voice-design",
  "source_description": "A warm, deep male voice with a slight British accent, like a BBC presenter",
  "language": "en",
  "tags": ["male", "british", "calm", "authoritative"],
  "embedding_file": "Angie.pt",
  "sample_audio": "Angie_sample.wav",
  "usage_count": 12
}
```

> [!TIP]
> The `usage_count` tracks how often the voice has been used. The agent can suggest frequently used voices to the user.

#### 5.5.5 Agent Behavior Rules for Voice Persistence

The SKILL.md instructs the agent to follow these rules:

1. **After voice-design or voice-clone:** Always ask *"Would you like to save this voice? What should I call it?"*
2. **When user requests TTS with a name:** Check saved voices first. If `"voice": "Angie"` exists → use it. If not → tell the user and offer to design/clone one.
3. **When user says "list my voices":** Call `GET /v1/voices` and present a formatted list.
4. **When user says "delete voice X":** Confirm with user, then call `DELETE /v1/voices/X`.
5. **When user says "rename voice X to Y":** Call `PATCH /v1/voices/X` with new name.
6. **Voice names are case-insensitive** but stored in the casing the user provided.
7. **No duplicate names allowed.** If a name already exists, ask the user for a different name or confirm overwrite.

### 5.6 Audio Format Conversion

```bash
# Convert WAV to OGG/Opus (for messaging)
curl -X POST http://localhost:8880/v1/audio/convert \
  -F "audio=@~/clawd/skills/qwen3-tts/output/speech.wav" \
  -F "target_format=ogg" \
  --output ~/clawd/skills/qwen3-tts/output/speech.ogg

# Or use the shell script directly
bash ~/clawd/skills/qwen3-tts/scripts/convert_to_ogg_opus.sh input.wav output.ogg
```

### 5.7 Sending Audio via Telegram

**When to use:** User is interacting via Telegram, or explicitly asks to send audio to a Telegram chat.

**Requirements:** A Telegram Bot Token and Chat ID must be configured.

```bash
# Send as PTT voice message
curl -X POST http://localhost:8880/v1/audio/send/telegram \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "~/clawd/skills/qwen3-tts/output/speech.wav",
    "chat_id": "123456789",
    "bot_token": "BOT_TOKEN_HERE",
    "caption": "Here is your audio message"
  }'
```

**Behind the scenes:**
1. Convert WAV → OGG/Opus using ffmpeg: `ffmpeg -i input.wav -c:a libopus -b:a 64k output.ogg`
2. Send via Telegram Bot API `sendVoice` method (not `sendAudio` — this is critical for PTT display)
3. Telegram displays it as a native waveform voice message, playable inline

### 5.8 Sending Audio via WhatsApp

**When to use:** User is interacting via WhatsApp, or explicitly asks to send audio to WhatsApp.

**Requirements:** WhatsApp Business API access (phone number ID + access token).

```bash
# Send as PTT voice message
curl -X POST http://localhost:8880/v1/audio/send/whatsapp \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "~/clawd/skills/qwen3-tts/output/speech.wav",
    "phone_number_id": "PHONE_NUMBER_ID",
    "recipient": "+14155551234",
    "access_token": "ACCESS_TOKEN_HERE"
  }'
```

**Behind the scenes:**
1. Convert WAV → OGG/Opus
2. Upload media to WhatsApp Business API (`POST /<phone_number_id>/media`)
3. Send audio message with `type: audio` and the media ID
4. WhatsApp displays it as a native voice message (PTT)

---

## 6. Configuration

The skill stores its configuration in `~/clawd/skills/qwen3-tts/config.json`:

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8880,
    "workers": 1
  },
  "models": {
    "default_model": "base-1.7b",
    "cache_dir": "~/clawd/skills/qwen3-tts/models/",
    "auto_download": true,
    "device": "auto"
  },
  "audio": {
    "sample_rate": 24000,
    "default_format": "wav",
    "output_dir": "~/clawd/skills/qwen3-tts/output/"
  },
  "telegram": {
    "bot_token": "",
    "default_chat_id": ""
  },
  "whatsapp": {
    "phone_number_id": "",
    "access_token": ""
  },
  "hardware": {
    "detected": "auto",
    "override": null
  }
}
```

The agent can update these values when the user provides API keys or changes preferences.

---

## 7. Lifecycle Management

### 7.1 Starting the Server

```bash
#!/bin/bash
# scripts/start_server.sh
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$SKILL_DIR/.venv"
PIDFILE="$SKILL_DIR/.server.pid"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Server already running (PID $(cat "$PIDFILE"))"
    exit 0
fi

source "$VENV/bin/activate"
cd "$SKILL_DIR/server"
nohup python -m uvicorn tts_server:app \
    --host 127.0.0.1 --port 8880 \
    --log-level info \
    > "$SKILL_DIR/server.log" 2>&1 &
echo $! > "$PIDFILE"
echo "Server started (PID $!)"
```

### 7.2 Stopping the Server

```bash
#!/bin/bash
# scripts/stop_server.sh
SKILL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE="$SKILL_DIR/.server.pid"

if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null
    rm "$PIDFILE"
    echo "Server stopped"
else
    echo "No server running"
fi
```

### 7.3 Health Check

```bash
#!/bin/bash
# scripts/health_check.sh
response=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8880/health)
if [ "$response" = "200" ]; then
    echo "OK"
    exit 0
else
    echo "UNHEALTHY (HTTP $response)"
    exit 1
fi
```

---

## 8. Security Considerations

| Concern | Mitigation |
|---------|------------|
| API tokens in config | Config file has `600` permissions; tokens never logged |
| Server exposure | Binds to `127.0.0.1` only — no external access |
| Voice cloning abuse | Log all clone requests; warn user about consent |
| Model downloads | Use official HuggingFace repos only; verify checksums |
| Disk space | Warn if < 10GB free before downloading models |

---

## 9. Error Handling & Troubleshooting

### Common Issues

| Issue | Detection | Resolution |
|-------|-----------|------------|
| No GPU detected | `detect_gpu.sh` returns `cpu` | Install GPU drivers, or accept CPU mode (slower) |
| CUDA version mismatch | PyTorch import fails | Reinstall with correct CUDA index URL |
| ROCm not found | `rocminfo` missing | Install `rocm-hip-sdk` package |
| Out of VRAM | OOM error during inference | Switch to 0.6B model, reduce batch size, or use CPU offload |
| ffmpeg not installed | Audio conversion fails | `sudo pacman -S ffmpeg` (CachyOS/Arch) |
| Model download fails | Network timeout | Retry with `huggingface-cli download` |
| Telegram send fails | 401/403 from API | Verify bot token and chat ID |
| WhatsApp send fails | 400 from API | Verify phone_number_id and access_token |
| Port 8880 in use | Server won't start | Change port in `config.json` |

---

## 10. Usage Examples in Workflows

### Example 1: Simple TTS in a Workflow

```markdown
# Workflow: Daily News Briefing
1. Fetch today's news headlines
2. Summarize into a 2-minute script
3. Use qwen3-tts to generate audio:
   - Voice: "bbc_presenter" (designed voice)
   - Language: en
   - Format: ogg
4. Send via Telegram as voice message
```

### Example 2: Voice Cloning for Personalization

```markdown
# Workflow: Clone User's Voice
1. Ask user to record a 10-second voice sample
2. Transcribe the sample using whisper
3. Use qwen3-tts voice-clone to create a voice profile "my_voice"
4. From now on, all TTS uses "my_voice"
```

### Example 3: Multi-language Translation + Speech

```markdown
# Workflow: Translate and Speak
1. User provides text in English
2. Translate to German using LLM
3. Use qwen3-tts with language=de to generate German audio
4. Send audio file to user
```

### Example 4: Character Voice Design for Storytelling

```markdown
# Workflow: Audiobook with Multiple Characters
1. User provides a story with dialogue
2. Design voices for each character:
   - "narrator": "A warm, mature female voice, storyteller style"
   - "villain": "A raspy, menacing male voice with a slow pace"
   - "hero": "A young, energetic male voice, confident"
3. Generate audio for each paragraph/dialogue using the appropriate voice
4. Concatenate all audio segments into one file
5. Export as MP3
```

---

## 11. Performance Expectations

| Hardware | Model | Latency (first token) | Real-time Factor | VRAM Usage |
|----------|-------|-----------------------|-------------------|------------|
| NVIDIA RTX 4090 | 1.7B | ~97ms | ~0.1x (10x faster than real-time) | ~5 GB |
| NVIDIA RTX 3060 | 0.6B | ~150ms | ~0.3x | ~2.5 GB |
| AMD RX 7900 XTX (ROCm) | 1.7B | ~130ms | ~0.2x | ~5 GB |
| Intel ARC A770 (XPU) | 0.6B | ~200ms | ~0.5x | ~3 GB |
| CPU (Ryzen 9 7950X) | 0.6B | ~800ms | ~2.0x | ~4 GB RAM |

> [!NOTE]
> These are approximate benchmarks. Actual performance depends on system load, model quantization, and text length.

---

## 12. Supported Qwen3-TTS Models

| Model ID | Size | Use Case | Required VRAM |
|----------|------|----------|---------------|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | Fast general TTS | ~2 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | High-quality general TTS | ~5 GB |
| `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 1.7B | Voice design from description | ~5 GB |
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | Voice cloning | ~2 GB |
| `Qwen/Qwen3-Voice-Embedding-12Hz-0.6B` | 0.6B | Speaker embedding extraction | ~1 GB |

---

## 13. Distro Compatibility Matrix

| Distribution | Package Manager | GPU Support | Notes |
|-------------|-----------------|-------------|-------|
| **CachyOS** | pacman/yay | CUDA, ROCm, Intel | Primary target. Arch-based, rolling release. |
| **Arch Linux** | pacman/yay | CUDA, ROCm, Intel | Same as CachyOS |
| **Ubuntu 22.04+** | apt | CUDA, ROCm, Intel | Most tested PyTorch platform |
| **Fedora 39+** | dnf | CUDA, ROCm, Intel | RPM Fusion for NVIDIA drivers |
| **openSUSE Tumbleweed** | zypper | CUDA, ROCm | Rolling release, good support |
| **Debian 12+** | apt | CUDA, ROCm | Stable, may need backports |
| **NixOS** | nix | CUDA, ROCm | Needs flake/overlay for PyTorch |

---

## 14. GitHub & ClawHub Distribution

This skill is designed to be **shared on GitHub and ClawHub** as a standalone installable skill.

### 14.1 GitHub Repository Structure

```
github.com/daMustermann/qwen3-tts
├── SKILL.md            # OpenClaw reads this
├── README.md           # Human-readable docs, badges, screenshots
├── LICENSE             # MIT
├── .gitignore          # See below
├── scripts/            # Setup & lifecycle scripts
├── server/             # FastAPI server code
├── voices/.gitkeep     # Empty dir, user voices are local
├── output/.gitkeep     # Empty dir
├── models/.gitkeep     # Empty dir, models are downloaded
└── requirements/       # Per-accelerator requirements
```

### 14.2 `.gitignore`

```gitignore
# Python
.venv/
__pycache__/
*.pyc
*.pyo

# Models (large, downloaded at install time)
models/*.bin
models/*.safetensors
models/*.pt
models/Qwen*/

# User data (private per install)
voices/*.pt
voices/*.json
!voices/.gitkeep
output/*
!output/.gitkeep
config.json

# Runtime
.server.pid
server.log
*.log
```

### 14.3 ClawHub Installation

Users install the skill from ClawHub with:
```bash
claw install daMustermann/qwen3-tts
```
or manually by cloning into their skills directory:
```bash
git clone https://github.com/daMustermann/qwen3-tts.git ~/clawd/skills/qwen3-tts
```

On first use, the agent runs `setup_env.sh` which handles all hardware detection and dependency installation automatically.

### 14.4 README.md Contents

The README should include:
- **Badges:** ClawHub version, license, Python version, GPU support
- **Quick start:** one-liner install + first TTS command
- **Features list** with screenshots/audio samples
- **Hardware requirements** table
- **Voice persistence** explanation (save/reuse by name)
- **Telegram/WhatsApp setup** guide
- **Contributing** guidelines
- **Credits:** Qwen team (Alibaba Cloud) for the model

---

## 15. Implementation Phases

### Phase 1: Core Infrastructure
- [x] Plan & design (this document)
- [ ] Create directory structure & `.gitignore`
- [ ] Write `detect_gpu.sh`
- [ ] Write `setup_env.sh`
- [ ] Write `requirements/*.txt` files

### Phase 2: TTS Server + Voice Persistence
- [ ] Implement `tts_server.py` (FastAPI + OpenAI-compatible endpoints)
- [ ] Implement `voice_manager.py` (full CRUD: save/load/list/rename/delete + embedding persistence)
- [ ] Implement `voice_designer.py` (natural-language voice creation)
- [ ] Implement `voice_cloner.py` (reference audio → voice profile)
- [ ] Implement `audio_converter.py` (format conversions via ffmpeg/pydub)

### Phase 3: Messaging Integration
- [ ] Implement `telegram_sender.py`
- [ ] Implement `whatsapp_sender.py`
- [ ] Write `convert_to_ogg_opus.sh`

### Phase 4: Skill Definition & Distribution
- [ ] Write complete `SKILL.md` with voice save prompting rules
- [ ] Write `config.json` template
- [ ] Write lifecycle scripts (start/stop/health)
- [ ] Write `README.md` for GitHub/ClawHub
- [ ] Create `LICENSE` (MIT)
- [ ] Create `.gitignore`

### Phase 5: Testing & Validation
- [ ] Test on CachyOS with NVIDIA GPU
- [ ] Test on CachyOS with AMD GPU (ROCm)
- [ ] Test on CachyOS with CPU-only
- [ ] Test voice design → save → reuse by name
- [ ] Test voice cloning → save → reuse by name
- [ ] Test voice listing, renaming, deleting
- [ ] Test Telegram voice message delivery
- [ ] Test WhatsApp voice message delivery
- [ ] Test cross-language synthesis

---

## 16. Summary

This skill transforms OpenClaw into a **full-featured TTS studio**:

- 🎤 **Generate speech** in 10+ languages with human-quality voices
- 🎨 **Design voices** using natural language (*"a raspy pirate voice"*)
- 🔄 **Clone any voice** from just 3 seconds of audio
- 💾 **Persist voices by name** — save as "Angie", reuse forever with `"voice": "Angie"`
- 🗣️ **Agent asks to save** — after every design/clone, the agent prompts to name and save
- 📱 **Send voice messages** natively via Telegram and WhatsApp PTT
- 🖥️ **Auto-detect hardware** — works on CUDA, ROCm, Intel XPU, or CPU
- 🐧 **Linux-native** — CachyOS/Arch first, all major distros supported
- 🔌 **OpenAI-compatible API** — drop-in replacement, works with any OpenAI TTS client
- 📦 **Shareable** — publish on GitHub & ClawHub as `daMustermann/qwen3-tts`
