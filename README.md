# 🎤 Qwen3-TTS — OpenClaw Skill

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ClawHub](https://img.shields.io/badge/ClawHub-daMustermann%2Fqwen3--tts-purple.svg)](https://clawhub.dev/daMustermann/qwen3-tts)

> High-quality text-to-speech for OpenClaw agents. Voice cloning, voice design, 10+ languages, Telegram & WhatsApp voice messages. Auto-detects CUDA, ROCm, Intel XPU, or CPU.

## ⚡ Quick Start

```bash
# Install from ClawHub
claw install daMustermann/qwen3-tts

# Or clone manually
git clone https://github.com/daMustermann/qwen3-tts.git ~/clawd/skills/qwen3-tts
```

On first use, the agent automatically runs setup — detecting your GPU, creating a virtual environment, and installing all dependencies.

## ✨ Features

- **🎤 Generate Speech** — Human-quality TTS in 10+ languages (EN, ZH, JA, KO, DE, FR, RU, PT, ES, IT)
- **🎨 Design Voices** — Create new voices from natural language descriptions (*"a warm British accent, calm and deep"*)
- **🔄 Clone Voices** — Clone any voice from just 3 seconds of audio
- **💾 Persist Voices by Name** — Save as "Angie", reuse forever with `voice: "Angie"`
- **📱 Telegram & WhatsApp PTT** — Send audio as native voice messages
- **🖥️ Auto-Detect Hardware** — NVIDIA CUDA, AMD ROCm, Intel XPU, or CPU-only
- **🔌 OpenAI-Compatible API** — Drop-in replacement at `localhost:8880`

## 🎯 Supported Hardware

| Hardware | Accelerator | Recommended Model |
|----------|-------------|-------------------|
| NVIDIA GPU | CUDA | 1.7B (high quality) |
| AMD GPU | ROCm | 1.7B or 0.6B |
| Intel GPU | XPU | 0.6B (fast) |
| CPU only | — | 0.6B |

## 🗣️ Voice Persistence

Every designed or cloned voice can be saved with a name you choose:

```
User: "Design a voice — a raspy pirate captain"
Agent: [generates audio] "Would you like to save this voice? What should I call it?"
User: "Call it Captain Hook"
Agent: "Voice saved as 'Captain Hook'! You can use it anytime."

User: "Say 'Ahoy mateys!' with voice Captain Hook"
Agent: [generates using saved voice]
```

Voices are stored locally in `voices/` and persist across sessions.

## 📱 Messaging

### Telegram
Configure your bot token in `config.json`, then the agent sends audio as native PTT voice messages using `sendVoice` (OGG/Opus format).

### WhatsApp
Configure your WhatsApp Business API credentials in `config.json`, then audio is sent as native voice messages.

## 🔧 Manual Setup

If you prefer manual setup instead of the agent's automatic first-run:

```bash
cd ~/clawd/skills/qwen3-tts
bash scripts/setup_env.sh
bash scripts/start_server.sh
bash scripts/health_check.sh
```

## 📋 Requirements

- **OS:** Linux (CachyOS, Arch, Ubuntu, Fedora, Debian, openSUSE, NixOS)
- **Python:** 3.10+
- **ffmpeg:** Required for audio format conversion
- **Disk:** ~10 GB for models

## 🤝 Contributing

Contributions welcome! Please open an issue or PR on [GitHub](https://github.com/daMustermann/qwen3-tts).

## 📜 Credits

- **Model:** [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) by the Qwen team (Alibaba Cloud)
- **Author:** [daMustermann](https://github.com/daMustermann)
- **License:** MIT
