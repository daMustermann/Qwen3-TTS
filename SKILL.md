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

# Qwen3-TTS Skill

You have access to a powerful text-to-speech system that can generate human-quality speech, design new voices from descriptions, clone existing voices, and send audio via Telegram/WhatsApp.

## First-Time Setup

If the server is not running (health check fails) or the `.venv/` directory does not exist:

```bash
bash ~/clawd/skills/qwen3-tts/scripts/setup_env.sh
```

This auto-detects the GPU (CUDA, ROCm, Intel XPU, or CPU-only), creates a Python venv, and installs all dependencies. It takes 5-15 minutes on first run.

## Starting & Stopping the Server

Before any TTS operation, ensure the server is running:

```bash
# Start
bash ~/clawd/skills/qwen3-tts/scripts/start_server.sh

# Check health
bash ~/clawd/skills/qwen3-tts/scripts/health_check.sh

# Stop (when done)
bash ~/clawd/skills/qwen3-tts/scripts/stop_server.sh
```

The server runs at `http://localhost:8880`.

---

## Capabilities

### 1. Generate Speech from Text

**When to use:** User asks to speak text, read something aloud, generate audio, do a voiceover, narrate.

```bash
curl -X POST http://localhost:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "base-1.7b",
    "input": "TEXT_HERE",
    "voice": "VOICE_NAME_OR_default",
    "language": "en",
    "response_format": "wav"
  }' \
  --output ~/clawd/skills/qwen3-tts/output/speech.wav
```

- **Supported languages:** en, zh, ja, ko, de, fr, ru, pt, es, it
- **Supported formats:** wav, mp3, ogg, flac
- **Voice:** Use `"default"` or any saved voice name (e.g. `"Angie"`)

### 2. Design a New Voice

**When to use:** User wants to create a custom voice, describe how a character sounds.

```bash
curl -X POST http://localhost:8880/v1/audio/voice-design \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voice-design",
    "input": "TEXT_TO_SPEAK",
    "voice_description": "DESCRIBE THE VOICE IN NATURAL LANGUAGE",
    "language": "en",
    "response_format": "wav"
  }' \
  --output ~/clawd/skills/qwen3-tts/output/designed.wav
```

The response includes a `X-Voice-Id` header — use it to save the voice (see §5).

### 3. Clone a Voice

**When to use:** User provides a reference audio clip and wants to use that voice.

```bash
curl -X POST http://localhost:8880/v1/audio/voice-clone \
  -F "reference_audio=@/path/to/reference.wav" \
  -F "reference_text=Transcript of the reference audio" \
  -F "input=New text to speak in cloned voice" \
  -F "language=en" \
  -F "response_format=wav" \
  --output ~/clawd/skills/qwen3-tts/output/cloned.wav
```

- Minimum 3 seconds, recommended 10-30 seconds of reference audio
- Accurate transcription improves quality
- The response includes a `X-Voice-Id` header for saving

### 4. ⭐ CRITICAL: Voice Save Prompting Rules

**YOU MUST FOLLOW THESE RULES:**

1. **After EVERY voice-design or voice-clone request**, ask the user:
   > "Would you like to save this voice for future use? What name should I give it?"

2. **If the user says yes**, save it with their chosen name:
   ```bash
   curl -X POST http://localhost:8880/v1/voices \
     -H "Content-Type: application/json" \
     -d '{
       "name": "USER_CHOSEN_NAME",
       "source_voice_id": "VOICE_ID_FROM_X_VOICE_ID_HEADER",
       "description": "Description of the voice",
       "tags": ["tag1", "tag2"]
     }'
   ```

3. **When user requests TTS with a voice name** (e.g. "say this with Angie"):
   - Use `"voice": "Angie"` in the speech request
   - If the name doesn't exist, tell the user and offer to design or clone one

4. **When user asks to list voices:**
   ```bash
   curl http://localhost:8880/v1/voices
   ```
   Present the results as a formatted list with name, description, and usage count.

5. **When user asks to delete a voice:** Confirm first, then:
   ```bash
   curl -X DELETE http://localhost:8880/v1/voices/VOICE_NAME
   ```

6. **When user asks to rename a voice:**
   ```bash
   curl -X PATCH http://localhost:8880/v1/voices/OLD_NAME \
     -H "Content-Type: application/json" \
     -d '{"name": "NEW_NAME"}'
   ```

7. **Voice names are case-insensitive.** No duplicate names allowed.

### 5. Convert Audio Formats

```bash
curl -X POST http://localhost:8880/v1/audio/convert \
  -F "audio=@input.wav" \
  -F "target_format=mp3" \
  --output output.mp3
```

Supported: wav, mp3, ogg (Opus), flac

### 6. Send via Telegram (PTT Voice Message)

**When to use:** User is on Telegram, or explicitly asks to send audio there.

```bash
curl -X POST http://localhost:8880/v1/audio/send/telegram \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "/path/to/audio.wav",
    "chat_id": "CHAT_ID",
    "bot_token": "BOT_TOKEN",
    "caption": "Optional caption"
  }'
```

Audio is auto-converted to OGG/Opus and sent via `sendVoice` (displays as native PTT).

### 7. Send via WhatsApp (PTT Voice Message)

**When to use:** User is on WhatsApp, or explicitly asks to send audio there.

```bash
curl -X POST http://localhost:8880/v1/audio/send/whatsapp \
  -H "Content-Type: application/json" \
  -d '{
    "audio_file": "/path/to/audio.wav",
    "phone_number_id": "PHONE_ID",
    "recipient": "+14155551234",
    "access_token": "ACCESS_TOKEN"
  }'
```

Audio is auto-converted to OGG/Opus and sent as a native WhatsApp voice message.

---

## How to Respond

After generating speech:
1. Tell the user the audio has been generated
2. Provide the file path
3. If it was voice-design or voice-clone, **ask to save the voice** (Rule §4)
4. If the user is on Telegram/WhatsApp, offer to send it as a voice message

After saving a voice:
- Confirm the name and tell the user they can use it anytime with that name

After sending via Telegram/WhatsApp:
- Confirm successful delivery

## Configuration

The agent can update `~/clawd/skills/qwen3-tts/config.json` to set:
- Telegram bot token and default chat ID
- WhatsApp phone number ID and access token
- Default model (base-0.6b or base-1.7b)
- Default audio format
