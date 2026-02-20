"""
tts_server.py — Qwen3-TTS FastAPI Server (OpenAI-compatible).

Provides a comprehensive TTS API with:
  - OpenAI-compatible /v1/audio/speech endpoint
  - Voice design from natural language descriptions
  - Voice cloning from reference audio
  - Persistent named voice management (save/load/list/rename/delete)
  - Audio format conversion
  - Telegram & WhatsApp PTT voice message delivery

Author: daMustermann · Version: 1.0
"""

import json
import os
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# Add server directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from audio_converter import convert_audio, convert_to_ogg_opus, get_audio_info
from voice_cloner import VoiceCloner
from voice_designer import VoiceDesigner
from voice_manager import VoiceManager

# ─── Configuration ───
SKILL_DIR = Path(__file__).parent.parent
CONFIG_PATH = SKILL_DIR / "config.json"


def load_config() -> dict:
    """Load configuration from config.json."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "models": {
            "default_model": "base-1.7b",
            "cache_dir": str(SKILL_DIR / "models"),
            "auto_download": True,
            "device": "auto",
        },
        "audio": {
            "sample_rate": 24000,
            "default_format": "wav",
            "output_dir": str(SKILL_DIR / "output"),
        },
        "voices": {
            "dir": str(SKILL_DIR / "voices"),
        },
    }


# ─── Model registry ───
MODELS = {
    "base-0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "base-1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice-design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "custom-voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "voice-embed": "Qwen/Qwen3-Voice-Embedding-12Hz-0.6B",
}


# ─── Device detection ───
def get_device(override: Optional[str] = None) -> str:
    if override and override != "auto":
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


# ─── Globals (initialized at startup) ───
config: dict = {}
voice_manager: Optional[VoiceManager] = None
voice_designer: Optional[VoiceDesigner] = None
voice_cloner: Optional[VoiceCloner] = None
tts_model = None
tts_model_id: Optional[str] = None
output_dir: Path = SKILL_DIR / "output"

# Temporary storage for voice embeddings from recent generations
_temp_voice_store: dict[str, torch.Tensor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources at startup."""
    global config, voice_manager, output_dir

    config = load_config()

    # Initialize voice manager
    voices_dir = config.get("voices", {}).get("dir", str(SKILL_DIR / "voices"))
    voice_manager = VoiceManager(voices_dir)

    # Ensure output directory exists
    output_dir = Path(config.get("audio", {}).get("output_dir", str(SKILL_DIR / "output")))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(config.get("models", {}).get("device"))
    print(f"[INFO] Qwen3-TTS server starting on device: {device}")
    print(f"[INFO] Voices directory: {voices_dir}")
    print(f"[INFO] Output directory: {output_dir}")
    print(f"[INFO] Loaded {len(voice_manager.list_voices())} saved voices")

    yield

    # Cleanup
    if voice_designer:
        voice_designer.unload()
    if voice_cloner:
        voice_cloner.unload()
    _temp_voice_store.clear()


app = FastAPI(
    title="Qwen3-TTS OpenClaw Skill",
    description="High-quality TTS with voice design, voice cloning, and named voice persistence.",
    version="1.0",
    lifespan=lifespan,
)


# ═══════════════════════════════════════════════
#  Request/Response Models
# ═══════════════════════════════════════════════

class SpeechRequest(BaseModel):
    model: str = "base-1.7b"
    input: str
    voice: str = "default"
    language: str = "en"
    response_format: str = "wav"
    speed: float = 1.0


class VoiceDesignRequest(BaseModel):
    model: str = "voice-design"
    input: str
    voice_description: str
    language: str = "en"
    response_format: str = "wav"


class VoiceSaveRequest(BaseModel):
    name: str
    source_voice_id: Optional[str] = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    language: str = "en"


class VoiceUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    language: Optional[str] = None


class TelegramSendRequest(BaseModel):
    audio_file: str
    chat_id: str
    bot_token: Optional[str] = None
    caption: Optional[str] = None


class WhatsAppSendRequest(BaseModel):
    audio_file: str
    phone_number_id: Optional[str] = None
    recipient: str
    access_token: Optional[str] = None


# ═══════════════════════════════════════════════
#  Helper Functions
# ═══════════════════════════════════════════════

def _get_tts_model(model_name: str = "base-1.7b"):
    """Lazy-load a TTS base model."""
    global tts_model, tts_model_id

    if tts_model is not None and tts_model_id == model_name:
        return tts_model

    model_hf_id = MODELS.get(model_name)
    if not model_hf_id:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    try:
        from qwen_tts import QwenTTS
        cache_dir = config.get("models", {}).get("cache_dir")
        device = get_device(config.get("models", {}).get("device"))

        tts_model = QwenTTS.from_pretrained(
            model_hf_id,
            cache_dir=cache_dir,
            device=device,
        )
        tts_model_id = model_name
        return tts_model
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="qwen-tts package not installed. Run: pip install qwen-tts",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")


def _get_voice_designer():
    """Lazy-load the voice designer."""
    global voice_designer
    if voice_designer is None:
        cache_dir = config.get("models", {}).get("cache_dir")
        device = get_device(config.get("models", {}).get("device"))
        voice_designer = VoiceDesigner(cache_dir=cache_dir, device=device)
    return voice_designer


def _get_voice_cloner():
    """Lazy-load the voice cloner."""
    global voice_cloner
    if voice_cloner is None:
        cache_dir = config.get("models", {}).get("cache_dir")
        device = get_device(config.get("models", {}).get("device"))
        voice_cloner = VoiceCloner(cache_dir=cache_dir, device=device)
    return voice_cloner


def _generate_output_path(prefix: str = "speech", fmt: str = "wav") -> str:
    """Generate a unique output file path."""
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.{fmt}"
    return str(output_dir / filename)


def _convert_if_needed(audio_path: str, target_format: str) -> str:
    """Convert audio to target format if different."""
    if target_format == "wav":
        return audio_path
    output_path = str(Path(audio_path).with_suffix(f".{target_format}"))
    return convert_audio(audio_path, output_path, target_format)


# ═══════════════════════════════════════════════
#  Health Check
# ═══════════════════════════════════════════════

@app.get("/health")
async def health_check():
    device = get_device(config.get("models", {}).get("device"))
    voices = voice_manager.list_voices() if voice_manager else []
    return {
        "status": "ok",
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "saved_voices": len(voices),
        "version": "1.0",
    }


# ═══════════════════════════════════════════════
#  Speech Generation (OpenAI-compatible)
# ═══════════════════════════════════════════════

@app.post("/v1/audio/speech")
async def generate_speech(request: SpeechRequest):
    """
    Generate speech from text. OpenAI-compatible endpoint.

    If `voice` is set to a saved voice name, uses that voice's embedding.
    """
    model = _get_tts_model(request.model)
    sample_rate = config.get("audio", {}).get("sample_rate", 24000)
    output_path = _generate_output_path("speech", "wav")

    try:
        # Check if voice is a saved profile
        embedding = None
        if request.voice != "default" and voice_manager:
            embedding = voice_manager.load_embedding(request.voice)
            if embedding is None and request.voice.lower() != "default":
                # Check temp store
                embedding = _temp_voice_store.get(request.voice)

        # Generate speech
        if embedding is not None:
            result = model.generate(
                text=request.input,
                speaker_embedding=embedding,
                language=request.language,
                speed=request.speed,
            )
        else:
            result = model.generate(
                text=request.input,
                language=request.language,
                speed=request.speed,
            )

        # Extract audio
        audio = result.get("audio", result.get("waveform"))
        if audio is None:
            raise HTTPException(status_code=500, detail="Model returned no audio")

        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        if audio.ndim > 1:
            audio = audio.squeeze()

        sf.write(output_path, audio, sample_rate)

        # Convert to requested format
        final_path = _convert_if_needed(output_path, request.response_format)

        return FileResponse(
            final_path,
            media_type=f"audio/{request.response_format}",
            filename=Path(final_path).name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}")


# ═══════════════════════════════════════════════
#  Voice Design
# ═══════════════════════════════════════════════

@app.post("/v1/audio/voice-design")
async def design_voice(request: VoiceDesignRequest):
    """Generate speech with a natural-language voice description."""
    designer = _get_voice_designer()
    output_path = _generate_output_path("vdesign", "wav")

    try:
        result = designer.generate(
            text=request.input,
            voice_description=request.voice_description,
            language=request.language,
            output_path=output_path,
        )

        # Store embedding temporarily for potential saving
        voice_id = result.get("voice_id", "")
        if result.get("embedding") is not None:
            _temp_voice_store[voice_id] = result["embedding"]

        # Convert to requested format
        final_path = _convert_if_needed(output_path, request.response_format)

        return FileResponse(
            final_path,
            media_type=f"audio/{request.response_format}",
            filename=Path(final_path).name,
            headers={"X-Voice-Id": voice_id},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice design failed: {e}")


# ═══════════════════════════════════════════════
#  Voice Cloning
# ═══════════════════════════════════════════════

@app.post("/v1/audio/voice-clone")
async def clone_voice(
    input: str = Form(...),
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(""),
    language: str = Form("en"),
    response_format: str = Form("wav"),
):
    """Clone a voice from reference audio and generate new speech."""
    cloner = _get_voice_cloner()
    output_path = _generate_output_path("vclone", "wav")

    # Save uploaded reference audio to temp file
    ref_suffix = Path(reference_audio.filename or "ref.wav").suffix or ".wav"
    fd, ref_path = tempfile.mkstemp(suffix=ref_suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        result = cloner.clone(
            text=input,
            reference_audio_path=ref_path,
            reference_text=reference_text,
            language=language,
            output_path=output_path,
        )

        # Store embedding temporarily
        voice_id = result.get("voice_id", "")
        if result.get("embedding") is not None:
            _temp_voice_store[voice_id] = result["embedding"]

        # Convert to requested format
        final_path = _convert_if_needed(output_path, response_format)

        return FileResponse(
            final_path,
            media_type=f"audio/{response_format}",
            filename=Path(final_path).name,
            headers={"X-Voice-Id": voice_id},
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")
    finally:
        # Clean up temp reference file
        if os.path.exists(ref_path):
            os.remove(ref_path)


# ═══════════════════════════════════════════════
#  Voice Management (CRUD)
# ═══════════════════════════════════════════════

@app.get("/v1/voices")
async def list_voices():
    """List all saved voice profiles, sorted by usage count."""
    if not voice_manager:
        raise HTTPException(status_code=500, detail="Voice manager not initialized")
    return {"voices": voice_manager.list_voices()}


@app.post("/v1/voices")
async def save_voice(request: VoiceSaveRequest):
    """Save/persist a voice profile with a user-chosen name."""
    if not voice_manager:
        raise HTTPException(status_code=500, detail="Voice manager not initialized")

    # Check for duplicates
    if voice_manager.voice_exists(request.name):
        raise HTTPException(
            status_code=409,
            detail=f"Voice '{request.name}' already exists. Choose a different name or delete the existing one.",
        )

    # Get embedding from temp store or raise error
    embedding = None
    if request.source_voice_id:
        embedding = _temp_voice_store.pop(request.source_voice_id, None)

    if embedding is None:
        raise HTTPException(
            status_code=400,
            detail="No voice embedding found. Generate a voice first using voice-design or voice-clone.",
        )

    try:
        profile = voice_manager.save_voice(
            name=request.name,
            embedding=embedding,
            description=request.description,
            source="voice-design" if request.source_voice_id and request.source_voice_id.startswith("vd_") else "voice-clone",
            source_description=request.description,
            language=request.language,
            tags=request.tags,
        )
        return {"status": "saved", "voice": profile.to_dict()}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/voices/{name}")
async def get_voice(name: str):
    """Get details for a specific voice profile."""
    if not voice_manager:
        raise HTTPException(status_code=500, detail="Voice manager not initialized")

    profile = voice_manager.get_voice(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    return {"voice": profile.to_dict()}


@app.patch("/v1/voices/{name}")
async def update_voice(name: str, request: VoiceUpdateRequest):
    """Rename or update a voice profile."""
    if not voice_manager:
        raise HTTPException(status_code=500, detail="Voice manager not initialized")

    profile = voice_manager.get_voice(name)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    # Handle rename
    if request.name and request.name != name:
        try:
            profile = voice_manager.rename_voice(name, request.name)
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    # Handle metadata updates
    updated = voice_manager.update_voice(
        request.name or name,
        description=request.description,
        tags=request.tags,
        language=request.language,
    )

    if updated:
        return {"status": "updated", "voice": updated.to_dict()}
    return {"status": "no changes"}


@app.delete("/v1/voices/{name}")
async def delete_voice(name: str):
    """Delete a voice profile and all associated files."""
    if not voice_manager:
        raise HTTPException(status_code=500, detail="Voice manager not initialized")

    if not voice_manager.delete_voice(name):
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found")

    return {"status": "deleted", "name": name}


# ═══════════════════════════════════════════════
#  Audio Conversion
# ═══════════════════════════════════════════════

@app.post("/v1/audio/convert")
async def convert_audio_format(
    audio: UploadFile = File(...),
    target_format: str = Form("ogg"),
):
    """Convert audio between formats (WAV, MP3, OGG/Opus, FLAC)."""
    # Save uploaded audio
    in_suffix = Path(audio.filename or "input.wav").suffix or ".wav"
    fd, input_path = tempfile.mkstemp(suffix=in_suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            content = await audio.read()
            f.write(content)

        output_path = _generate_output_path("converted", target_format)
        convert_audio(input_path, output_path, target_format)

        return FileResponse(
            output_path,
            media_type=f"audio/{target_format}",
            filename=Path(output_path).name,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


# ═══════════════════════════════════════════════
#  Telegram Integration
# ═══════════════════════════════════════════════

@app.post("/v1/audio/send/telegram")
async def send_telegram(request: TelegramSendRequest):
    """Send audio as a Telegram PTT voice message."""
    from messaging.telegram_sender import send_voice_message

    bot_token = request.bot_token or config.get("telegram", {}).get("bot_token", "")
    if not bot_token:
        raise HTTPException(
            status_code=400,
            detail="Telegram bot_token required (in request or config.json)",
        )

    audio_path = os.path.expanduser(request.audio_file)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        result = await send_voice_message(
            audio_path=audio_path,
            bot_token=bot_token,
            chat_id=request.chat_id,
            caption=request.caption,
        )
        return {"status": "sent", "telegram_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Telegram send failed: {e}")


# ═══════════════════════════════════════════════
#  WhatsApp Integration
# ═══════════════════════════════════════════════

@app.post("/v1/audio/send/whatsapp")
async def send_whatsapp(request: WhatsAppSendRequest):
    """Send audio as a WhatsApp PTT voice message."""
    from messaging.whatsapp_sender import send_voice_message

    phone_number_id = request.phone_number_id or config.get("whatsapp", {}).get("phone_number_id", "")
    access_token = request.access_token or config.get("whatsapp", {}).get("access_token", "")

    if not phone_number_id or not access_token:
        raise HTTPException(
            status_code=400,
            detail="WhatsApp phone_number_id and access_token required (in request or config.json)",
        )

    audio_path = os.path.expanduser(request.audio_file)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail=f"Audio file not found: {audio_path}")

    try:
        result = await send_voice_message(
            audio_path=audio_path,
            phone_number_id=phone_number_id,
            recipient=request.recipient,
            access_token=access_token,
        )
        return {"status": "sent", "whatsapp_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"WhatsApp send failed: {e}")


# ═══════════════════════════════════════════════
#  Server Info
# ═══════════════════════════════════════════════

@app.get("/v1/models")
async def list_models():
    """List available TTS models."""
    return {
        "models": [
            {"id": key, "hf_id": val, "object": "model"}
            for key, val in MODELS.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8880)
