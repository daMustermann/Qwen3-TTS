"""
voice_designer.py — Create new voices from natural language descriptions.

Uses Qwen3-TTS-12Hz-1.7B-VoiceDesign model to generate speech with
a voice matching the given description, and optionally extracts the
speaker embedding for persistence.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf


class VoiceDesigner:
    """
    Generates speech using a natural language voice description.

    The voice-design model takes a text description of the desired voice
    (e.g. "a warm British male voice, calm and authoritative") and produces
    speech in that style. The resulting speaker embedding can be saved
    for future reuse.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                 cache_dir: Optional[str] = None, device: str = "auto"):
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device)
        self._model = None
        self._loaded = False

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        return "cpu"

    def _ensure_loaded(self):
        """Lazy-load the model on first use."""
        if self._loaded:
            return

        try:
            from qwen_tts import QwenTTS
            self._model = QwenTTS.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                device=self.device,
            )
            self._loaded = True
        except ImportError:
            raise RuntimeError(
                "qwen-tts package not installed. Run: pip install qwen-tts"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load voice design model: {e}")

    def generate(
        self,
        text: str,
        voice_description: str,
        language: str = "en",
        output_path: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> dict:
        """
        Generate speech with a designed voice.

        Args:
            text: The text to synthesize
            voice_description: Natural language description of the desired voice
            language: Target language code
            output_path: Where to save the audio. If None, uses a temp file.
            sample_rate: Audio sample rate

        Returns:
            dict with keys:
                - audio_path: Path to the generated audio file
                - voice_id: Temporary voice ID for saving
                - embedding: Speaker embedding tensor (if extractable)
        """
        self._ensure_loaded()

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            # Generate speech with voice design
            result = self._model.generate_voice_design(
                text=text,
                voice_description=voice_description,
                language=language,
            )

            # Extract audio and embedding
            audio = result.get("audio", result.get("waveform", None))
            embedding = result.get("speaker_embedding", result.get("embedding", None))

            if audio is None:
                raise RuntimeError("Model did not return audio data")

            # Convert to numpy if tensor
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            # Ensure 1D
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Save audio
            sf.write(output_path, audio, sample_rate)

            # Generate a temporary voice ID
            voice_id = f"vd_{hash(voice_description) & 0xFFFFFFFF:08x}"

            return {
                "audio_path": output_path,
                "voice_id": voice_id,
                "embedding": embedding,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            raise RuntimeError(f"Voice design generation failed: {e}")

    def unload(self):
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
