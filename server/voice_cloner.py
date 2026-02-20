"""
voice_cloner.py — Clone voices from reference audio.

Uses Qwen3-TTS-12Hz-0.6B-CustomVoice model to clone a voice from
a short reference audio clip (minimum 3 seconds, recommended 10-30s),
then generate new speech in that voice.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import soundfile as sf


class VoiceCloner:
    """
    Clones a voice from reference audio and generates new speech in that voice.

    Requirements:
        - Minimum 3 seconds of reference audio
        - Recommended 10-30 seconds for best quality
        - Accurate transcription of reference audio improves results
        - Supports cross-language cloning
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        embedding_model_id: str = "Qwen/Qwen3-Voice-Embedding-12Hz-0.6B",
        cache_dir: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        self.cache_dir = cache_dir
        self.device = self._resolve_device(device)
        self._model = None
        self._embedding_model = None
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
        """Lazy-load models on first use."""
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
            raise RuntimeError(f"Failed to load voice cloning model: {e}")

    def clone(
        self,
        text: str,
        reference_audio_path: str,
        reference_text: str = "",
        language: str = "en",
        output_path: Optional[str] = None,
        sample_rate: int = 24000,
    ) -> dict:
        """
        Clone a voice from reference audio and generate new speech.

        Args:
            text: The new text to synthesize in the cloned voice
            reference_audio_path: Path to the reference audio file
            reference_text: Transcription of the reference audio (recommended)
            language: Target language code
            output_path: Where to save the output. If None, uses a temp file.
            sample_rate: Audio sample rate

        Returns:
            dict with keys:
                - audio_path: Path to the generated audio
                - voice_id: Temporary voice ID for saving
                - embedding: Speaker embedding tensor
        """
        self._ensure_loaded()

        if not os.path.exists(reference_audio_path):
            raise FileNotFoundError(
                f"Reference audio not found: {reference_audio_path}"
            )

        # Check minimum duration
        from audio_converter import get_audio_duration
        duration = get_audio_duration(reference_audio_path)
        if 0 < duration < 3.0:
            raise ValueError(
                f"Reference audio is too short ({duration:.1f}s). "
                f"Minimum 3 seconds required, 10-30 seconds recommended."
            )

        if output_path is None:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        try:
            # Generate cloned speech
            result = self._model.generate_voice_clone(
                text=text,
                reference_audio=reference_audio_path,
                reference_text=reference_text,
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
            voice_id = f"vc_{hash(reference_audio_path) & 0xFFFFFFFF:08x}"

            return {
                "audio_path": output_path,
                "voice_id": voice_id,
                "embedding": embedding,
                "reference_audio": reference_audio_path,
                "reference_text": reference_text,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            raise RuntimeError(f"Voice cloning failed: {e}")

    def extract_embedding(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        Extract a speaker embedding from an audio file without generating speech.

        Useful for pre-processing reference audio into a reusable embedding.
        """
        try:
            from qwen_tts import QwenTTS

            if self._embedding_model is None:
                self._embedding_model = QwenTTS.from_pretrained(
                    self.embedding_model_id,
                    cache_dir=self.cache_dir,
                    device=self.device,
                )

            result = self._embedding_model.extract_embedding(audio_path)
            return result.get("embedding", result.get("speaker_embedding", None))

        except Exception as e:
            print(f"[WARN] Embedding extraction failed: {e}")
            return None

    def unload(self):
        """Unload models to free memory."""
        for model in [self._model, self._embedding_model]:
            if model is not None:
                del model
        self._model = None
        self._embedding_model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
