"""Build clone prompts from reference audio with the Base checkpoint's own encoders.

Mirrors ``Qwen3TTSModel.create_voice_clone_prompt`` from qwen-tts 0.1.1:

* ref codes : ``speech_tokenizer.encode(wav, sr)`` (12 Hz, 16 codebooks) with the
              tokenizer in BF16 like the official model. The engine's decoder copy
              runs in FP32, and encoding with it flips up to ~64 % of the deeper
              residual codes (measured), so the builder keeps its own BF16 copy.
* x-vector  : ``speaker_encoder(mel(wav @ 24 kHz))`` with n_fft 1024,
              128 mels, hop 256, win 1024, fmin 0, fmax 12 kHz, run in the
              model dtype (BF16) like the official model.

The upstream modules are reused rather than reimplemented so the features
match the checkpoint bit for bit.
"""

from __future__ import annotations

import contextlib
import io
import threading
from pathlib import Path

import numpy as np
import torch

from nari_qwen3_tts.model.voices import NUM_CODE_GROUPS, VoicePrompt

SPEAKER_SAMPLE_RATE = 24_000
MIN_REFERENCE_SECONDS = 1.0
MAX_REFERENCE_SECONDS = 60.0


def _quiet_qwen_tts_import():
    # qwen_tts prints a flash-attn advisory banner at import time.
    return contextlib.redirect_stdout(io.StringIO())


class VoicePromptBuilder:
    """Own the speaker encoder and a BF16 speech tokenizer used only for encoding."""

    def __init__(
        self,
        *,
        model_directory: Path,
        speaker_encoder_config: dict,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        with _quiet_qwen_tts_import():
            from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
            from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder
            from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer
        from nari_qwen3_tts.model.loader import iter_safetensors_shards

        self.device = device
        self.dtype = dtype
        encoder = Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**speaker_encoder_config))
        prefix = "speaker_encoder."
        state = {
            name.removeprefix(prefix): tensor
            for name, tensor in iter_safetensors_shards(model_directory, device=torch.device("cpu"), prefix=prefix)
        }
        if not state:
            raise RuntimeError("Base checkpoint has no speaker_encoder.* weights")
        encoder.load_state_dict(state, strict=True)
        self.speaker_encoder = encoder.to(device=device, dtype=dtype).eval()

        self.speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(
            str(model_directory / "speech_tokenizer"), device_map=str(device), dtype=dtype
        )
        self.speech_tokenizer.model.eval()
        # Prompt building is independent of the engine's captured graphs; builds serialize among themselves.
        self._lock = threading.Lock()

    @torch.inference_mode()
    def build(
        self,
        *,
        voice_id: str,
        audio: np.ndarray,
        sample_rate: int,
        ref_text: str | None,
        x_vector_only: bool = False,
        xvector_audio: tuple[np.ndarray, int] | None = None,
    ) -> VoicePrompt:
        wav = self._validate_audio(audio, sample_rate)
        text = (ref_text or "").strip()
        icl = not x_vector_only and bool(text)
        if xvector_audio is not None:
            embed_wav, embed_rate = self._validate_audio(xvector_audio[0], xvector_audio[1]), xvector_audio[1]
        else:
            embed_wav, embed_rate = wav, sample_rate
        with self._lock:
            x_vector = self._speaker_embedding(embed_wav, embed_rate)
            ref_codes = self._ref_codes(wav, sample_rate) if icl else None
        return VoicePrompt(
            voice_id=voice_id,
            x_vector=x_vector,
            ref_codes=ref_codes,
            ref_text=text if icl else None,
        )

    @staticmethod
    def _validate_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 8_000 <= sample_rate <= 192_000:
            raise ValueError(f"sample_rate must be an integer in [8000, 192000], got {sample_rate!r}")
        wav = np.asarray(audio, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1 if wav.shape[1] <= 8 else 0)
        if wav.ndim != 1:
            raise ValueError("reference audio must be mono or [samples, channels]")
        if not np.isfinite(wav).all():
            raise ValueError("reference audio contains non-finite samples")
        seconds = wav.shape[0] / sample_rate
        if not MIN_REFERENCE_SECONDS <= seconds <= MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio must be {MIN_REFERENCE_SECONDS:g}-{MAX_REFERENCE_SECONDS:g} s, got {seconds:.2f} s"
            )
        peak = float(np.abs(wav).max())
        if peak == 0.0:
            raise ValueError("reference audio is silent")
        if peak > 1.0:
            wav = wav / peak
        return wav

    def _speaker_embedding(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        import librosa

        with _quiet_qwen_tts_import():
            from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        if sample_rate != SPEAKER_SAMPLE_RATE:
            wav = librosa.resample(y=wav.astype(np.float32), orig_sr=sample_rate, target_sr=SPEAKER_SAMPLE_RATE)
        mels = mel_spectrogram(
            torch.from_numpy(wav).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=SPEAKER_SAMPLE_RATE,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        embedding = self.speaker_encoder(mels.to(self.device).to(self.dtype))[0]
        return embedding.float().cpu()

    def _ref_codes(self, wav: np.ndarray, sample_rate: int) -> torch.Tensor:
        encoded = self.speech_tokenizer.encode(wav, sr=sample_rate)
        codes = encoded.audio_codes[0]
        if codes.ndim != 2 or codes.shape[1] != NUM_CODE_GROUPS:
            raise RuntimeError(f"speech tokenizer returned codes of shape {tuple(codes.shape)}")
        return codes.to("cpu", torch.long)


__all__ = ["MAX_REFERENCE_SECONDS", "MIN_REFERENCE_SECONDS", "VoicePromptBuilder"]
