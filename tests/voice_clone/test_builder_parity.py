"""Opt-in parity check: VoicePromptBuilder against qwen-tts 0.1.1 create_voice_clone_prompt.

Needs a local Base snapshot: set QWEN3_TTS_BASE_DIR (and optionally QWEN3_TTS_PARITY_DEVICE).
Runs on CPU in a few minutes; the engine's GPU path reuses the same modules.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
import torch

BASE_DIR = os.environ.get("QWEN3_TTS_BASE_DIR")
pytestmark = pytest.mark.skipif(not BASE_DIR, reason="set QWEN3_TTS_BASE_DIR to a Qwen3-TTS Base snapshot")

REF_TEXT = "The quick brown fox jumps over the lazy dog, then naps in the warm afternoon sun."


def _reference_audio(sample_rate: int = 24_000, seconds: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    pitch = 120 + 30 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return (0.15 * voiced + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(os.environ.get("QWEN3_TTS_PARITY_DEVICE", "cpu"))


@pytest.fixture(scope="module")
def official(device):
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts import Qwen3TTSModel
    return Qwen3TTSModel.from_pretrained(BASE_DIR, device_map=str(device), dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def builder(device):
    with contextlib.redirect_stdout(io.StringIO()):
        from qwen_tts.inference.qwen3_tts_tokenizer import Qwen3TTSTokenizer

    from nari_qwen3_tts.model.config import Qwen3TTSConfig
    from nari_qwen3_tts.model.voice_builder import VoicePromptBuilder

    config = Qwen3TTSConfig.from_pretrained_dir(BASE_DIR)
    # Same load path as CheckpointLoader.load_codec + load_assets (BF16 load, then float()).
    codec = Qwen3TTSTokenizer.from_pretrained(
        str(Path(BASE_DIR) / "speech_tokenizer"), device_map=str(device), dtype=torch.bfloat16
    ).model.eval().float()
    return VoicePromptBuilder(
        model_directory=Path(BASE_DIR),
        speaker_encoder_config=config.speaker_encoder.raw,
        codec_model=codec,
        device=device,
    )


def test_xvector_and_codes_match_official(official, builder) -> None:
    audio = _reference_audio()
    expected = official.create_voice_clone_prompt(ref_audio=(audio, 24_000), ref_text=REF_TEXT)[0]
    prompt = builder.build(voice_id="parity", audio=audio, sample_rate=24_000, ref_text=REF_TEXT)

    reference_xvector = expected.ref_spk_embedding.float().cpu().reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(prompt.x_vector, reference_xvector, dim=0).item()
    assert cosine > 0.999, cosine

    reference_codes = expected.ref_code.cpu().to(torch.long)
    assert prompt.ref_codes.shape == reference_codes.shape
    agreement = (prompt.ref_codes == reference_codes).float().mean().item()
    # The engine's codec runs in FP32 (the official tokenizer in BF16); a few boundary codes may flip.
    assert agreement > 0.9, agreement


def test_reference_text_ids_match_official(official) -> None:
    from nari_qwen3_tts.model.text import Qwen3TTSTextDomain

    domain = Qwen3TTSTextDomain(BASE_DIR)
    expected = official._tokenize_texts([official._build_ref_text(REF_TEXT)])[0][0, 3:-2]
    assert domain.tokenize_ref_text(REF_TEXT).tolist() == expected.tolist()
