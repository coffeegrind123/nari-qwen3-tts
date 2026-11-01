from __future__ import annotations

import pytest
import torch

from nari_qwen3_tts.contract.request import SynthesisRequest, VoiceCloneMode
from nari_qwen3_tts.model.voices import (
    NUM_CODE_GROUPS,
    VoiceCapacityError,
    VoiceModeError,
    VoiceNotFoundError,
    VoicePrompt,
    VoiceRegistry,
)

HIDDEN = 4
VOCAB = 32


def _registry(rows: int = 16) -> VoiceRegistry:
    codec = torch.nn.Embedding(VOCAB, HIDDEN)
    frames = [torch.nn.Embedding(8, HIDDEN) for _ in range(NUM_CODE_GROUPS - 1)]
    return VoiceRegistry(
        codec_embedding=codec,
        frame_embeddings=frames,
        voice_rows=rows,
        tokenize_ref_text=lambda text: torch.arange(len(text.split()), dtype=torch.long) + 50,
    )


def _prompt(voice_id: str, frames: int = 0) -> VoicePrompt:
    if frames == 0:
        return VoicePrompt(voice_id=voice_id, x_vector=torch.randn(HIDDEN))
    codes = torch.zeros(frames, NUM_CODE_GROUPS, dtype=torch.long)
    return VoicePrompt(voice_id=voice_id, x_vector=torch.randn(HIDDEN), ref_codes=codes, ref_text="a b c")


def test_table_starts_with_checkpoint_rows_and_writes_voice_rows() -> None:
    registry = _registry()
    torch.testing.assert_close(registry.table[:VOCAB], registry._codec_embedding.weight)
    prompt = _prompt("one")
    registry.register(prompt)
    voice = registry.acquire("one", mode=VoiceCloneMode.AUTO, live=False)
    assert voice.speaker_codec_id >= VOCAB
    torch.testing.assert_close(registry.table[voice.speaker_codec_id], prompt.x_vector)


def test_auto_mode_prefers_icl_and_falls_back_for_live_input() -> None:
    registry = _registry()
    registry.register(_prompt("ref", frames=3))
    assert registry.acquire("ref", mode=VoiceCloneMode.AUTO, live=False).icl is not None
    assert registry.acquire("ref", mode=VoiceCloneMode.AUTO, live=True).icl is None
    with pytest.raises(VoiceModeError, match="full text"):
        registry.acquire("ref", mode=VoiceCloneMode.ICL, live=True)


def test_icl_requires_a_transcript() -> None:
    registry = _registry()
    registry.register(_prompt("plain"))
    with pytest.raises(VoiceModeError, match="transcript"):
        registry.acquire("plain", mode=VoiceCloneMode.ICL, live=False)


def test_icl_frame_rows_follow_the_xvector_row() -> None:
    registry = _registry()
    registry.register(_prompt("ref", frames=3))
    voice = registry.acquire("ref", mode=VoiceCloneMode.ICL, live=False)
    assert voice.icl.frame_codec_ids.tolist() == [voice.speaker_codec_id + 1 + i for i in range(3)]
    assert voice.icl.ref_text_ids.tolist() == [50, 51, 52]


def test_unknown_voice_raises() -> None:
    with pytest.raises(VoiceNotFoundError):
        _registry().acquire("missing", mode=VoiceCloneMode.AUTO, live=False)


def test_pinned_voice_is_never_evicted_or_replaced() -> None:
    registry = _registry(rows=8)
    registry.register(_prompt("busy", frames=5))  # 6 rows
    registry.acquire("busy", mode=VoiceCloneMode.AUTO, live=False)
    with pytest.raises(VoiceCapacityError, match="in use"):
        registry.register(_prompt("big", frames=4))  # needs 5, only 2 free
    with pytest.raises(VoiceModeError):
        registry.unregister("busy")
    with pytest.raises(VoiceModeError):
        registry.register(_prompt("busy"))
    registry.release("busy")
    registry.register(_prompt("big", frames=4))
    assert {info.voice_id for info in registry.list()} == {"big"}


def test_lru_eviction_frees_the_least_recently_used_idle_voice() -> None:
    registry = _registry(rows=6)
    registry.register(_prompt("a", frames=1))  # 2 rows
    registry.register(_prompt("b", frames=1))  # 2 rows
    registry.register(_prompt("c", frames=1))  # 2 rows: pool full
    registry.acquire("a", mode=VoiceCloneMode.XVECTOR, live=False)
    registry.release("a")  # "a" is now the most recently used
    registry.register(_prompt("d", frames=1))
    assert {info.voice_id for info in registry.list()} == {"a", "c", "d"}


def test_freed_ranges_coalesce() -> None:
    registry = _registry(rows=6)
    for name in ("a", "b", "c"):
        registry.register(_prompt(name, frames=1))
    registry.unregister("a")
    registry.unregister("b")
    registry.register(_prompt("wide", frames=3))  # needs 4 contiguous rows
    assert registry.free_rows == 0


def test_prompt_larger_than_pool_is_rejected_up_front() -> None:
    with pytest.raises(VoiceCapacityError, match="shorten"):
        _registry(rows=4).register(_prompt("huge", frames=10))


def test_release_without_acquire_is_a_bug() -> None:
    registry = _registry()
    registry.register(_prompt("x"))
    with pytest.raises(RuntimeError):
        registry.release("x")


def test_prompt_round_trips_through_safetensors() -> None:
    prompt = _prompt("round", frames=4)
    restored = VoicePrompt.from_bytes(prompt.to_bytes())
    assert restored.voice_id == "round"
    assert restored.ref_text == "a b c"
    torch.testing.assert_close(restored.x_vector, prompt.x_vector)
    assert torch.equal(restored.ref_codes, prompt.ref_codes)
    xvector_only = VoicePrompt.from_bytes(_prompt("xv").to_bytes(), voice_id="renamed")
    assert xvector_only.voice_id == "renamed" and xvector_only.ref_codes is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ref_codes": torch.zeros(3, NUM_CODE_GROUPS, dtype=torch.long)},
        {"ref_text": "orphan transcript"},
        {"ref_codes": torch.zeros(3, 4, dtype=torch.long), "ref_text": "t"},
        {"ref_codes": torch.zeros(3, NUM_CODE_GROUPS), "ref_text": "t"},
    ],
)
def test_invalid_prompts_fail_closed(kwargs) -> None:
    with pytest.raises((ValueError, TypeError)):
        VoicePrompt(voice_id="bad", x_vector=torch.zeros(HIDDEN), **kwargs)


def test_request_contract_accepts_clone_names_and_modes() -> None:
    request = SynthesisRequest(text="hi", voice="v_01abc", voice_mode="xvector")
    assert request.voice == "v_01abc"
    assert request.voice_mode is VoiceCloneMode.XVECTOR
    with pytest.raises(ValueError, match="voice_mode"):
        SynthesisRequest(text="hi", voice="v_01abc", voice_mode="karaoke")
    with pytest.raises(ValueError, match="voice name"):
        SynthesisRequest(text="hi", voice="../etc/passwd")


def test_registry_built_under_inference_mode_accepts_later_registrations() -> None:
    # build_qwen3_tts_engine runs under @torch.inference_mode(), so the table is an
    # inference tensor; serve.py and the HTTP API register voices outside that mode.
    with torch.inference_mode():
        registry = _registry()
    prompt = _prompt("late", frames=2)
    registry.register(prompt)
    voice = registry.acquire("late", mode=VoiceCloneMode.ICL, live=False)
    torch.testing.assert_close(registry.table[voice.speaker_codec_id], prompt.x_vector)
