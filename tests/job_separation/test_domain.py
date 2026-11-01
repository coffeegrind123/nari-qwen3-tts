from __future__ import annotations

import pytest

from nari_qwen3_tts.contract.request import SynthesisRequest


def test_public_request_domain_is_strict() -> None:
    request = SynthesisRequest(
        text="hello",
        voice="Aiden",
        language="English",
        non_streaming_mode=False,
        random_seed=7,
    )
    assert request.voice == "aiden"
    assert request.language == "english"
    assert request.non_streaming_mode is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": "   "},
        {"text": "hello", "random_seed": True},
        {"text": "hello", "top_k": True},
        {"text": "hello", "voice": 3},
        {"text": "hello", "do_sample": 1},
        {"text": "hello", "non_streaming_mode": 0},
    ],
)
def test_public_request_rejects_bool_as_int_and_invalid_text(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        SynthesisRequest(**kwargs)


def test_malformed_voice_and_unknown_language_fail_closed() -> None:
    # Voice existence is checked at admission (presets or registered clones);
    # the contract rejects names that can never be valid.
    with pytest.raises(ValueError, match="voice name"):
        SynthesisRequest(text="hello", voice="no/such voice!")
    with pytest.raises(ValueError, match="language"):
        SynthesisRequest(text="hello", language="unknown")
