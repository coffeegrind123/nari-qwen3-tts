from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract import EncodedText
from nari_qwen3_tts.contract.errors import RequestRejected
from nari_qwen3_tts.contract.request import SynthesisRequest
from nari_qwen3_tts.engine.engine import Engine
from nari_qwen3_tts.model.voices import NUM_CODE_GROUPS, VoicePrompt, VoiceRegistry
from nari_qwen3_tts.planner.catalog import CaptureCatalog
from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

HIDDEN = 4


class _StopError(Exception):
    pass


def _registry() -> VoiceRegistry:
    registry = VoiceRegistry(
        codec_embedding=torch.nn.Embedding(32, HIDDEN),
        frame_embeddings=[torch.nn.Embedding(8, HIDDEN) for _ in range(NUM_CODE_GROUPS - 1)],
        voice_rows=64,
        tokenize_ref_text=lambda text: torch.tensor([7, 8], dtype=torch.long),
    )
    registry.register(
        VoicePrompt(
            voice_id="v_clone",
            x_vector=torch.randn(HIDDEN),
            ref_codes=torch.zeros(3, NUM_CODE_GROUPS, dtype=torch.long),
            ref_text="hello there",
        )
    )
    return registry


def _engine(registry: VoiceRegistry, *, fail_planning: bool = False):
    seen: list = []

    class Talker:
        @staticmethod
        def prepare_prepared_inputs(prepared_inputs, *, voices=None):
            seen.append(voices)
            raise _StopError if fail_planning else _StopError("stop after voice resolution")

    engine = object.__new__(Engine)
    def encode(request, **_):
        return EncodedText(request, torch.tensor([1, 2, 3]), torch.empty(0, dtype=torch.long))

    engine.model = SimpleNamespace(prepare=encode, prepare_live=encode)
    config = ProfileLoader().load_profile(ExecutionProfile.BALANCED)
    engine.executor = SimpleNamespace(talker=Talker(), config=config)
    engine.catalog = CaptureCatalog.from_config(config.stages)
    engine.voices = registry
    engine._voice_pins = {}
    return engine, seen


def _pins(registry: VoiceRegistry) -> int:
    return registry.get("v_clone").pins


def test_admission_resolves_clone_voice_and_releases_pin_when_planning_fails() -> None:
    registry = _registry()
    engine, seen = _engine(registry)
    with pytest.raises(_StopError):
        engine._admit_request("r1", SynthesisRequest(text="hi", voice="v_clone"), admitted_at_s=0.0, live=False)
    assert seen[0][0].icl is not None
    assert _pins(registry) == 0
    assert engine._voice_pins == {}


def test_live_admission_falls_back_to_xvector() -> None:
    registry = _registry()
    engine, seen = _engine(registry)
    with pytest.raises(_StopError):
        engine._admit_request("r1", SynthesisRequest(text="hi", voice="v_clone"), admitted_at_s=0.0, live=True)
    assert seen[0][0].icl is None


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"voice": "v_missing"}, "unknown voice"),
        ({"voice": "v_clone", "instruct": "whisper"}, "instruction"),
    ],
)
def test_admission_rejects_unknown_voices_and_instructions(request_kwargs, message) -> None:
    registry = _registry()
    engine, _seen = _engine(registry)
    with pytest.raises(RequestRejected, match=message):
        engine._admit_request("r1", SynthesisRequest(text="hi", **request_kwargs), admitted_at_s=0.0, live=False)
    assert _pins(registry) == 0


def test_icl_on_live_input_is_rejected() -> None:
    registry = _registry()
    engine, _seen = _engine(registry)
    with pytest.raises(RequestRejected, match="full text"):
        engine._admit_request(
            "r1", SynthesisRequest(text="hi", voice="v_clone", voice_mode="icl"), admitted_at_s=0.0, live=True
        )


def test_pin_is_held_until_removal() -> None:
    registry = _registry()
    engine, _seen = _engine(registry)

    # Simulate a successful admission's bookkeeping, then removal.
    registry.acquire("v_clone", mode=SynthesisRequest(text="x").voice_mode, live=False)
    engine._voice_pins["r9"] = "v_clone"
    engine.output = SimpleNamespace(discard_request=lambda request_id: 0)
    state = SimpleNamespace(codec=SimpleNamespace(pending_outputs=0))
    engine.pipeline = SimpleNamespace(request=lambda request_id: state, remove=lambda request_id: None)
    assert _pins(registry) == 1
    engine._remove_request("r9")
    assert _pins(registry) == 0


def test_preset_engine_rejects_unknown_preset_at_admission() -> None:
    engine = object.__new__(Engine)
    engine.model = SimpleNamespace(
        prepare=lambda request: EncodedText(request, torch.tensor([1, 2, 3]), torch.empty(0, dtype=torch.long)),
        model_config=SimpleNamespace(talker=SimpleNamespace(spk_id={"aiden": 1})),
    )
    with pytest.raises(RequestRejected, match="preset"):
        engine._admit_request("r1", SynthesisRequest(text="hi", voice="nobody"), admitted_at_s=0.0, live=False)
