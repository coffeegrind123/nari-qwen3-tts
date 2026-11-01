"""Clone-voice registration endpoints (Base checkpoints only).

The engine keeps prompts resident in GPU voice rows but owns no durable
storage: callers persist ``prompt_base64`` (safetensors) and re-register it
after a restart with ``PUT /v1/voices/{voice_id}``.
"""

from __future__ import annotations

import base64
import binascii
import io
import re
from collections.abc import Iterator
from contextlib import contextmanager

import anyio.to_thread
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from nari_qwen3_tts.api.schemas import VoiceBuildBody, VoiceBuildResult, VoiceInfoBody
from nari_qwen3_tts.engine.engine import Engine
from nari_qwen3_tts.model.voices import (
    VoiceCapacityError,
    VoiceInfo,
    VoiceModeError,
    VoiceNotFoundError,
    VoicePrompt,
)

_VOICE_ID = re.compile(r"^[a-z0-9][a-z0-9_]{0,63}$")
MAX_PROMPT_BYTES = 8 * 1024 * 1024


def _info(info: VoiceInfo) -> VoiceInfoBody:
    return VoiceInfoBody(
        voice_id=info.voice_id,
        supports_icl=info.supports_icl,
        frames=info.frames,
        rows=info.rows,
        pins=info.pins,
    )


def _require_voice_id(voice_id: str) -> str:
    if not _VOICE_ID.fullmatch(voice_id):
        raise HTTPException(422, f"invalid voice id {voice_id!r}")
    return voice_id


def _decode_audio(payload: str):
    import soundfile

    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(422, "audio_base64 is not valid base64") from error
    try:
        audio, sample_rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    except (RuntimeError, soundfile.LibsndfileError) as error:
        raise HTTPException(422, f"unreadable reference audio: {error}") from error
    return audio, int(sample_rate)


@contextmanager
def _registry_errors() -> Iterator[None]:
    """Map voice-registry failures onto HTTP statuses the gateway understands."""
    try:
        yield
    except VoiceCapacityError as error:
        raise HTTPException(507, str(error)) from error
    except VoiceModeError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


def build_voice_router(engine: Engine) -> APIRouter:
    router = APIRouter(prefix="/v1/voices", tags=["voices"])

    def registry():
        if engine.voices is None or engine.voice_builder is None:
            raise HTTPException(404, "this engine serves a CustomVoice checkpoint; clone voices need the Base model")
        return engine.voices

    def require_ready() -> None:
        if not engine.readiness().ready:
            raise HTTPException(503, "engine is not ready")

    @router.get("")
    def list_voices() -> dict[str, object]:
        voices = registry()
        return {
            "object": "list",
            "data": [_info(info).model_dump() for info in voices.list()],
            "free_rows": voices.free_rows,
            "total_rows": voices.voice_rows,
        }

    @router.get("/{voice_id}")
    def get_voice(voice_id: str) -> VoiceInfoBody:
        voices = registry()
        try:
            return _info(voices.get(_require_voice_id(voice_id)))
        except VoiceNotFoundError as error:
            raise HTTPException(404, f"voice {voice_id!r} is not registered") from error

    @router.post("", status_code=201)
    def build_voice(body: VoiceBuildBody) -> VoiceBuildResult:
        voices = registry()
        require_ready()
        audio, sample_rate = _decode_audio(body.audio_base64)
        xvector = _decode_audio(body.xvector_audio_base64) if body.xvector_audio_base64 else None
        with _registry_errors():
            prompt = engine.voice_builder.build(
                voice_id=body.voice_id,
                audio=audio,
                sample_rate=sample_rate,
                ref_text=body.ref_text,
                x_vector_only=body.x_vector_only,
                xvector_audio=xvector,
            )
            info = voices.register(prompt)
        return VoiceBuildResult(
            voice=_info(info),
            prompt_base64=base64.b64encode(prompt.to_bytes()).decode("ascii"),
        )

    @router.put("/{voice_id}")
    async def register_prompt(voice_id: str, request: Request) -> VoiceInfoBody:
        voices = registry()
        require_ready()
        _require_voice_id(voice_id)
        payload = await request.body()
        if not payload or len(payload) > MAX_PROMPT_BYTES:
            raise HTTPException(413 if payload else 422, f"prompt must be 1..{MAX_PROMPT_BYTES} bytes of safetensors")
        try:
            prompt = VoicePrompt.from_bytes(payload, voice_id=voice_id)
        except (ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, f"invalid voice prompt: {error}") from error
        with _registry_errors():
            info = await anyio.to_thread.run_sync(voices.register, prompt)
        return _info(info)

    @router.delete("/{voice_id}", status_code=204)
    def delete_voice(voice_id: str) -> Response:
        voices = registry()
        try:
            voices.unregister(_require_voice_id(voice_id))
        except VoiceNotFoundError as error:
            raise HTTPException(404, f"voice {voice_id!r} is not registered") from error
        except VoiceModeError as error:
            raise HTTPException(409, str(error)) from error
        return Response(status_code=204)

    return router


__all__ = ["build_voice_router"]
