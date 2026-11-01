"""Registered clone voices and the extended codec table that carries them.

Layout of the table the Talker prefill graph reads codec embeddings from::

    row 0 .. V-1        checkpoint talker.model.codec_embedding (copied once)
    row V .. V+R-1      voice-row pool
                          voice A: [x-vector][frame 0][frame 1]...[frame T-1]
                          voice B: [x-vector]                 (x-vector only)
                          free ...

A voice is one contiguous range. Its first row is the speaker encoder output
(the official ``ref_spk_embedding``); the rest, when present, are the summed
16-codebook embeddings of each reference frame (the official ICL
``codec_embed``). Prompts then reference rows by integer id, so the captured
prefill graphs need no new inputs.

Concurrency contract:
* ``register``/``unregister`` run on API threads; ``acquire``/``release`` run
  on the engine thread at admission/removal. One lock guards the bookkeeping.
* A range is written only while no admitted request references it (new
  ranges come from the free list; eviction requires zero pins), and the write
  is synchronized before the voice becomes resolvable. In-flight graph
  replays therefore never observe a partially written voice.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import torch

from nari_qwen3_tts.contract.request import VoiceCloneMode
from nari_qwen3_tts.model.input_layout import IclPrompt, ResolvedVoice

PROMPT_FORMAT_VERSION = 1
NUM_CODE_GROUPS = 16


class VoiceNotFoundError(LookupError):
    """The requested voice is neither a preset nor a registered prompt."""


class VoiceCapacityError(RuntimeError):
    """The voice-row pool cannot fit the prompt even after evicting idle voices."""


class VoiceModeError(ValueError):
    """The requested clone mode is not available for this voice or request."""


@dataclass(frozen=True, slots=True)
class VoicePrompt:
    """Portable, model-specific clone prompt (persisted by the gateway).

    ``x_vector`` is float32 ``[hidden]``. ``ref_codes`` is ``[T, 16]`` codec
    ids of the reference audio and ``ref_text`` its exact transcript; both are
    required for ICL and absent for x-vector-only voices.
    """

    voice_id: str
    x_vector: torch.Tensor
    ref_codes: torch.Tensor | None = None
    ref_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.voice_id, str) or not self.voice_id:
            raise ValueError("voice_id must be a non-empty string")
        if not isinstance(self.x_vector, torch.Tensor) or self.x_vector.ndim != 1:
            raise TypeError("x_vector must be a 1-D tensor")
        if not torch.isfinite(self.x_vector).all():
            raise ValueError("x_vector must be finite")
        if (self.ref_codes is None) != (self.ref_text is None or self.ref_text == ""):
            raise ValueError("ref_codes and ref_text must be provided together")
        if self.ref_codes is not None:
            codes = self.ref_codes
            if codes.ndim != 2 or codes.shape[1] != NUM_CODE_GROUPS or codes.shape[0] < 1:
                raise ValueError(f"ref_codes must be [frames, {NUM_CODE_GROUPS}]")
            if codes.dtype not in (torch.int16, torch.int32, torch.int64):
                raise TypeError("ref_codes must be an integer tensor")

    @property
    def supports_icl(self) -> bool:
        return self.ref_codes is not None

    @property
    def frames(self) -> int:
        return 0 if self.ref_codes is None else int(self.ref_codes.shape[0])

    def to_bytes(self) -> bytes:
        """Serialize to safetensors; text metadata travels in the header."""
        from safetensors.torch import save

        tensors = {"x_vector": self.x_vector.detach().to("cpu", torch.float32).contiguous()}
        metadata = {"format": str(PROMPT_FORMAT_VERSION), "voice_id": self.voice_id}
        if self.ref_codes is not None:
            tensors["ref_codes"] = self.ref_codes.detach().to("cpu", torch.int32).contiguous()
            metadata["ref_text"] = self.ref_text or ""
        return save(tensors, metadata=metadata)

    @classmethod
    def from_bytes(cls, payload: bytes, *, voice_id: str | None = None) -> VoicePrompt:
        # safetensors.torch.load reads tensors from bytes but drops the header
        # metadata, so the metadata block is parsed from the header directly.
        from safetensors.torch import load

        tensors = load(payload)
        metadata = _safetensors_metadata(payload)
        if metadata.get("format") != str(PROMPT_FORMAT_VERSION):
            raise ValueError(f"unsupported voice prompt format: {metadata.get('format')!r}")
        ref_codes = tensors.get("ref_codes")
        return cls(
            voice_id=voice_id or metadata.get("voice_id", ""),
            x_vector=tensors["x_vector"],
            ref_codes=None if ref_codes is None else ref_codes.to(torch.long),
            ref_text=metadata.get("ref_text") if ref_codes is not None else None,
        )


def _safetensors_metadata(payload: bytes) -> dict[str, str]:
    import json
    import struct

    if len(payload) < 8:
        raise ValueError("voice prompt payload is truncated")
    (header_size,) = struct.unpack("<Q", payload[:8])
    if header_size <= 0 or 8 + header_size > len(payload):
        raise ValueError("voice prompt payload has an invalid header")
    header = json.loads(payload[8 : 8 + header_size])
    metadata = header.get("__metadata__", {})
    if not isinstance(metadata, dict):
        raise ValueError("voice prompt metadata must be a mapping")
    return {str(key): str(value) for key, value in metadata.items()}


@dataclass(slots=True)
class _ResidentVoice:
    prompt: VoicePrompt
    start: int
    rows: int
    ref_text_ids: torch.Tensor | None
    pins: int = 0
    last_used: float = field(default_factory=time.monotonic)

    @property
    def xvector_row(self) -> int:
        return self.start

    def frame_rows(self) -> torch.Tensor:
        return torch.arange(self.start + 1, self.start + self.rows, dtype=torch.long)


class _RangeAllocator:
    """First-fit allocator over ``[base, base + size)`` with coalescing frees."""

    def __init__(self, base: int, size: int) -> None:
        self._free: list[tuple[int, int]] = [(base, size)] if size > 0 else []

    def allocate(self, rows: int) -> int | None:
        for index, (start, length) in enumerate(self._free):
            if length >= rows:
                if length == rows:
                    self._free.pop(index)
                else:
                    self._free[index] = (start + rows, length - rows)
                return start
        return None

    def free(self, start: int, rows: int) -> None:
        self._free.append((start, rows))
        self._free.sort()
        merged: list[tuple[int, int]] = []
        for begin, length in self._free:
            if merged and merged[-1][0] + merged[-1][1] == begin:
                merged[-1] = (merged[-1][0], merged[-1][1] + length)
            else:
                merged.append((begin, length))
        self._free = merged

    @property
    def free_rows(self) -> int:
        return sum(length for _start, length in self._free)

    @property
    def largest_free(self) -> int:
        return max((length for _start, length in self._free), default=0)


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    voice_id: str
    supports_icl: bool
    frames: int
    rows: int
    pins: int


class VoiceRegistry:
    """Own the extended codec table and every resident clone prompt."""

    def __init__(
        self,
        *,
        codec_embedding: torch.nn.Embedding,
        frame_embeddings: list[torch.nn.Embedding],
        voice_rows: int,
        tokenize_ref_text,
    ) -> None:
        if isinstance(voice_rows, bool) or not isinstance(voice_rows, int) or voice_rows < 2:
            raise ValueError("voice_rows must be an integer >= 2")
        if len(frame_embeddings) != NUM_CODE_GROUPS - 1:
            raise ValueError(f"expected {NUM_CODE_GROUPS - 1} code predictor embeddings")
        weight = codec_embedding.weight
        self.vocab_size = int(weight.shape[0])
        self.hidden_size = int(weight.shape[1])
        self.device = weight.device
        self.dtype = weight.dtype
        self._codec_embedding = codec_embedding
        self._frame_embeddings = frame_embeddings
        self._tokenize_ref_text = tokenize_ref_text
        self.table = torch.empty(
            (self.vocab_size + voice_rows, self.hidden_size),
            dtype=self.dtype,
            device=self.device,
        )
        with torch.no_grad():
            self.table[: self.vocab_size].copy_(weight)
            self.table[self.vocab_size :].zero_()
        self.voice_rows = voice_rows
        self._allocator = _RangeAllocator(self.vocab_size, voice_rows)
        self._voices: dict[str, _ResidentVoice] = {}
        self._lock = threading.Lock()

    # ---- registration (API threads) -------------------------------------------------

    def register(self, prompt: VoicePrompt) -> VoiceInfo:
        """Write the prompt into the pool, replacing an unpinned voice with the same id."""
        if prompt.x_vector.numel() != self.hidden_size:
            raise ValueError(f"x_vector must have {self.hidden_size} values, got {prompt.x_vector.numel()}")
        rows = 1 + prompt.frames
        if rows > self.voice_rows:
            raise VoiceCapacityError(
                f"voice needs {rows} rows but the pool holds {self.voice_rows}; shorten the reference audio"
            )
        ref_text_ids = self._tokenize_ref_text(prompt.ref_text) if prompt.supports_icl else None
        block = self._prompt_rows(prompt)
        with self._lock:
            previous = self._voices.get(prompt.voice_id)
            if previous is not None:
                if previous.pins:
                    raise VoiceModeError(f"voice {prompt.voice_id!r} is in use and cannot be replaced now")
                self._evict_locked(prompt.voice_id)
            start = self._allocator.allocate(rows)
            while start is None:
                if not self._evict_lru_locked():
                    raise VoiceCapacityError(
                        f"voice pool full: need {rows} contiguous rows, "
                        f"{self._allocator.largest_free} available and every resident voice is in use"
                    )
                start = self._allocator.allocate(rows)
            with torch.no_grad():
                self.table[start : start + rows].copy_(block)
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            resident = _ResidentVoice(prompt=prompt, start=start, rows=rows, ref_text_ids=ref_text_ids)
            self._voices[prompt.voice_id] = resident
            return self._info(resident)

    def unregister(self, voice_id: str) -> None:
        with self._lock:
            resident = self._voices.get(voice_id)
            if resident is None:
                raise VoiceNotFoundError(voice_id)
            if resident.pins:
                raise VoiceModeError(f"voice {voice_id!r} is in use by {resident.pins} request(s)")
            self._evict_locked(voice_id)

    def list(self) -> list[VoiceInfo]:
        with self._lock:
            return [self._info(voice) for voice in self._voices.values()]

    def get(self, voice_id: str) -> VoiceInfo:
        with self._lock:
            resident = self._voices.get(voice_id)
            if resident is None:
                raise VoiceNotFoundError(voice_id)
            return self._info(resident)

    @property
    def free_rows(self) -> int:
        with self._lock:
            return self._allocator.free_rows

    # ---- admission (engine thread) --------------------------------------------------

    def acquire(self, voice_id: str, *, mode: VoiceCloneMode, live: bool) -> ResolvedVoice:
        """Pin a voice for one admitted request and return its prompt rows."""
        with self._lock:
            resident = self._voices.get(voice_id)
            if resident is None:
                raise VoiceNotFoundError(voice_id)
            use_icl = self._select_icl(resident, mode=mode, live=live)
            resident.pins += 1
            resident.last_used = time.monotonic()
            icl = None
            if use_icl:
                assert resident.ref_text_ids is not None
                icl = IclPrompt(ref_text_ids=resident.ref_text_ids, frame_codec_ids=resident.frame_rows())
            return ResolvedVoice(speaker_codec_id=resident.xvector_row, icl=icl)

    def release(self, voice_id: str) -> None:
        with self._lock:
            resident = self._voices.get(voice_id)
            if resident is None or resident.pins == 0:
                raise RuntimeError(f"voice {voice_id!r} released without a matching acquire")
            resident.pins -= 1
            resident.last_used = time.monotonic()

    @staticmethod
    def _select_icl(resident: _ResidentVoice, *, mode: VoiceCloneMode, live: bool) -> bool:
        if mode is VoiceCloneMode.XVECTOR:
            return False
        if mode is VoiceCloneMode.ICL:
            if not resident.prompt.supports_icl:
                raise VoiceModeError(
                    f"voice {resident.prompt.voice_id!r} has no reference transcript; ICL is unavailable"
                )
            if live:
                # Streaming ICL zips target text against reference frames inside
                # prefill, so the target text must be known at admission.
                raise VoiceModeError("ICL voices need the full text up front; use voice_mode=xvector for live input")
            return True
        return resident.prompt.supports_icl and not live

    # ---- internals ------------------------------------------------------------------

    @torch.no_grad()
    def _prompt_rows(self, prompt: VoicePrompt) -> torch.Tensor:
        """Build ``[1 + T, hidden]`` rows exactly as qwen-tts 0.1.1 builds the embeddings.

        x-vector: ``ref_spk_embedding.to(talker.dtype)``.
        frames:   ``cat([emb_0(c[:, :1]), emb_1(c[:, 1:2]), ...], dim=1).sum(1)``
                  in the model dtype on the model device.
        """
        rows = [prompt.x_vector.to(device=self.device, dtype=self.dtype).reshape(1, -1)]
        if prompt.ref_codes is not None:
            codes = prompt.ref_codes.to(device=self.device, dtype=torch.long)
            if int(codes[:, 0].max()) >= self.vocab_size or int(codes.min()) < 0:
                raise ValueError("ref_codes contain ids outside the codec vocabulary")
            pieces = [self._codec_embedding(codes[:, :1])]
            for group in range(1, NUM_CODE_GROUPS):
                embedding = self._frame_embeddings[group - 1]
                if int(codes[:, group].max()) >= embedding.num_embeddings:
                    raise ValueError(f"ref_codes group {group} exceeds its codebook size")
                pieces.append(embedding(codes[:, group : group + 1]))
            rows.append(torch.cat(pieces, dim=1).sum(1))
        return torch.cat(rows, dim=0)

    def _evict_locked(self, voice_id: str) -> None:
        resident = self._voices.pop(voice_id)
        self._allocator.free(resident.start, resident.rows)

    def _evict_lru_locked(self) -> bool:
        idle = [voice for voice in self._voices.values() if voice.pins == 0]
        if not idle:
            return False
        victim = min(idle, key=lambda voice: voice.last_used)
        self._evict_locked(victim.prompt.voice_id)
        return True

    @staticmethod
    def _info(resident: _ResidentVoice) -> VoiceInfo:
        return VoiceInfo(
            voice_id=resident.prompt.voice_id,
            supports_icl=resident.prompt.supports_icl,
            frames=resident.prompt.frames,
            rows=resident.rows,
            pins=resident.pins,
        )


__all__ = [
    "PROMPT_FORMAT_VERSION",
    "VoiceCapacityError",
    "VoiceInfo",
    "VoiceModeError",
    "VoiceNotFoundError",
    "VoicePrompt",
    "VoiceRegistry",
]
