"""Input construction helpers for the two Qwen3-TTS Talker modes.

Every Talker prefill position is an integer triple (text id, codec id, codec
mask). Voice-clone conditioning that the official implementation feeds as
float vectors (the x-vector and the summed reference-frame embeddings) is
expressed as codec ids >= the checkpoint vocabulary, pointing into the voice
rows of the engine's extended codec table (see ``model/voices.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class IclPrompt:
    """In-context reference: transcript tokens plus one codec row per frame.

    ``ref_text_ids`` is the official ``ref_ids[:, 3:-2]`` slice of
    ``<|im_start|>assistant\n{ref_text}<|im_end|>\n``. ``frame_codec_ids``
    are extended-table rows holding the summed 16-codebook embedding of each
    reference frame.
    """

    ref_text_ids: torch.Tensor
    frame_codec_ids: torch.Tensor

    def __post_init__(self) -> None:
        for name in ("ref_text_ids", "frame_codec_ids"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim != 1:
                raise TypeError(f"ICL {name} must be a 1-D torch.long tensor")
        if self.ref_text_ids.numel() < 1:
            raise ValueError("ICL reference transcript cannot be empty")
        if self.frame_codec_ids.numel() < 1:
            raise ValueError("ICL reference audio must contain at least one frame")


@dataclass(frozen=True, slots=True)
class ResolvedVoice:
    """A speaker resolved to its codec-prefix conditioning.

    ``speaker_codec_id`` is a preset ``spk_id`` or a registered x-vector row.
    ``dialect`` is the preset's dialect language (CustomVoice only).
    """

    speaker_codec_id: int
    dialect: str | None = None
    icl: IclPrompt | None = None

    def __post_init__(self) -> None:
        if isinstance(self.speaker_codec_id, bool) or not isinstance(self.speaker_codec_id, int):
            raise TypeError("speaker_codec_id must be an integer")
        if self.speaker_codec_id < 0:
            raise ValueError("speaker_codec_id must be non-negative")
        if self.dialect is not None and self.icl is not None:
            raise ValueError("dialect presets cannot carry an ICL reference")


def resolve_preset_voice(config, speaker: str) -> ResolvedVoice:
    """Resolve a CustomVoice preset speaker name."""
    talker_config = config.talker_config
    speaker_key = speaker.lower()
    if speaker_key not in talker_config.spk_id:
        raise ValueError(f"Unsupported Qwen3-TTS speaker: {speaker!r}")
    dialect = talker_config.spk_is_dialect[speaker_key]
    return ResolvedVoice(
        speaker_codec_id=talker_config.spk_id[speaker_key],
        dialect=dialect if dialect else None,
    )


@dataclass(frozen=True)
class PackedTalkerTokenLayout:
    """Integer Talker prefill layout plus request-local continuation data."""

    text_token_ids: torch.Tensor
    codec_token_ids: torch.Tensor
    codec_token_mask: torch.Tensor
    seq_lens: list[int]
    trailing_text_ids: list[torch.Tensor]
    tts_pad_id: torch.Tensor


def _codec_prefix(
    config,
    *,
    language: str,
    voice: ResolvedVoice,
) -> list[int]:
    """Resolve request metadata to the official think/language codec prefix."""
    talker_config = config.talker_config
    language_key = language.lower()
    if language_key == "auto":
        language_id = None
    elif language_key in talker_config.codec_language_id:
        language_id = talker_config.codec_language_id[language_key]
    else:
        raise ValueError(f"Unsupported Qwen3-TTS language: {language!r}")

    if language_key in {"chinese", "auto"} and voice.dialect is not None:
        language_id = talker_config.codec_language_id[voice.dialect]

    if language_id is None:
        codec_prefix = [
            talker_config.codec_nothink_id,
            talker_config.codec_think_bos_id,
            talker_config.codec_think_eos_id,
        ]
    else:
        codec_prefix = [
            talker_config.codec_think_id,
            talker_config.codec_think_bos_id,
            language_id,
            talker_config.codec_think_eos_id,
        ]
    return codec_prefix


def talker_input_seq_len(
    config,
    *,
    text_token_count: int,
    instruct_token_count: int,
    language: str,
    voice: ResolvedVoice,
    non_streaming_mode: bool,
) -> int:
    """Return the prefill length without launching tensor operations."""
    codec_prefix = _codec_prefix(config, language=language, voice=voice)
    role_token_count = min(text_token_count, 3)
    prefix_token_count = len(codec_prefix) + 2
    target_token_count = max(0, text_token_count - 8)
    if voice.icl is not None:
        text_part = voice.icl.ref_text_ids.numel() + target_token_count + 1
        codec_part = voice.icl.frame_codec_ids.numel() + 1
        continuation_token_count = text_part + codec_part if non_streaming_mode else codec_part
    elif non_streaming_mode:
        continuation_token_count = target_token_count + 2
    else:
        continuation_token_count = int(text_token_count > 3)
    return instruct_token_count + role_token_count + prefix_token_count + continuation_token_count


def custom_voice_talker_input_seq_len(
    config,
    *,
    text_token_count: int,
    instruct_token_count: int,
    language: str,
    speaker: str,
    non_streaming_mode: bool,
) -> int:
    """Preset-speaker prefill length (kept for existing CustomVoice callers)."""
    return talker_input_seq_len(
        config,
        text_token_count=text_token_count,
        instruct_token_count=instruct_token_count,
        language=language,
        voice=resolve_preset_voice(config, speaker),
        non_streaming_mode=non_streaming_mode,
    )


def build_batched_custom_voice_talker_token_layout(
    config,
    text_inputs: list[torch.Tensor],
    instruct_inputs: list[torch.Tensor],
    *,
    languages: list[str],
    speakers: list[str],
    non_streaming_modes: list[bool],
) -> PackedTalkerTokenLayout:
    """Preset-speaker layout (kept for existing CustomVoice callers)."""
    return build_batched_talker_token_layout(
        config,
        text_inputs,
        instruct_inputs,
        languages=languages,
        voices=[resolve_preset_voice(config, speaker) for speaker in speakers],
        non_streaming_modes=non_streaming_modes,
    )


def build_batched_talker_token_layout(  # noqa: PLR0912,PLR0915 - one explicit fixed layout
    config,
    text_inputs: list[torch.Tensor],
    instruct_inputs: list[torch.Tensor],
    *,
    languages: list[str],
    voices: list[ResolvedVoice],
    non_streaming_modes: list[bool],
) -> PackedTalkerTokenLayout:
    """Build the integer Talker layout consumed by captured prefill.

    Keeping embedding lookup out of this function lets the Talker CUDA Graph
    capture text/codec lookup and addition together with transformer prefill.
    Trailing text is returned separately so callers can materialize it in one
    batched lookup without copying the full prefill embedding tensor.

    Rows with ``voice.icl`` follow ``Qwen3TTSForConditionalGeneration.
    generate_icl_prompt`` from qwen-tts 0.1.1 position for position.
    """
    batch_size = len(text_inputs)
    if batch_size == 0:
        raise ValueError("Qwen3-TTS Talker prefill batch cannot be empty")
    if not (len(instruct_inputs) == len(languages) == len(voices) == len(non_streaming_modes) == batch_size):
        raise ValueError("Qwen3-TTS Talker prefill batch fields must have equal lengths")
    talker_config = config.talker_config
    text_dtype = text_inputs[0].dtype
    device = text_inputs[0].device
    special_text_ids = torch.tensor(
        [
            config.tts_bos_token_id,
            config.tts_eos_token_id,
            config.tts_pad_token_id,
        ],
        dtype=text_dtype,
        device=device,
    )
    tts_bos_id = special_text_ids[0:1]
    tts_eos_id = special_text_ids[1:2]
    tts_pad_id = special_text_ids[2:3]

    prefill_text_pieces: list[torch.Tensor] = []
    prefill_codec_pieces: list[torch.Tensor] = []
    prefill_codec_ids: list[int] = []
    prefill_codec_mask: list[bool] = []
    trailing_text_ids: list[torch.Tensor] = []
    seq_lens: list[int] = []
    packed_length = 0

    def flush_codec_ids() -> None:
        if prefill_codec_ids:
            prefill_codec_pieces.append(torch.tensor(prefill_codec_ids, dtype=text_dtype, device=device))
            prefill_codec_ids.clear()

    def append_codec_tensor(values: torch.Tensor) -> None:
        flush_codec_ids()
        prefill_codec_pieces.append(values.to(device=device, dtype=text_dtype))
        prefill_codec_mask.extend([True] * values.numel())

    for raw_text, raw_instruct, language, voice, non_streaming_mode in zip(
        text_inputs,
        instruct_inputs,
        languages,
        voices,
        non_streaming_modes,
        strict=True,
    ):
        text = raw_text.reshape(-1)
        instruct = raw_instruct.reshape(-1)
        codec_prefix = _codec_prefix(config, language=language, voice=voice)

        if instruct.numel():
            prefill_text_pieces.append(instruct)
            prefill_codec_ids.extend([talker_config.codec_pad_id] * instruct.numel())
            prefill_codec_mask.extend([False] * instruct.numel())
        role_text = text[:3]
        if role_text.numel():
            prefill_text_pieces.append(role_text)
            prefill_codec_ids.extend([talker_config.codec_pad_id] * role_text.numel())
            prefill_codec_mask.extend([False] * role_text.numel())

        # Prefix text is PAD for every codec conditioning token except the
        # final slot, where the official layout uses text BOS.
        prefill_text_pieces.append(tts_pad_id.expand(len(codec_prefix) + 1))
        prefill_text_pieces.append(tts_bos_id)
        request_codec_ids = [
            *codec_prefix,
            voice.speaker_codec_id,
            talker_config.codec_pad_id,
        ]
        prefill_codec_ids.extend(request_codec_ids)
        prefill_codec_mask.extend([True] * len(request_codec_ids))

        target_text = text[3:-5]
        if voice.icl is not None:
            # text_part = ref + target + EOS; codec_part = codec BOS + frame rows.
            text_part = torch.cat(
                [voice.icl.ref_text_ids.to(device=device, dtype=text_dtype), target_text, tts_eos_id]
            )
            codec_part = torch.cat(
                [
                    torch.tensor([talker_config.codec_bos_id], dtype=text_dtype, device=device),
                    voice.icl.frame_codec_ids.to(device=device, dtype=text_dtype),
                ]
            )
            text_length = text_part.numel()
            codec_length = codec_part.numel()
            if non_streaming_mode:
                prefill_text_pieces.append(text_part)
                prefill_codec_ids.extend([talker_config.codec_pad_id] * text_length)
                prefill_codec_mask.extend([True] * text_length)
                prefill_text_pieces.append(tts_pad_id.expand(codec_length))
                append_codec_tensor(codec_part)
                trailing_text_ids.append(tts_pad_id)
            elif text_length > codec_length:
                prefill_text_pieces.append(text_part[:codec_length])
                append_codec_tensor(codec_part)
                trailing_text_ids.append(text_part[codec_length:])
            else:
                prefill_text_pieces.append(text_part)
                if codec_length > text_length:
                    prefill_text_pieces.append(tts_pad_id.expand(codec_length - text_length))
                append_codec_tensor(codec_part)
                trailing_text_ids.append(tts_pad_id)
        elif non_streaming_mode:
            if target_text.numel():
                prefill_text_pieces.append(target_text)
            prefill_text_pieces.extend([tts_eos_id, tts_pad_id])
            continuation_codec_ids = [talker_config.codec_pad_id] * (target_text.numel() + 1)
            continuation_codec_ids.append(talker_config.codec_bos_id)
            prefill_codec_ids.extend(continuation_codec_ids)
            prefill_codec_mask.extend([True] * len(continuation_codec_ids))
            trailing_text_ids.append(tts_pad_id)
        else:
            first_text = text[3:4]
            if first_text.numel():
                prefill_text_pieces.append(first_text)
                prefill_codec_ids.append(talker_config.codec_bos_id)
                prefill_codec_mask.append(True)

            trailing_text = text[4:-5]
            trailing_text_ids.append(torch.cat([trailing_text, tts_eos_id], dim=0))

        seq_len = talker_input_seq_len(
            config,
            text_token_count=text.numel(),
            instruct_token_count=instruct.numel(),
            language=language,
            voice=voice,
            non_streaming_mode=non_streaming_mode,
        )
        if len(prefill_codec_mask) != packed_length + seq_len:
            raise RuntimeError("Qwen3-TTS Talker prefill layout length mismatch")
        seq_lens.append(seq_len)
        packed_length += seq_len

    flush_codec_ids()
    packed_text_ids = torch.cat(prefill_text_pieces, dim=0)
    packed_codec_ids = torch.cat(prefill_codec_pieces, dim=0)
    packed_codec_mask = torch.tensor(
        prefill_codec_mask,
        dtype=torch.bool,
        device=device,
    )
    if not (packed_text_ids.shape == packed_codec_ids.shape == packed_codec_mask.shape):
        raise RuntimeError("Qwen3-TTS packed prefill fields have mismatched lengths")
    if packed_codec_ids.numel() != packed_length:
        raise RuntimeError("Qwen3-TTS Talker prefill layout length mismatch")
    return PackedTalkerTokenLayout(
        text_token_ids=packed_text_ids,
        codec_token_ids=packed_codec_ids,
        codec_token_mask=packed_codec_mask,
        seq_lens=seq_lens,
        trailing_text_ids=trailing_text_ids,
        tts_pad_id=tts_pad_id,
    )
