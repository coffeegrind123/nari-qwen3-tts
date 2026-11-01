"""Clone prompts built from integer layouts must equal qwen-tts 0.1.1 embeddings.

``_official_prompt`` is a line-for-line transcription of the prompt section of
``Qwen3TTSForConditionalGeneration.generate`` and ``generate_icl_prompt``
(qwen-tts 0.1.1), with ``text_projection(text_embedding(ids))`` replaced by a
lookup into a precomputed projected table (the engine's projected text cache).
Everything runs in float32 on CPU so any mismatch is a layout bug, not noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nari_qwen3_tts.contract.request import VoiceCloneMode
from nari_qwen3_tts.model.input_layout import (
    ResolvedVoice,
    build_batched_talker_token_layout,
    resolve_preset_voice,
)
from nari_qwen3_tts.model.voices import NUM_CODE_GROUPS, VoicePrompt, VoiceRegistry

HIDDEN = 8
TEXT_VOCAB = 120
CODEC_VOCAB = 96
CODEBOOK = 40


def _config(*, with_preset: bool = False):
    talker = SimpleNamespace(
        spk_id={"aiden": 70} if with_preset else {},
        spk_is_dialect={"aiden": False} if with_preset else {},
        codec_language_id={"english": 60, "chinese": 61},
        codec_think_id=80,
        codec_nothink_id=81,
        codec_think_bos_id=82,
        codec_think_eos_id=83,
        codec_pad_id=84,
        codec_bos_id=85,
        codec_eos_token_id=86,
        hidden_size=HIDDEN,
        vocab_size=CODEC_VOCAB,
        num_code_groups=NUM_CODE_GROUPS,
    )
    return SimpleNamespace(
        tts_bos_token_id=110,
        tts_eos_token_id=111,
        tts_pad_token_id=112,
        talker=talker,
        talker_config=talker,
    )


class _Weights:
    def __init__(self, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.projected_text = torch.randn(TEXT_VOCAB, HIDDEN, generator=generator)
        self.codec = torch.nn.Embedding(CODEC_VOCAB, HIDDEN)
        self.frames = [torch.nn.Embedding(CODEBOOK, HIDDEN) for _ in range(NUM_CODE_GROUPS - 1)]
        with torch.no_grad():
            self.codec.weight.copy_(torch.randn(CODEC_VOCAB, HIDDEN, generator=generator))
            for embedding in self.frames:
                embedding.weight.copy_(torch.randn(CODEBOOK, HIDDEN, generator=generator))

    def text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.projected_text[ids]


def _official_prompt(  # noqa: PLR0912,PLR0913 - mirrors the upstream code path for path
    weights: _Weights,
    config,
    *,
    input_id: torch.Tensor,
    language: str,
    speaker_embed: torch.Tensor | None,
    non_streaming_mode: bool,
    ref_id: torch.Tensor | None = None,
    ref_code: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (talker_input_embed [T, D], trailing_text_hidden [L, D]) for one row."""
    talker_config = config.talker_config
    input_id = input_id.unsqueeze(0)
    emb = weights.codec

    language_id = None if language == "auto" else talker_config.codec_language_id[language]
    tts_bos_embed, tts_eos_embed, tts_pad_embed = weights.text(
        torch.tensor([[config.tts_bos_token_id, config.tts_eos_token_id, config.tts_pad_token_id]])
    ).chunk(3, dim=1)
    if language_id is None:
        codec_prefill_list = [[talker_config.codec_nothink_id, talker_config.codec_think_bos_id,
                               talker_config.codec_think_eos_id]]
    else:
        codec_prefill_list = [[talker_config.codec_think_id, talker_config.codec_think_bos_id, language_id,
                               talker_config.codec_think_eos_id]]
    codec_input_emebdding_0 = emb(torch.tensor(codec_prefill_list))
    codec_input_emebdding_1 = emb(torch.tensor([[talker_config.codec_pad_id, talker_config.codec_bos_id]]))
    if speaker_embed is None:
        codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
    else:
        codec_input_emebdding = torch.cat(
            [codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1
        )
    _talker_input_embed_role = weights.text(input_id[:, :3])
    _talker_input_embed = torch.cat(
        (tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1
    ) + codec_input_emebdding[:, :-1]
    talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

    if ref_code is not None:
        text_id = input_id[:, 3:-5]
        text_embed = weights.text(torch.cat([ref_id.unsqueeze(0), text_id], dim=-1))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        codec_embed = []
        for i in range(NUM_CODE_GROUPS):
            if i == 0:
                codec_embed.append(emb(ref_code[:, :1]))
            else:
                codec_embed.append(weights.frames[i - 1](ref_code[:, i:i + 1]))
        codec_embed = torch.cat(codec_embed, dim=1).sum(1).unsqueeze(0)
        codec_embed = torch.cat([emb(torch.tensor([[talker_config.codec_bos_id]])), codec_embed], dim=1)
        text_lens = text_embed.shape[1]
        codec_lens = codec_embed.shape[1]
        if non_streaming_mode:
            icl_input_embed = text_embed + emb(torch.tensor([[talker_config.codec_pad_id] * text_lens]))
            icl_input_embed = torch.cat([icl_input_embed, codec_embed + tts_pad_embed], dim=1)
            trailing = tts_pad_embed
        elif text_lens > codec_lens:
            icl_input_embed, trailing = text_embed[:, :codec_lens] + codec_embed, text_embed[:, codec_lens:]
        else:
            text_embed = torch.cat([text_embed] + [tts_pad_embed] * (codec_lens - text_lens), dim=1)
            icl_input_embed, trailing = text_embed + codec_embed, tts_pad_embed
        talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
    else:
        talker_input_embed = torch.cat(
            [talker_input_embed, weights.text(input_id[:, 3:4]) + codec_input_emebdding[:, -1:]], dim=1
        )
        if non_streaming_mode:
            talker_input_embed = talker_input_embed[:, :-1]
            talker_input_embed = torch.cat(
                [
                    talker_input_embed,
                    torch.cat((weights.text(input_id[:, 3:-5]), tts_eos_embed), dim=1)
                    + emb(torch.tensor([[talker_config.codec_pad_id] * (input_id[:, 3:-5].shape[1] + 1)])),
                    tts_pad_embed + emb(torch.tensor([[talker_config.codec_bos_id]])),
                ],
                dim=1,
            )
            trailing = tts_pad_embed
        else:
            trailing = torch.cat((weights.text(input_id[:, 4:-5]), tts_eos_embed), dim=1)
    return talker_input_embed[0], trailing[0]


def _engine_prompt(weights, table, config, *, text, voice, language, non_streaming_mode):
    layout = build_batched_talker_token_layout(
        config,
        [text],
        [torch.empty(0, dtype=torch.long)],
        languages=[language],
        voices=[voice],
        non_streaming_modes=[non_streaming_mode],
    )
    text_embed = weights.projected_text[layout.text_token_ids]
    codec_embed = torch.nn.functional.embedding(layout.codec_token_ids, table)
    prompt = torch.where(layout.codec_token_mask.unsqueeze(1), text_embed + codec_embed, text_embed)
    trailing = weights.projected_text[layout.trailing_text_ids[0]]
    assert layout.seq_lens == [prompt.shape[0]]
    return prompt, trailing


def _registry(weights, *, rows: int = 512) -> VoiceRegistry:
    return VoiceRegistry(
        codec_embedding=weights.codec,
        frame_embeddings=weights.frames,
        voice_rows=rows,
        tokenize_ref_text=lambda text: torch.tensor([int(piece) for piece in text.split()], dtype=torch.long),
    )


def _wrapped(target: list[int]) -> torch.Tensor:
    # role(3) + target + 5 suffix tokens, like "<|im_start|>assistant\n{t}<|im_end|>\n<|im_start|>assistant\n"
    return torch.tensor([1, 2, 3, *target, 4, 5, 6, 7, 8], dtype=torch.long)


def _codes(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(0, CODEC_VOCAB - 16, (frames, 1), generator=generator)
    rest = torch.randint(0, CODEBOOK, (frames, NUM_CODE_GROUPS - 1), generator=generator)
    return torch.cat([first, rest], dim=1)


REF_TEXT = "20 21 22 23"


@pytest.mark.parametrize("non_streaming_mode", [True, False])
@pytest.mark.parametrize("language", ["auto", "english"])
@pytest.mark.parametrize(
    ("target_tokens", "frames"),
    [
        (list(range(30, 33)), 12),  # codec longer than text: streaming pads text
        (list(range(30, 60)), 5),  # text longer than codec: streaming trails text
        (list(range(30, 39)), 13),  # exactly equal lengths: 4 + 9 + 1 == 13 + 1
    ],
)
def test_icl_prompt_matches_official(non_streaming_mode, language, target_tokens, frames) -> None:
    config = _config()
    weights = _Weights()
    registry = _registry(weights)
    codes = _codes(frames, seed=frames)
    x_vector = torch.randn(HIDDEN)
    registry.register(VoicePrompt(voice_id="clone", x_vector=x_vector, ref_codes=codes, ref_text=REF_TEXT))
    voice = registry.acquire("clone", mode=VoiceCloneMode.ICL, live=False)
    text = _wrapped(target_tokens)

    expected, expected_trailing = _official_prompt(
        weights,
        config,
        input_id=text,
        language=language,
        speaker_embed=x_vector,
        non_streaming_mode=non_streaming_mode,
        ref_id=torch.tensor([20, 21, 22, 23]),
        ref_code=codes,
    )
    actual, actual_trailing = _engine_prompt(
        weights, registry.table, config,
        text=text, voice=voice, language=language, non_streaming_mode=non_streaming_mode,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_trailing, expected_trailing, rtol=0, atol=1e-6)


@pytest.mark.parametrize("non_streaming_mode", [True, False])
def test_xvector_prompt_matches_official(non_streaming_mode) -> None:
    config = _config()
    weights = _Weights(seed=3)
    registry = _registry(weights)
    x_vector = torch.randn(HIDDEN)
    registry.register(VoicePrompt(voice_id="xv", x_vector=x_vector))
    voice = registry.acquire("xv", mode=VoiceCloneMode.AUTO, live=False)
    assert voice.icl is None
    text = _wrapped(list(range(30, 41)))

    expected, expected_trailing = _official_prompt(
        weights, config, input_id=text, language="english",
        speaker_embed=x_vector, non_streaming_mode=non_streaming_mode,
    )
    actual, actual_trailing = _engine_prompt(
        weights, registry.table, config,
        text=text, voice=voice, language="english", non_streaming_mode=non_streaming_mode,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_trailing, expected_trailing, rtol=0, atol=1e-6)


@pytest.mark.parametrize("non_streaming_mode", [True, False])
def test_preset_prompt_still_matches_official(non_streaming_mode) -> None:
    config = _config(with_preset=True)
    weights = _Weights(seed=5)
    text = _wrapped(list(range(30, 41)))
    expected, expected_trailing = _official_prompt(
        weights, config, input_id=text, language="english",
        speaker_embed=weights.codec(torch.tensor(70)), non_streaming_mode=non_streaming_mode,
    )
    actual, actual_trailing = _engine_prompt(
        weights, weights.codec.weight, config,
        text=text, voice=resolve_preset_voice(config, "aiden"), language="english",
        non_streaming_mode=non_streaming_mode,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_trailing, expected_trailing, rtol=0, atol=1e-6)


def test_dialect_voice_cannot_carry_icl() -> None:
    with pytest.raises(ValueError, match="dialect"):
        ResolvedVoice(speaker_codec_id=1, dialect="beijing_dialect", icl=object())  # type: ignore[arg-type]
