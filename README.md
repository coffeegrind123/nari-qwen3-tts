# Nari Qwen3-TTS

> [!IMPORTANT]
> **About this fork:** this fork adds an experimental, opt-in consumer-GPU
> profile for people who want to run Nari Qwen3-TTS on NVIDIA hardware other
> than an H100. The upstream H100/FP8 path remains the default and is unchanged.
> The portable path keeps the Talker projection in BF16 and reduces the CUDA
> Graph, KV-cache, and FlashInfer workspace footprint for single-request use.
>
> It has been validated on an RTX 5060 Ti 16 GB (SM 120), where the resident
> service used about 6.6 GiB of VRAM and reached first audible audio in 47 ms
> median / 57 ms p95 over a 90-utterance local panel. These measurements do not
> claim upstream's H100 throughput. See the reproducible
> [Model Skyline comparison](https://github.com/bglusman/model_skyline/blob/main/examples/voice-runtime-frontiers/README.md#what-the-matched-winner-comparison-adds).
>
> On top of that, this fork serves the **Qwen3-TTS 1.7B Base** checkpoint with
> **voice cloning** (x-vector and in-context reference prompts) through the same
> CUDA Graph engine, and adds an `ada24g` profile for 24 GB Ada GPUs. See
> [Voice cloning](#voice-cloning-base-checkpoint).

## TL;DR

Nari Qwen3-TTS is a high-performance, single-H100 serving implementation of
[Qwen3-TTS 1.7B CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice).
It exposes streaming and non-streaming speech generation over HTTP, plus
WebSocket-based input streaming for incremental text input.

It achieves **10 requests per second (RPS)** and **sub-50 ms p95
time-to-first-audio (TTFA)** while maintaining real-time playback on a **single
NVIDIA H100 SXM**. Even at **20 RPS**, it sustains **sub-80 ms p95 TTFA**.

Below is a performance comparison with popular serving engines.

![p95 TTFA under load comparison](docs/assets/p95-ttfa-under-load.png)

> [!NOTE]
> **Methodology**
>
> Read our [blog post](https://nari-labs.com/blog/qwen3-tts-speed-cost-frontier/)
> for the full methodology. For details on the benchmark, see the
> [benchmark repository](https://github.com/nari-labs/benchmarks).

## Run with Docker

The default production configuration requires an NVIDIA H100, the NVIDIA
Container Toolkit, and a driver compatible with CUDA 13.0. The published image
supports Linux x86_64 (`linux/amd64`) NVIDIA GPU hosts. The engine has been
tested only with English as the primary language.

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e HF_TOKEN \
  -e QWEN3_TTS_PROFILE=ttfa \
  -v nari-qwen3-tts-cache:/home/nari/.cache \
  ghcr.io/nari-labs/nari-qwen3-tts:latest
```

### Consumer GPUs (experimental)

The `consumer` profile provides a single-request, memory-reduced capture
surface for CUDA GPUs with roughly 16 GB of VRAM. It keeps BF16 Talker weights
on devices that cannot run the Hopper-only FP8 GEMM, uses one Talker capture
slot, and reduces the KV cache and FlashInfer workspace. This is an explicit
opt-in: omitting `--allow-non-h100` preserves the upstream H100 requirement.
Start it with `uv`:

```bash
nari-qwen3-tts-server \
  --profile consumer \
  --allow-non-h100
```

Or, after building this tree as `nari-qwen3-tts:local`, pass the opt-in through
the container entrypoint:

```bash
docker run --rm --gpus all \
  -p 8000:8000 \
  -e QWEN3_TTS_PROFILE=consumer \
  -v nari-qwen3-tts-cache:/home/nari/.cache \
  nari-qwen3-tts:local \
  --allow-non-h100
```

This compatibility path does not claim the H100 latency or throughput numbers.
The packaged H100 profiles and their strict FP8 requirement are unchanged.

Model files and compiled kernels are cached in the named volume. Once model
loading and CUDA Graph capture finish, check readiness with:

```bash
curl --fail http://127.0.0.1:8000/ready
```

To build the image locally instead:

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t nari-qwen3-tts:local .
```

## Run with uv

Python and uv versions are pinned in `.python-version` and `pyproject.toml`.
The checked-in `uv.lock` defines the complete environment.

Install the required system packages (Debian/Ubuntu):

```bash
sudo apt-get install -y build-essential libsndfile1 sox
```

```bash
uv sync --frozen --extra codec --extra cuda --extra serving
uv run --frozen nari-qwen3-tts-server --profile ttfa
```

Add `--local-files-only` after the model is cached to prevent downloads at
startup. `--frozen` is intentional: an out-of-date lockfile fails instead of
silently resolving a different CUDA or Python environment.

The distribution name uses hyphens, while Python imports use underscores:

```python
from nari_qwen3_tts import ModelAssetConfig, open_model
```

## Profiles

- `ttfa`: prioritizes time to first audio with latency-oriented scheduling and
  smaller initial Codec chunks.
- `balanced`: the default container profile, balancing first-audio latency and
  sustained request throughput.
- `throughput`: uses larger Codec chunks and batches to prioritize aggregate
  throughput under load.
- `consumer`: limits execution to one GPU row and a reduced capture/cache
  surface for experimental use on roughly 16 GB CUDA GPUs.
- `ada24g`: 24 GB Ada GPUs (RTX 4090 class) with `--allow-non-h100`: BF16
  Talker, batches up to 16, prefill buckets up to 1024 tokens for clone
  prompts, and a KV budget that leaves VRAM for other processes.

Select a Docker profile with `QWEN3_TTS_PROFILE`, or pass `--profile` to
`nari-qwen3-tts-server`. You can also set `QWEN3_TTS_MODEL_CACHE_DIR` and
`QWEN3_TTS_LOCAL_FILES_ONLY=1` in Docker.

For advanced tuning, apply a strict partial YAML overlay:

```bash
nari-qwen3-tts-server \
  --local-files-only \
  --engine-config /path/to/engine.yaml
```

The overlay may name its packaged base with `extends: ttfa`, `balanced`,
`throughput`, or `consumer`; alternatively, pass `--profile` and omit `extends`.
Unknown keys, invalid capture lists, and profile/base mismatches fail before
model loading. The fully resolved config and its SHA-256 are printed at
startup.

## Voice cloning (Base checkpoint)

Serve `Qwen/Qwen3-TTS-12Hz-1.7B-Base` instead of CustomVoice and register
voices from reference audio:

```bash
nari-qwen3-tts-server --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --revision fd4b254389122332181a7c3db7f27e918eec64e3 \
  --allow-non-h100 --profile ada24g
```

- `POST /v1/voices` `{"voice_id", "audio_base64", "ref_text"?, "x_vector_only"?, "xvector_audio_base64"?}`
  builds a prompt with the checkpoint's own speaker encoder and 12 Hz speech
  tokenizer, registers it, and returns it as `prompt_base64` (safetensors).
- `PUT /v1/voices/{voice_id}` re-registers a stored prompt (the engine keeps
  prompts in GPU memory only; persist `prompt_base64` yourself).
- `GET /v1/voices`, `GET/DELETE /v1/voices/{voice_id}`.
- `POST /v1/audio/speech` with `"voice": "<voice_id>"` and optional
  `"voice_mode": "auto" | "icl" | "xvector"`. `auto` uses in-context
  conditioning when the voice has a transcript, x-vector otherwise. Live
  (WebSocket) input falls back to x-vector, because ICL places target text
  alongside the reference frames in prefill. `instruct` is rejected: the Base
  model has no instruction control.

How it works: every clone input the official implementation passes as a float
vector (the x-vector, and the summed 16-codebook embedding of each reference
frame) is written into extra rows of an extended codec embedding table. The
prompt then stays a sequence of integer ids, so the captured prefill CUDA
Graphs run unchanged. `tests/voice_clone/` checks the resulting embeddings
against a transcription of `qwen-tts` 0.1.1's `generate` /
`generate_icl_prompt` for both input modes.

## API and architecture

The service exposes:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/audio/speech`
- `WS /v1/audio/speech/ws`

See [WebSocket speech API](docs/websocket.md) for the live-text protocol and
client example.

### HTTP client example

`POST /v1/audio/speech` follows the OpenAI Audio Speech request shape. Nari Qwen3-TTS
supports only the model listed above, `wav` and `pcm` output, and `speed: 1.0`.
It also accepts Nari-specific controls such as `language`.

```bash
curl http://127.0.0.1:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from Nari Labs.",
    "voice": "ryan",
    "language": "english",
    "response_format": "wav",
    "stream": false
  }' \
  --output speech.wav
```

Readiness remains false until CUDA Graph capture and a warm-up TTS request have
both completed.

## Thanks and references

Built by [Nari Labs](https://nari-labs.com). Thanks to the Qwen3-TTS authors
for releasing Qwen3-TTS, and to these projects for their work on
high-performance multimodal and speech serving:

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)
- [SGLang-Omni](https://github.com/sgl-project/sglang-omni)
- [VoxServe](https://github.com/vox-serve/vox-serve)
- [M*](https://github.com/mstar-project/mstar)
