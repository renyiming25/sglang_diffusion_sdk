# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **sglang diffusion video generation wrapper** that adapts Wan-AI video generation models (Wan2.2-T2V-A14B-Diffusers for text-to-video, Wan2.2-I2V-A14B-Diffusers for image-to-video) to the aiges WrapperBase plugin framework. The wrapper spawns an `sglang serve` subprocess and provides streaming video generation via task creation + polling.

## Key Commands

### Self-test (without aiges framework)
```bash
# 1. Start sglang server manually
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --port 30010

# 2. Run self-test script
python src/scripts/sglang_video_selftest.py --base-url http://127.0.0.1:30010/v1 --prompt "A calico cat playing a piano on stage" --size 832x480 --out out.mp4
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## Architecture

```
src/ailab/
├── utils/wrapper.py              # Entry point, delegates to inference_wrapper
├── inference_wrapper/
│   ├── wrapper.py                # Router: selects wrapper based on MODEL_TASK_TYPE env
│   └── diffusers/
│       ├── video_diffusion/wrapper.py    # Main video generation (T2V/I2V)
│       ├── image_diffusion/wrapper.py    # Image generation
│       └── audio_diffusion/wrapper.py    # Audio generation
```

**Request flow:**
1. `wrapperInit()` spawns `sglang serve` subprocess, waits for `/health` ready
2. `wrapperCreate()` creates session, returns handle
3. `wrapperWrite()` receives streaming request chunks (DataBegin/DataContinue/DataEnd)
4. On DataEnd, task queued to thread pool
5. Worker thread: creates video task via `/v1/videos`, polls `/v1/videos/{id}`, fetches content
6. Results streamed back via `callback()` with DataBegin/DataContinue/DataEnd

## Environment Variables

**Required:**
- `SGLANG_MODEL_PATH` - Model path for sglang serve
- `PRETRAINED_MODEL_NAME` - Model name (used by router)
- `MODEL_TASK_TYPE` - Task type: `video_generation`, `image_generation`, `audio_generation`, `third_video_generation`

**Optional:**
- `SGLANG_MODEL_NAME` - Display name for model
- `SGLANG_CMD_EXTRA_ARGS` - Extra args for sglang serve (e.g., `--num-gpus 4 --text-encoder-cpu-offload`)
- `SGLANG_READY_TIMEOUT_S` - Server ready timeout (default: 600s)
- `POLL_INTERVAL_MS` - Polling interval (default: 1500ms)
- `POLL_TIMEOUT_S` - Task timeout (default: 900s)
- `MAX_VIDEO_BYTES` - Max video size (default: 100MB)
- `THREAD_POOL_SIZE` - Worker threads (default: 1)

**I2V image input security:**
- `IMAGE_URL_ALLOWLIST` - Allowed hosts for image URLs (comma-separated)
- `MAX_IMAGE_BYTES` - Max image size (default: 10MB)
- `IMAGE_PATH_ALLOW_PREFIX` - Allowed local path prefix

## Key Patterns

### Streaming Response (aiges callback)
- `DataBegin` (0): First chunk with video_id and initial status
- `DataContinue` (1): Progress updates during polling
- `DataEnd` (2): Final chunk with video binary or error

### Error Codes
| Code | Meaning |
|------|---------|
| 1001 | Downstream call failed (videos.create/retrieve) |
| 1002 | Invalid parameters (missing prompt, bad size) |
| 1003 | sglang server not ready |
| 1004 | Poll timeout |
| 1005 | Video too large |
| 1006 | Image read failed (URL/path/binary) |

### Thread Pool Model
Each worker thread runs its own asyncio event loop. Tasks are queued per-thread, with load balancing via `alloc_min_thread()` selecting the thread with fewest queued tasks.

## Reference Implementation

`refer/wrapper_text_generate.py` shows the text generation wrapper pattern (vLLM + OpenAI SDK) that the video wrapper follows for subprocess management, health checks, and streaming callbacks.
