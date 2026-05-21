# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sglang diffusion wrapper that adapts Wan-AI diffusion models to the aiges WrapperBase plugin framework. Supports video generation (T2V/I2V), image generation (T2I/I2I), and audio generation. The wrapper spawns an `sglang serve` subprocess and provides generation via task creation + polling.

## Key Commands

### Build
```bash
make build          # poetry build + install from dist
make publish        # publish to PyPI
make publish-custom # publish to custom repo
```

### Install dependencies
```bash
pip install -r requirements.txt
pip install "sglang[diffusion]" --prerelease=allow
```

### Self-test (without aiges framework)
```bash
# 1. Start sglang server manually
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --port 30010

# 2. Run self-test script
python src/scripts/sglang_video_selftest.py --base-url http://127.0.0.1:30010/v1 --prompt "A calico cat playing a piano on stage" --size 832x480 --out out.mp4
```

### Plugin tests (with aiges framework, requires GPU + model)
```bash
# Video wrapper (wrapperRead sync mode) — T2V
python test/test_wrapper_plugin.py --mode t2v --config test/config_i2v.json --prompt "A cat playing piano"

# Video wrapper — I2V with image URL
python test/test_wrapper_plugin.py --mode i2v --config test/config_i2v.json --image-url https://example.com/cat.jpg

# Image wrapper (wrapperOnceExecAsync + callback mode) — T2I
python test/test_image_wrapper_plugin.py --mode t2i --config config.json --prompt "A cat playing piano"

# Direct API test (requires running sglang server)
python test/test_video_wrapper.py --base-url http://127.0.0.1:30010/v1
```

### Docker
```bash
docker build -t sglang-diffusion-sdk -f docker/Dockerfile .
```

## Architecture

```
src/ailab/
├── utils/wrapper.py                          # aiges entry point, delegates to inference_wrapper
├── inference_wrapper/
│   ├── wrapper.py                            # Router: selects wrapper by MODEL_TASK_TYPE env
│   └── diffusers/
│       ├── video_diffusion/wrapper.py        # T2V/I2V — wrapperCreate/Write/Read (session streaming)
│       ├── image_diffusion/wrapper.py        # T2I/I2I — wrapperOnceExecAsync + callback
│       ├── audio_diffusion/wrapper.py        # Audio generation
│       └── third_diffusion/                  # Third-party video generation
```

### Two execution modes

**Video wrapper** uses session-based streaming:
1. `wrapperCreate()` → session handle
2. `wrapperWrite(handle, DataEnd)` → queues task to thread pool
3. `wrapperRead(handle)` → polls for results (DataBegin/DataContinue/DataEnd)
4. `wrapperDestroy(handle)` → cleanup

**Image wrapper** uses one-shot async:
1. `wrapperOnceExecAsync(params, reqData, usrTag)` → submits task
2. Result delivered via `callback(Response, usrTag)`

### Common init flow (both wrappers)
1. `wrapperInit(config)` picks a free port, spawns `sglang serve` subprocess
2. Polls `/health` until ready (timeout from `SGLANG_READY_TIMEOUT_S`)
3. Writes port to `MAAS_PORT_FILE` for service discovery
4. Creates OpenAI client pointing at `http://127.0.0.1:{port}/v1`

### Thread Pool Model
Each worker thread runs its own asyncio event loop. Tasks are queued per-thread, with load balancing via `alloc_min_thread()` selecting the thread with fewest queued tasks.

## Environment Variables

**Required:**
- `FULL_MODEL_PATH` - Model path for sglang serve (also aliased as `SGLANG_MODEL_PATH`)
- `PRETRAINED_MODEL_NAME` - Model name (used by router to select wrapper)
- `MODEL_TASK_TYPE` - `video_generation`, `image_generation`, `audio_generation`, `third_video_generation`

**Config dict (passed to wrapperInit):**
- `modelName` - Display model name (e.g., `wan2.2-t2v-14b`)
- `modelTaskType` - Sub-task: `t2v`, `i2v`, `t2i`, `i2i`
- `supportedResolutions` - Map of model name → resolution list (e.g., `["480P", "720P"]`)

**Optional env:**
- `SGLANG_CMD_EXTRA_ARGS` - Extra args for sglang serve (e.g., `--num-gpus 4 --text-encoder-cpu-offload`)
- `SGLANG_READY_TIMEOUT_S` - Server ready timeout (default: 600s)
- `MAAS_PORT_FILE` - Port file path (default: `/home/aiges/maas_port`)
- `POLL_INTERVAL_MS` - Polling interval (default: 5000ms)
- `POLL_TIMEOUT_S` - Task timeout (default: 1800s)
- `THREAD_POOL_SIZE` - Worker threads (default: 1)
- `LOG_LEVEL` - Logging level (default: INFO)

**I2V/I2I image input security:**
- `IMAGE_URL_ALLOWLIST` - Allowed hosts for image URLs (comma-separated)
- `MAX_IMAGE_BYTES` - Max image size (default: 10MB)
- `IMAGE_PATH_ALLOW_PREFIX` - Allowed local path prefix

## Error Codes
| Code | Meaning |
|------|---------|
| 1001 | Downstream call failed (videos.create/retrieve/images) |
| 1002 | Invalid parameters (missing prompt, bad size) |
| 1003 | sglang server not ready |
| 1004 | Poll timeout |
| 1005 | Image/video too large or image read failed |

## Resolution Presets

Video: `480P` → 832x480, 480x832, 624x624 | `720P` → 1280x720, 720x1280, 960x960, 1088x832, 832x1088

Image: `512P` → 512x512, 768x512, 512x768 | `1024P` → 1024x1024, 1536x1024, 1024x1536, 1280x720, 720x1280

## Reference Implementation

`refer/wrapper_text_generate.py` shows the text generation wrapper pattern (vLLM + OpenAI SDK) that the diffusion wrappers follow for subprocess management, health checks, and streaming callbacks.
