# sglang-diffusion-sdk

基于 [sglang](https://github.com/sgl-project/sglang) 的 Diffusion 视频生成 Wrapper，对接 aiges `WrapperBase` 插件框架，支持 Wan-AI 系列模型的文生视频（T2V）与图生视频（I2V）推理。

## 支持模型

| 模型 | 任务类型 |
|------|----------|
| Wan-AI/Wan2.2-T2V-A14B-Diffusers | 文生视频 (T2V) |
| Wan-AI/Wan2.2-I2V-A14B-Diffusers | 图生视频 (I2V) |

**请求流程：**

1. `wrapperInit()` — 启动 `sglang serve` 子进程，等待 `/health` 就绪
2. `wrapperCreate()` — 创建会话，返回 handle
3. `wrapperWrite()` — 接收流式请求分片（DataBegin / DataContinue / DataEnd）
4. DataEnd 时任务入队线程池
5. Worker 线程：通过 `/v1/videos` 创建视频任务，轮询 `/v1/videos/{id}`，获取结果
6. 结果通过 `out_q` 推送，由 `wrapperRead()` 同步读取

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
pip install "sglang[diffusion]" --prerelease=allow
```

或使用 poetry 构建：

```bash
make build
```

### Docker 构建

```bash
docker build -t sglang-diffusion-sdk -f docker/Dockerfile .
```

### 自测（不依赖 aiges）

1. 启动 sglang 服务：

```bash
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --port 30010
```

2. 运行自测脚本：

```bash
python src/scripts/sglang_video_selftest.py \
  --base-url http://127.0.0.1:30010/v1 \
  --prompt "A calico cat playing a piano on stage" \
  --size 832x480 \
  --out out.mp4
```

### aiges 框架内运行

设置环境变量后由 aiges 框架加载，`wrapperInit()` 会自动启动 sglang 服务并等待就绪。

## 环境变量

### 必填

| 变量 | 说明 |
|------|------|
| `FULL_MODEL_PATH` | sglang serve 使用的模型路径 |
| `PRETRAINED_MODEL_NAME` | 模型名称（路由选择依据） |
| `MODEL_TASK_TYPE` | 任务类型：`video_generation` / `image_generation` / `audio_generation` / `third_video_generation` |

### 配置项（config 传入）

| 参数 | 说明 |
|------|------|
| `modelName` | 对外展示的模型名 |
| `modelTaskType` | 任务类型：`t2v` / `i2v` |
| `supportedResolutions` | 模型支持的分辨率配置 |

### 可选环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SGLANG_CMD_EXTRA_ARGS` | — | sglang serve 额外参数（如 `--num-gpus 4`） |
| `SGLANG_READY_TIMEOUT_S` | `600` | 服务就绪等待超时（秒） |
| `MAAS_PORT_FILE` | `/home/aiges/maas_port` | 端口写入文件路径 |
| `POLL_INTERVAL_MS` | `5000` | 轮询间隔（毫秒） |
| `POLL_TIMEOUT_S` | `1800` | 任务超时（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

### I2V 图片输入安全（可选）

| 变量 | 说明 |
|------|------|
| `IMAGE_URL_ALLOWLIST` | 允许下载的 host 白名单，逗号分隔 |
| `MAX_IMAGE_BYTES` | 最大图片大小（默认 10MB） |
| `IMAGE_PATH_ALLOW_PREFIX` | 允许读取的本地路径前缀 |

## 分辨率支持

| 等级 | 尺寸 |
|------|------|
| 480P | 832x480, 480x832, 624x624 |
| 720P | 1280x720, 720x1280, 960x960, 1088x832, 832x1088 |

## 错误码

| 错误码 | 含义 |
|--------|------|
| 1001 | 下游调用失败（videos.create / retrieve） |
| 1002 | 参数无效（缺少 prompt、尺寸不合法等） |
| 1003 | sglang 服务未就绪 |
| 1004 | 轮询超时 |
| 1005 | 图片读取失败（URL / 路径 / 二进制） |

## sglang serve 启动示例

```bash
export FULL_MODEL_PATH="Wan-AI/Wan2.2-T2V-A14B-Diffusers"
export SGLANG_CMD_EXTRA_ARGS="--num-gpus 4 --text-encoder-cpu-offload --pin-cpu-memory --ulysses-degree=2 --ring-degree=2"
```

## 流式响应协议

| 状态 | 含义 |
|------|------|
| `DataBegin` (0) | 首包，包含 task_id 和初始状态 |
| `DataContinue` (1) | 进度更新 |
| `DataEnd` (2) | 最终结果，包含视频信息与 usage |

## 项目结构说明

- **双层 Wrapper**：`utils/wrapper.py`（aiges 入口）→ `inference_wrapper/wrapper.py`（路由）→ 具体 Diffusion Wrapper
- **线程池模型**：每个 Worker 线程运行独立的 asyncio 事件循环，通过 `alloc_min_thread()` 负载均衡
- **健康探针**：自动向 `/var/run/wrapper_status` 写入健康状态（0=healthy, -1=unhealthy）