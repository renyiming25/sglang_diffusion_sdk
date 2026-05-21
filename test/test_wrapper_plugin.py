#!/usr/bin/env python
# coding:utf-8
"""
sglang diffusion video wrapper 插件测试脚本

模拟加载器加载流程，使用 wrapperOnceExecAsync + callback 模式：
1. wrapperInit(config) — 初始化，启动 sglang serve
2. wrapperOnceExecAsync(params, reqData, usrTag) — 提交请求
3. callback 返回结果 — 视频生成结果 (含 video URL)

请求通过 raw_req 字段传入 JSON payload，格式如:
{
    "prompt": "A curious raccoon exploring a forest, cinematic lighting",
    "negative_prompt": "",
    "size": "1280x720",
    "seconds": 4,
    "fps": 24,
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "reference_url": "https://example.com/image.jpg"  # I2V only
}

使用方式:
1. 运行 T2V 测试 (使用 --config 传入配置文件):
   python test_wrapper_plugin.py --mode t2v --config config_t2v.json --prompt "A cat playing piano"

2. 运行 I2V 测试 (使用图片 URL):
   python test_wrapper_plugin.py --mode i2v --config config_i2v.json --image-url https://example.com/cat.jpg --prompt "The cat starts moving"

3. 运行 I2V 测试 (使用本地图片, 自动转 base64):
   python test_wrapper_plugin.py --mode i2v --config config_i2v.json --image-path /path/to/image.png --prompt "The cat starts moving"

配置文件格式 (JSON):
{
    "modelName": "wan2.2-t2v-14b",
    "modelTaskType": "t2v",
    "sglStorageType": "s3",
    "sglS3EndpointURL": "https://s3.example.com",
    "sglS3BucketName": "my-bucket",
    "sglS3SecretKey": "secret",
    "sglS3AccessKey": "access",
    "pollIntervalMs": "5000",
    "supportedResolutions": {
        "wan2.2-t2v-14b": ["480P", "720P"]
    }
}
"""

import argparse
import base64
import json
import os
import sys
import time
import threading

# aiges 框架导入
from aiges.dto import Response, DataListNode, DataListCls
from aiges.core.types import DataEnd

# ==================== 配置区 ====================
# 默认模型配置
DEFAULT_MODEL_PATH = os.environ.get("FULL_MODEL_PATH", "/workspace/LLamFile/ModelFile/wan_ai/Wan2.2-T2V-A14B-Diffusers")
DEFAULT_MODEL_NAME_T2V = "wan2.2-t2v-14b"
DEFAULT_MODEL_NAME_I2V = "wan2.2-i2v-14b"

# 默认推理参数
DEFAULT_SIZE = "832x480"
DEFAULT_STEPS = 20
DEFAULT_SEED = 1234
DEFAULT_GUIDANCE_SCALE = 5.0

# callback 等待超时
DEFAULT_CALLBACK_TIMEOUT_S = 1800
# ================================================


# ---------- callback 结果收集 ----------

class CallbackResultCollector:
    """收集 callback 返回的结果，支持线程安全等待"""
    def __init__(self):
        self.result = None
        self.error_code = None
        self.event = threading.Event()

    def on_callback(self, res, usr_tag):
        """callback 回调函数，由 wrapper 线程池调用"""
        if isinstance(res, Response) and res.list:
            self.result = res
            self.error_code = 0
        elif isinstance(res, Response) and hasattr(res, 'err_code'):
            self.error_code = res.err_code
        else:
            self.error_code = -1
        self.event.set()

    def wait(self, timeout_s: float) -> bool:
        """等待 callback 返回，返回是否在超时前收到结果"""
        return self.event.wait(timeout=timeout_s)


def load_config(config_path: str) -> dict:
    """加载 JSON 配置文件"""
    if not config_path:
        return {}
    if not os.path.exists(config_path):
        print(f"[WARN] Config file not found: {config_path}, using empty config")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[CONFIG] Loaded from {config_path}: {json.dumps(config, ensure_ascii=False, indent=2)[:500]}")
    return config


def build_config(args) -> dict:
    """
    构建 wrapperInit 所需的 config 字典。
    wrapper 要求: modelName, modelTaskType, supportedResolutions 必须存在。
    优先级: 命令行参数 > 配置文件 > 默认值
    """
    # 从配置文件加载基础配置
    config = load_config(args.config)

    # 确定默认 model_name (根据模式)
    default_model_name = DEFAULT_MODEL_NAME_I2V if args.mode == "i2v" else DEFAULT_MODEL_NAME_T2V

    # 命令行参数覆盖
    if args.model_path:
        os.environ["FULL_MODEL_PATH"] = args.model_path
    if args.model_name:
        config["modelName"] = args.model_name
    elif args.pretrained_name:
        config.setdefault("modelName", args.pretrained_name)
    # 确保 modelName 始终存在 (wrapperInit 必需)
    if "modelName" not in config or not config["modelName"]:
        config["modelName"] = default_model_name

    # 确保 modelTaskType 始终存在 (wrapperInit 必需, 值为 t2v/i2v)
    config["modelTaskType"] = args.mode

    if args.poll_interval:
        config["pollIntervalMs"] = str(args.poll_interval)

    # 支持的分辨率: 如果配置文件未指定，根据 model_name 生成默认值
    model_name = config.get("modelName", default_model_name)
    if "supportedResolutions" not in config or not config["supportedResolutions"]:
        if "i2v" in model_name:
            config["supportedResolutions"] = {model_name: ["480P", "720P"]}
        elif "1.3b" in model_name:
            config["supportedResolutions"] = {model_name: ["480P"]}
        else:
            config["supportedResolutions"] = {model_name: ["480P", "720P"]}

    # S3 存储配置: 命令行可覆盖
    if args.s3_endpoint:
        config["sglS3EndpointURL"] = args.s3_endpoint
    if args.s3_bucket:
        config["sglS3BucketName"] = args.s3_bucket
    if args.s3_secret_key:
        config["sglS3SecretKey"] = args.s3_secret_key
    if args.s3_access_key:
        config["sglS3AccessKey"] = args.s3_access_key

    return config


def setup_environment(args, config: dict):
    """设置环境变量（config 中不包含的部分仍走环境变量）"""
    os.environ["POLL_TIMEOUT_S"] = str(args.timeout)
    os.environ["MAAS_PORT_FILE"] = "/tmp/sglang_port"
    os.environ["LOG_LEVEL"] = args.log_level

    if args.extra_args:
        os.environ["SGLANG_CMD_EXTRA_ARGS"] = args.extra_args

    # FULL_MODEL_PATH: 命令行 --model-path 已在 build_config() 中设置
    # 此处仅做兜底，确保环境变量始终存在
    if "FULL_MODEL_PATH" not in os.environ:
        os.environ["FULL_MODEL_PATH"] = DEFAULT_MODEL_PATH

    # 设置 Python 路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print(f"[ENV] FULL_MODEL_PATH: {os.environ.get('FULL_MODEL_PATH', 'NOT SET')}")
    print(f"[ENV] modelName: {config.get('modelName', 'NOT SET')}")
    print(f"[ENV] modelTaskType: {config.get('modelTaskType', 'NOT SET')}")


def import_wrapper():
    """动态导入 wrapper 模块"""
    import importlib
    module_path = "ailab.inference_wrapper.diffusers.video_diffusion.wrapper"
    wrapper_module = importlib.import_module(module_path)
    Wrapper = getattr(wrapper_module, "Wrapper")
    return Wrapper, wrapper_module


def image_to_base64_data_url(image_path: str) -> str:
    """读取本地图片文件并转为 data:image/...;base64,... 格式"""
    import imghdr
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    fmt = imghdr.what(None, h=image_bytes)
    if fmt == "jpg":
        fmt = "jpeg"
    if not fmt:
        fmt = "png"
    b64_str = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{fmt};base64,{b64_str}"


def build_raw_req(args, reference_url=None) -> dict:
    """构建 raw_req JSON payload"""
    raw_req = {
        "prompt": args.prompt,
        "size": args.size,
    }
    if args.negative_prompt:
        raw_req["negative_prompt"] = args.negative_prompt
    if args.seed is not None:
        raw_req["seed"] = args.seed
    if args.steps:
        raw_req["num_inference_steps"] = args.steps
    if args.guidance_scale is not None:
        raw_req["guidance_scale"] = args.guidance_scale
    if args.seconds:
        raw_req["seconds"] = args.seconds
    if args.fps:
        raw_req["fps"] = args.fps
    if reference_url:
        raw_req["reference_url"] = reference_url
    return raw_req


def build_req_data(raw_req: dict) -> DataListCls:
    """构建 wrapperOnceExecAsync 所需的 DataListCls (包含 raw_req 字段)"""
    raw_req_bytes = json.dumps(raw_req, ensure_ascii=False).encode("utf-8")

    node = DataListNode()
    node.key = "raw_req"
    node.data = raw_req_bytes
    node.status = DataEnd

    data_list = DataListCls()
    data_list.list = [node]
    return data_list


def monkey_patch_callback(collector: CallbackResultCollector):
    """
    Monkey-patch aiges callback 函数，将结果转发到 collector。
    wrapper 内部 import 的 callback 需要被替换。
    """
    import ailab.inference_wrapper.diffusers.video_diffusion.wrapper as wrapper_module
    wrapper_module.callback = collector.on_callback


def parse_callback_result(collector: CallbackResultCollector) -> dict:
    """解析 callback 返回的 Response，提取 raw_resp 和 usage"""
    result = {
        "success": False,
        "video_info": None,
        "usage": None,
        "error_code": collector.error_code,
    }

    if not collector.result or not collector.result.list:
        return result

    for item in collector.result.list:
        if not hasattr(item, 'data') or not item.data:
            continue
        try:
            data = item.data if isinstance(item.data, bytes) else item.data.encode("utf-8")
            content = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
            continue

        key = getattr(item, 'key', '')
        if key == "raw_resp":
            result["video_info"] = content
            result["success"] = True
        elif key == "usage":
            result["usage"] = content

    return result


def test_t2v(wrapper, args):
    """测试 T2V (Text-to-Video) — wrapperOnceExecAsync + callback"""
    print("\n" + "=" * 60)
    print(" T2V (Text-to-Video) Test")
    print("=" * 60)
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"Size: {args.size}")

    # 构建 raw_req payload
    raw_req = build_raw_req(args)
    print(f"raw_req: {json.dumps(raw_req, ensure_ascii=False)[:300]}")

    # 构建 reqData
    req_data = build_req_data(raw_req)

    # 设置 callback 收集器
    collector = CallbackResultCollector()
    monkey_patch_callback(collector)

    # 提交异步请求
    print("\n[1] Calling wrapperOnceExecAsync...")
    start_time = time.time()
    usr_tag = "test_t2v_tag"
    ret = wrapper.wrapperOnceExecAsync({}, req_data, usr_tag)
    print(f"    wrapperOnceExecAsync returned: {ret}")

    if ret != 0:
        print(f"[ERROR] wrapperOnceExecAsync failed with code: {ret}")
        return False

    # 等待 callback 返回
    print(f"\n[2] Waiting for callback (timeout={args.timeout}s)...")
    got_result = collector.wait(timeout_s=args.timeout)
    elapsed = time.time() - start_time

    if not got_result:
        print(f"[ERROR] Callback timeout after {args.timeout}s")
        return False

    print(f"    Callback received in {elapsed:.1f}s")

    # 解析结果
    print("\n[3] Result summary:")
    result = parse_callback_result(collector)

    if not result["success"]:
        print(f"    [FAILED] error_code={result['error_code']}")
        return False

    video_info = result["video_info"]
    usage = result["usage"]

    if video_info:
        print(f"    video_info: {json.dumps(video_info, ensure_ascii=False)[:500]}")
        video_url = video_info.get("url")
        if video_url:
            print(f"\n[4] Video URL: {video_url}")
        else:
            print("\n[4] No video URL in result (check video_info above)")

    if usage:
        print(f"    usage: {json.dumps(usage, ensure_ascii=False)}")

    return True


def test_i2v(wrapper, args):
    """测试 I2V (Image-to-Video) — wrapperOnceExecAsync + callback"""
    print("\n" + "=" * 60)
    print(" I2V (Image-to-Video) Test")
    print("=" * 60)
    print(f"Prompt: {args.prompt[:100]}...")

    # 确定 reference_url
    reference_url = None
    if args.image_url:
        reference_url = args.image_url
        print(f"Reference URL: {reference_url}")
    elif args.image_path:
        if not os.path.exists(args.image_path):
            print(f"[ERROR] Image file not found: {args.image_path}")
            return False
        reference_url = image_to_base64_data_url(args.image_path)
        print(f"Reference URL: (base64 from {args.image_path}, {len(reference_url)} chars)")
    else:
        print("[ERROR] --image-url or --image-path is required for I2V mode")
        return False

    # 构建 raw_req payload
    raw_req = build_raw_req(args, reference_url=reference_url)
    # 不打印完整 raw_req 避免刷屏 base64
    log_req = {k: (v[:80] + "..." if k == "reference_url" and isinstance(v, str) and len(v) > 80 else v)
               for k, v in raw_req.items()}
    print(f"raw_req: {json.dumps(log_req, ensure_ascii=False)[:300]}")

    # 构建 reqData
    req_data = build_req_data(raw_req)

    # 设置 callback 收集器
    collector = CallbackResultCollector()
    monkey_patch_callback(collector)

    # 提交异步请求
    print("\n[1] Calling wrapperOnceExecAsync...")
    start_time = time.time()
    usr_tag = "test_i2v_tag"
    ret = wrapper.wrapperOnceExecAsync({}, req_data, usr_tag)
    print(f"    wrapperOnceExecAsync returned: {ret}")

    if ret != 0:
        print(f"[ERROR] wrapperOnceExecAsync failed with code: {ret}")
        return False

    # 等待 callback 返回
    print(f"\n[2] Waiting for callback (timeout={args.timeout}s)...")
    got_result = collector.wait(timeout_s=args.timeout)
    elapsed = time.time() - start_time

    if not got_result:
        print(f"[ERROR] Callback timeout after {args.timeout}s")
        return False

    print(f"    Callback received in {elapsed:.1f}s")

    # 解析结果
    print("\n[3] Result summary:")
    result = parse_callback_result(collector)

    if not result["success"]:
        print(f"    [FAILED] error_code={result['error_code']}")
        return False

    video_info = result["video_info"]
    usage = result["usage"]

    if video_info:
        print(f"    video_info: {json.dumps(video_info, ensure_ascii=False)[:500]}")
        video_url = video_info.get("url")
        if video_url:
            print(f"\n[4] Video URL: {video_url}")
        else:
            print("\n[4] No video URL in result (check video_info above)")

    if usage:
        print(f"    usage: {json.dumps(usage, ensure_ascii=False)}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="sglang diffusion video wrapper plugin test (wrapperOnceExecAsync + callback mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # T2V 测试 (使用配置文件)
  python test_wrapper_plugin.py --mode t2v --config config_t2v.json --prompt "A cat playing piano"

  # T2V 测试 (命令行覆盖)
  python test_wrapper_plugin.py --mode t2v --config config_t2v.json \\
      --model-path /path/to/model --model-name wan2.2-t2v-14b

  # I2V 测试 (使用图片 URL)
  python test_wrapper_plugin.py --mode i2v --config config_i2v.json \\
      --image-url https://example.com/cat.jpg --prompt "The cat starts moving"

  # I2V 测试 (使用本地图片)
  python test_wrapper_plugin.py --mode i2v --config config_i2v.json \\
      --image-path /path/to/image.jpg --prompt "The cat starts moving"
        """
    )

    # 配置文件 (核心)
    parser.add_argument("--config", default=None,
                        help="JSON config file path, passed to wrapperInit as config dict")

    # 模式选择
    parser.add_argument("--mode", choices=["t2v", "i2v"], default="t2v",
                        help="Test mode: t2v (text-to-video) or i2v (image-to-video)")

    # 模型配置 (可覆盖 config 中的值)
    parser.add_argument("--model-path", default=None,
                        help=f"Model path (overrides FULL_MODEL_PATH env, default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--pretrained-name", default=None,
                        help="Model name (deprecated, use --model-name instead)")
    parser.add_argument("--model-name", default=None,
                        help="Model name for wrapperInit config.modelName")
    parser.add_argument("--extra-args", default="",
                        help="Extra sglang serve args (e.g., '--num-gpus 1')")

    # S3 存储配置 (可覆盖 config 中的值)
    parser.add_argument("--s3-endpoint", default=None,
                        help="S3 endpoint URL")
    parser.add_argument("--s3-bucket", default=None,
                        help="S3 bucket name")
    parser.add_argument("--s3-secret-key", default=None,
                        help="S3 secret access key")
    parser.add_argument("--s3-access-key", default=None,
                        help="S3 access key ID")

    # 输入配置
    parser.add_argument("--prompt", default=None,
                        help="Text prompt for video generation")
    parser.add_argument("--negative-prompt", default=None,
                        help="Negative prompt")
    parser.add_argument("--image-url", default=None,
                        help="Image URL for I2V mode (http/https link)")
    parser.add_argument("--image-path", default=None,
                        help="Local image path for I2V mode (auto-converted to base64 data URL)")

    # 视频参数
    parser.add_argument("--size", default=DEFAULT_SIZE,
                        help=f"Video size WxH (default: {DEFAULT_SIZE})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"Inference steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help=f"Guidance scale (default: {DEFAULT_GUIDANCE_SCALE})")
    parser.add_argument("--seconds", type=int, default=None,
                        help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=None,
                        help="Video frames per second")

    # 超时配置
    parser.add_argument("--poll-interval", type=int, default=None,
                        help="Poll interval in ms (passed to config as pollIntervalMs)")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CALLBACK_TIMEOUT_S,
                        help=f"Callback wait timeout in seconds (default: {DEFAULT_CALLBACK_TIMEOUT_S})")

    # 日志配置
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: INFO)")

    args = parser.parse_args()

    # 设置默认 prompt
    if not args.prompt:
        if args.mode == "t2v":
            args.prompt = (
                "A calico cat playing a piano on stage, cinematic lighting, "
                "professional photography, 4K quality"
            )
        else:
            args.prompt = (
                "Create a cinematic video based on this image, "
                "smooth camera movement, dramatic lighting"
            )

    # 构建 config 字典 (核心: 传入 wrapperInit)
    config = build_config(args)

    # 校验必要字段
    model_name = config.get("modelName", "")
    if not model_name:
        print(f"[ERROR] modelName is not set. Use --config or --model-name")
        return 1

    if args.mode == "i2v" and "i2v" not in model_name:
        print(f"[WARNING] I2V mode but model name doesn't contain 'i2v': {model_name}")
    if args.mode == "t2v" and "i2v" in model_name:
        print(f"[WARNING] T2V mode but model name contains 'i2v': {model_name}")

    # 设置环境变量
    setup_environment(args, config)

    # 导入 wrapper
    print("\n[INIT] Importing wrapper module...")
    Wrapper, _ = import_wrapper()
    wrapper = Wrapper()
    print("[INIT] Wrapper instance created")

    # 初始化 wrapper
    print(f"\n[INIT] Initializing wrapper with config: {json.dumps(config, ensure_ascii=False)[:300]}...")
    print("[INIT] (this may take a while to start sglang serve)")
    start_time = time.time()
    ret = wrapper.wrapperInit(config)
    elapsed = time.time() - start_time

    if ret != 0:
        print(f"[ERROR] wrapperInit failed with code: {ret}")
        return 1

    print(f"[INIT] wrapperInit succeeded in {elapsed:.1f}s")

    try:
        # 运行测试
        if args.mode == "t2v":
            success = test_t2v(wrapper, args)
        elif args.mode == "i2v":
            success = test_i2v(wrapper, args)
        else:
            print(f"[ERROR] Unknown mode: {args.mode}")
            success = False

        print("\n" + "=" * 60)
        if success:
            print(" TEST PASSED")
        else:
            print(" TEST FAILED")
        print("=" * 60)

        return 0 if success else 1

    finally:
        # 清理
        print("\n[CLEANUP] Finalizing wrapper...")
        wrapper.wrapperFini()


if __name__ == "__main__":
    sys.exit(main())
