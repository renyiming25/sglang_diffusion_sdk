#!/usr/bin/env python
# coding:utf-8
"""
sglang diffusion image wrapper 插件测试脚本

模拟加载器加载流程，使用 wrapperOnceExecAsync + callback 模式：
1. wrapperInit(config) — 初始化，启动 sglang serve
2. wrapperOnceExecAsync(params, reqData, usrTag) — 提交请求
3. callback 返回结果 — 图片二进制数据

请求通过 raw_req 字段传入 JSON payload，格式如:
{
    "prompt": "A calico cat playing a piano",
    "size": "1024x1024",
    "negative_prompt": "",
    "num_inference_steps": 10,
    "guidance_scale": 5.0,
    "seed": 42,
    "reference_url": "https://example.com/cat.jpg"  # i2i only
}

使用方式:
1. 文生图测试 (t2i):
   python test_image_wrapper_plugin.py --mode t2i --config config.json --prompt "A cat playing piano"

2. 图片编辑测试 (i2i, 使用图片 URL):
   python test_image_wrapper_plugin.py --mode i2i --config config.json --image-url https://example.com/cat.jpg --prompt "Make it snowy"

3. 图片编辑测试 (i2i, 使用本地图片):
   python test_image_wrapper_plugin.py --mode i2i --config config.json --image-path /path/to/image.png --prompt "Make it snowy"

配置文件格式 (JSON):
{
    "modelName": "wan2.2-flux-14b",
    "modelTaskType": "t2i",
    "supportedResolutions": {
        "wan2.2-flux-14b": ["512P", "1024P"]
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
from aiges.dto import Response, ResponseData, DataListNode, DataListCls, SessionCreateResponse
from aiges.core.types import DataImage, Once

# ==================== 配置区 ====================
DEFAULT_MODEL_PATH = os.environ.get("FULL_MODEL_PATH", "/workspace/LLamFile/ModelFile/wan_ai/Wan2.2-T2V-A14B-Diffusers")
DEFAULT_MODEL_NAME_T2I = "wan2.2-flux-14b"
DEFAULT_MODEL_NAME_I2I = "wan2.2-flux-i2i-14b"

DEFAULT_SIZE = "1024x1024"
DEFAULT_STEPS = 20
DEFAULT_SEED = 1234
DEFAULT_GUIDANCE_SCALE = 5.0

# callback 等待超时
DEFAULT_CALLBACK_TIMEOUT_S = 180
# ================================================


# ---------- callback 结果收集 ----------

class CallbackResultCollector:
    """收集 callback 返回的结果，支持线程安全等待"""
    def __init__(self):
        self.result = None
        self.error_code = None
        self.event = threading.Event()

    def on_callback(self, res, usrTag):
        """callback 回调函数，由 wrapper 线程池调用"""
        if isinstance(res, Response) and res.list:
            self.result = res
        else:
            self.error_code = res
        self.event.set()

    def wait(self, timeout_s=DEFAULT_CALLBACK_TIMEOUT_S):
        """阻塞等待 callback 结果"""
        self.event.wait(timeout=timeout_s)
        return self.event.is_set()

    def get_image_bytes(self):
        """从 Response 中提取图片二进制数据"""
        if not self.result or not self.result.list:
            return None
        for item in self.result.list:
            if item.key == "raw_resp" and item.data:
                return item.data
        return None

    def is_error(self):
        """检查是否返回了错误"""
        if self.error_code is not None:
            return True
        if self.result and hasattr(self.result, 'err'):
            return self.result.err != 0
        return False


# ---------- 配置构建 ----------

def load_config(config_path: str) -> dict:
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
    config = load_config(args.config)

    default_model_name = DEFAULT_MODEL_NAME_I2I if args.mode == "i2i" else DEFAULT_MODEL_NAME_T2I

    if args.model_path:
        os.environ["FULL_MODEL_PATH"] = args.model_path
    if args.model_name:
        config["modelName"] = args.model_name
    elif args.pretrained_name:
        config.setdefault("modelName", args.pretrained_name)
    if "modelName" not in config or not config["modelName"]:
        config["modelName"] = default_model_name

    config["modelTaskType"] = args.mode

    # supportedResolutions 默认值
    model_name = config.get("modelName", default_model_name)
    if "supportedResolutions" not in config or not config["supportedResolutions"]:
        config["supportedResolutions"] = {model_name: ["512P", "1024P"]}

    # S3 存储配置
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
    os.environ["MAAS_PORT_FILE"] = "/tmp/sglang_port"
    os.environ["LOG_LEVEL"] = args.log_level

    if args.extra_args:
        os.environ["SGLANG_CMD_EXTRA_ARGS"] = args.extra_args

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
    import importlib
    module_path = "ailab.inference_wrapper.diffusers.image_diffusion.wrapper"
    wrapper_module = importlib.import_module(module_path)
    Wrapper = getattr(wrapper_module, "Wrapper")
    return Wrapper, wrapper_module


def image_to_base64_data_url(image_path: str) -> str:
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
    if reference_url:
        raw_req["reference_url"] = reference_url
    return raw_req


def build_req_data(raw_req: dict) -> DataListCls:
    """构建 reqData，将 raw_req JSON 放入 raw_req 字段（一次传输）"""
    raw_req_bytes = json.dumps(raw_req, ensure_ascii=False).encode("utf-8")

    node = DataListNode()
    node.key = "raw_req"
    node.data = raw_req_bytes
    node.status = 2  # DataEnd

    data_list = DataListCls()
    data_list.list = [node]
    return data_list


def build_chunked_req_data(raw_req: dict, chunk_size: int = 256) -> list:
    """
    构建流式分片 reqData 列表，模拟 aiges 框架的 DataBegin/DataContinue/DataEnd 流式传输。

    将 raw_req JSON 字符串按 chunk_size 分片，返回 [(status, DataListCls), ...] 列表。
    """
    raw_req_str = json.dumps(raw_req, ensure_ascii=False)
    raw_req_bytes = raw_req_str.encode("utf-8")

    chunks = []
    for i in range(0, len(raw_req_bytes), chunk_size):
        chunk_data = raw_req_bytes[i:i + chunk_size]
        is_last = (i + chunk_size >= len(raw_req_bytes))

        if i == 0:
            status = 2 if is_last else 0  # DataEnd or DataBegin
        else:
            status = 2 if is_last else 1  # DataEnd or DataContinue

        node = DataListNode()
        node.key = "raw_req"
        node.data = chunk_data
        node.status = status

        data_list = DataListCls()
        data_list.list = [node]
        chunks.append((status, data_list))

    return chunks


def save_image(img_bytes: bytes, out_path: str) -> bool:
    """保存图片到文件"""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(img_bytes)
    file_size = len(img_bytes) / 1024
    # 检查图片格式
    if img_bytes[:4] == b'\x89PNG':
        fmt = "PNG"
    elif img_bytes[:2] == b'\xff\xd8':
        fmt = "JPEG"
    else:
        fmt = "unknown"
    print(f"  -> 保存成功: {out_path} ({fmt}, {file_size:.1f} KB)")
    return True


# ---------- 测试函数 ----------

def test_t2i(wrapper, args):
    """测试 T2I (Text-to-Image)"""
    print("\n" + "=" * 60)
    print(" T2I (Text-to-Image) Test")
    print("=" * 60)
    print(f"Prompt: {args.prompt[:100]}...")
    print(f"Size: {args.size}")

    raw_req = build_raw_req(args)
    print(f"raw_req: {json.dumps(raw_req, ensure_ascii=False)[:300]}")

    # 1. 构建 reqData
    reqData = build_req_data(raw_req)

    # 2. 注册 callback 收集器
    usrTag = "test_t2i_" + str(int(time.time()))
    collector = CallbackResultCollector()

    # Monkey-patch callback: 替换全局 callback 函数，拦截结果
    import importlib
    wrapper_module = importlib.import_module("ailab.inference_wrapper.diffusers.image_diffusion.wrapper")
    original_callback = wrapper_module.callback

    def patched_callback(res, tag):
        if tag == usrTag:
            collector.on_callback(res, tag)
        else:
            if original_callback:
                original_callback(res, tag)

    wrapper_module.callback = patched_callback
    # 同步替换 wrapper 引用的 callback
    import ailab.inference_wrapper.diffusers.image_diffusion.wrapper as img_wrapper_mod
    img_wrapper_mod.callback = patched_callback

    # 3. 调用 wrapperOnceExecAsync
    print(f"\n[1] Calling wrapperOnceExecAsync (usrTag={usrTag})...")
    start_time = time.time()
    params = {"sid": "test_t2i_session"}
    ret = wrapper.wrapperOnceExecAsync(params, reqData, usrTag=usrTag)
    print(f"    wrapperOnceExecAsync returned: {ret}")

    if ret != 0:
        print(f"[ERROR] wrapperOnceExecAsync failed with code: {ret}")
        # 恢复原始 callback
        wrapper_module.callback = original_callback
        img_wrapper_mod.callback = original_callback
        return False

    # 4. 等待 callback 结果
    print(f"\n[2] Waiting for callback result (timeout={args.timeout}s)...")
    arrived = collector.wait(timeout_s=args.timeout)

    elapsed = time.time() - start_time
    print(f"    Wait completed in {elapsed:.1f}s")

    # 恢复原始 callback
    wrapper_module.callback = original_callback
    img_wrapper_mod.callback = original_callback

    if not arrived:
        print(f"[ERROR] Callback timeout after {args.timeout}s")
        return False

    # 5. 检查结果
    if collector.is_error():
        print(f"[ERROR] Callback returned error: {collector.error_code}")
        return False

    img_bytes = collector.get_image_bytes()
    if not img_bytes:
        print(f"[ERROR] No image data in callback result")
        return False

    print(f"\n[3] Received image data: {len(img_bytes)} bytes")

    # 6. 保存图片
    out_path = args.out if args.out else os.path.join(args.output_dir, "test_t2i.png")
    save_image(img_bytes, out_path)

    print("\n" + "=" * 60)
    print(" T2I TEST PASSED")
    print("=" * 60)
    return True


def test_i2i(wrapper, args):
    """测试 I2I (Image-Edit)"""
    print("\n" + "=" * 60)
    print(" I2I (Image-Edit) Test")
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
        print("[ERROR] --image-url or --image-path is required for I2I mode")
        return False

    raw_req = build_raw_req(args, reference_url=reference_url)
    log_req = {k: (v[:80] + "..." if k == "reference_url" and isinstance(v, str) and len(v) > 80 else v)
               for k, v in raw_req.items()}
    print(f"raw_req: {json.dumps(log_req, ensure_ascii=False)[:300]}")

    # 1. 构建 reqData
    reqData = build_req_data(raw_req)

    # 2. 注册 callback 收集器
    usrTag = "test_i2i_" + str(int(time.time()))
    collector = CallbackResultCollector()

    import importlib
    wrapper_module = importlib.import_module("ailab.inference_wrapper.diffusers.image_diffusion.wrapper")
    original_callback = wrapper_module.callback

    def patched_callback(res, tag):
        if tag == usrTag:
            collector.on_callback(res, tag)
        else:
            if original_callback:
                original_callback(res, tag)

    wrapper_module.callback = patched_callback
    import ailab.inference_wrapper.diffusers.image_diffusion.wrapper as img_wrapper_mod
    img_wrapper_mod.callback = patched_callback

    # 3. 调用 wrapperOnceExecAsync
    print(f"\n[1] Calling wrapperOnceExecAsync (usrTag={usrTag})...")
    start_time = time.time()
    params = {"sid": "test_i2i_session"}
    ret = wrapper.wrapperOnceExecAsync(params, reqData, usrTag=usrTag)
    print(f"    wrapperOnceExecAsync returned: {ret}")

    if ret != 0:
        print(f"[ERROR] wrapperOnceExecAsync failed with code: {ret}")
        wrapper_module.callback = original_callback
        img_wrapper_mod.callback = original_callback
        return False

    # 4. 等待 callback 结果
    print(f"\n[2] Waiting for callback result (timeout={args.timeout}s)...")
    arrived = collector.wait(timeout_s=args.timeout)

    elapsed = time.time() - start_time
    print(f"    Wait completed in {elapsed:.1f}s")

    wrapper_module.callback = original_callback
    img_wrapper_mod.callback = original_callback

    if not arrived:
        print(f"[ERROR] Callback timeout after {args.timeout}s")
        return False

    # 5. 检查结果
    if collector.is_error():
        print(f"[ERROR] Callback returned error: {collector.error_code}")
        return False

    img_bytes = collector.get_image_bytes()
    if not img_bytes:
        print(f"[ERROR] No image data in callback result")
        return False

    print(f"\n[3] Received image data: {len(img_bytes)} bytes")

    # 6. 保存图片
    out_path = args.out if args.out else os.path.join(args.output_dir, "test_i2i.png")
    save_image(img_bytes, out_path)

    print("\n" + "=" * 60)
    print(" I2I TEST PASSED")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="sglang diffusion image wrapper plugin test (wrapperOnceExecAsync + callback mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # T2I 文生图测试
  python test_image_wrapper_plugin.py --mode t2i --config config.json --prompt "A cat playing piano"

  # I2I 图片编辑测试 (图片 URL)
  python test_image_wrapper_plugin.py --mode i2i --config config.json --image-url https://example.com/cat.jpg --prompt "Make it snowy"

  # I2I 图片编辑测试 (本地图片)
  python test_image_wrapper_plugin.py --mode i2i --config config.json --image-path /path/to/cat.png --prompt "Make it snowy"
        """
    )

    parser.add_argument("--config", default=None,
                        help="JSON config file path, passed to wrapperInit")
    parser.add_argument("--mode", choices=["t2i", "i2i"], default="t2i",
                        help="Test mode: t2i (text-to-image) or i2i (image-edit)")

    # 模型配置
    parser.add_argument("--model-path", default=None,
                        help=f"Model path (overrides FULL_MODEL_PATH env)")
    parser.add_argument("--model-name", default=None,
                        help="Model name for wrapperInit config.modelName")
    parser.add_argument("--pretrained-name", default=None,
                        help="Model name (deprecated, use --model-name)")
    parser.add_argument("--extra-args", default="",
                        help="Extra sglang serve args (e.g., '--num-gpus 1')")

    # S3 存储配置
    parser.add_argument("--s3-endpoint", default=None)
    parser.add_argument("--s3-bucket", default=None)
    parser.add_argument("--s3-secret-key", default=None)
    parser.add_argument("--s3-access-key", default=None)

    # 输入配置
    parser.add_argument("--prompt", default=None, help="Text prompt")
    parser.add_argument("--negative-prompt", default=None, help="Negative prompt")
    parser.add_argument("--image-url", default=None, help="Image URL for I2I mode")
    parser.add_argument("--image-path", default=None, help="Local image path for I2I mode")

    # 图片参数
    parser.add_argument("--size", default=DEFAULT_SIZE, help=f"Image size (default: {DEFAULT_SIZE})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Inference steps (default: {DEFAULT_STEPS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help=f"Guidance scale (default: {DEFAULT_GUIDANCE_SCALE})")

    # 输出与超时
    parser.add_argument("--out", default=None, help="Output image file path")
    parser.add_argument("--output-dir", default="./test_output", help="Output directory")
    parser.add_argument("--timeout", type=int, default=DEFAULT_CALLBACK_TIMEOUT_S,
                        help=f"Callback wait timeout in seconds (default: {DEFAULT_CALLBACK_TIMEOUT_S})")

    # 日志
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # 设置默认 prompt
    if not args.prompt:
        if args.mode == "t2i":
            args.prompt = "A calico cat playing a piano on stage, cinematic lighting, professional photography"
        else:
            args.prompt = "Make it snowy, winter landscape, soft lighting"

    # 构建 config
    config = build_config(args)

    model_name = config.get("modelName", "")
    if not model_name:
        print("[ERROR] modelName is not set. Use --config or --model-name")
        return 1

    if args.mode == "i2i" and "i2i" not in model_name:
        print(f"[WARNING] I2I mode but model name doesn't contain 'i2i': {model_name}")
    if args.mode == "t2i" and "i2i" in model_name:
        print(f"[WARNING] T2I mode but model name contains 'i2i': {model_name}")

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
        if args.mode == "t2i":
            success = test_t2i(wrapper, args)
        elif args.mode == "i2i":
            success = test_i2i(wrapper, args)
        else:
            print(f"[ERROR] Unknown mode: {args.mode}")
            success = False

        return 0 if success else 1

    finally:
        print("\n[CLEANUP] Finalizing wrapper...")
        wrapper.wrapperFini()


if __name__ == "__main__":
    sys.exit(main())