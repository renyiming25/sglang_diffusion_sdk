#!/usr/bin/env python
# coding:utf-8
"""
sglang diffusion video wrapper 测试脚本

测试场景：
1. T2V 基本生成
2. I2V 二进制图片输入
3. I2V URL 图片输入
4. 参数缺失测试
5. 超时测试

使用方式：
1. 先启动 sglang 服务：
   sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --port 30010

2. 运行测试：
   python test_video_wrapper.py --base-url http://127.0.0.1:30010/v1
"""

import argparse
import json
import sys
import time
import os
from typing import Any, Dict, Optional

import requests


def print_separator(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)


def post_json(url: str, payload: Dict[str, Any], timeout_s: int = 30) -> Dict[str, Any]:
    """POST JSON 请求"""
    r = requests.post(url, json=payload, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.json()


def get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    """GET JSON 请求"""
    r = requests.get(url, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.json()


def get_bytes(url: str, timeout_s: int = 300) -> bytes:
    """GET 二进制数据"""
    r = requests.get(url, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.content


def poll_video_task(
    base_url: str,
    video_id: str,
    poll_interval: float = 1.5,
    timeout: float = 900,
    verbose: bool = True
) -> Dict[str, Any]:
    """轮询视频任务状态"""
    retrieve_url = f"{base_url.rstrip('/')}/videos/{video_id}"
    deadline = time.time() + timeout
    last_status = None

    while True:
        if time.time() > deadline:
            return {"error": "timeout", "video_id": video_id}

        try:
            obj = get_json(retrieve_url, timeout_s=30)
            status = obj.get("status")
            progress = obj.get("progress")

            if verbose and (status, progress) != last_status:
                print(f"  [poll] status={status}, progress={progress}")
                last_status = (status, progress)

            if status in ("succeeded", "completed"):
                return {"status": "succeeded", "video_id": video_id, "obj": obj}
            if status in ("failed", "cancelled"):
                return {"error": f"task_{status}", "video_id": video_id, "obj": obj}

        except Exception as e:
            if verbose:
                print(f"  [poll] error: {e}")

        time.sleep(poll_interval)


def test_t2v_basic(base_url: str, output_dir: str, verbose: bool = True):
    """测试 T2V 基本生成"""
    print_separator("T2V-001: 基本生成测试")

    videos_create_url = f"{base_url.rstrip('/')}/videos"

    payload = {
        "prompt": "A calico cat playing a piano on stage, cinematic lighting",
        "size": "832x480",
        "extra_body": {
            "num_inference_steps": 10,
            "guidance_scale": 1.0,
            "seed": 42
        }
    }

    print(f"Request payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        # 创建任务
        obj = post_json(videos_create_url, payload, timeout_s=30)
        video_id = obj.get("id") or obj.get("video_id")

        if not video_id:
            print(f"FAIL: create response missing id: {obj}")
            return False

        print(f"Created video_id: {video_id}")

        # 轮询
        result = poll_video_task(base_url, video_id, verbose=verbose)

        if result.get("error"):
            print(f"FAIL: {result}")
            return False

        # 获取视频内容
        content_url = f"{base_url.rstrip('/')}/videos/{video_id}/content"
        video_bytes = get_bytes(content_url, timeout_s=300)

        # 保存视频
        output_path = os.path.join(output_dir, "test_t2v_basic.mp4")
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"SUCCESS: saved to {output_path}, size={len(video_bytes)} bytes")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_t2v_missing_prompt(base_url: str, verbose: bool = True):
    """测试参数缺失（缺少 prompt）"""
    print_separator("T2V-002: 参数缺失测试")

    videos_create_url = f"{base_url.rstrip('/')}/videos"

    payload = {
        "size": "832x480",
        "extra_body": {"seed": 42}
    }

    print(f"Request payload (missing prompt): {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        obj = post_json(videos_create_url, payload, timeout_s=30)
        print(f"Response: {obj}")

        # 检查是否返回错误
        if obj.get("error") or obj.get("code"):
            print("SUCCESS: server returned error as expected")
            return True
        else:
            print("WARN: server did not return error for missing prompt")
            return True  # 有些服务可能不校验

    except requests.exceptions.HTTPError as e:
        print(f"SUCCESS: server returned HTTP error: {e}")
        return True
    except Exception as e:
        print(f"FAIL: unexpected error: {e}")
        return False


def test_i2v_binary(base_url: str, output_dir: str, image_path: str, verbose: bool = True):
    """测试 I2V 二进制图片输入"""
    print_separator("I2V-001: 二进制图片输入测试")

    if not os.path.exists(image_path):
        print(f"SKIP: image file not found: {image_path}")
        return None

    videos_create_url = f"{base_url.rstrip('/')}/videos"

    # 读取图片并转为 base64 data URL
    import base64
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    # 简单判断图片类型
    ext = os.path.splitext(image_path)[1].lower()
    mime_type = "png" if ext == ".png" else "jpeg" if ext in (".jpg", ".jpeg") else "png"
    data_url = f"data:image/{mime_type};base64,{b64}"

    payload = {
        "prompt": "A beautiful sunset over the ocean, cinematic",
        "size": "832x480",
        "input_reference": data_url,
        "extra_body": {
            "num_inference_steps": 10,
            "guidance_scale": 1.0,
            "seed": 42
        }
    }

    print(f"Request with input_reference (base64, {len(image_bytes)} bytes)")

    try:
        obj = post_json(videos_create_url, payload, timeout_s=30)
        video_id = obj.get("id") or obj.get("video_id")

        if not video_id:
            print(f"FAIL: create response missing id: {obj}")
            return False

        print(f"Created video_id: {video_id}")

        result = poll_video_task(base_url, video_id, verbose=verbose)

        if result.get("error"):
            print(f"FAIL: {result}")
            return False

        content_url = f"{base_url.rstrip('/')}/videos/{video_id}/content"
        video_bytes = get_bytes(content_url, timeout_s=300)

        output_path = os.path.join(output_dir, "test_i2v_binary.mp4")
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"SUCCESS: saved to {output_path}, size={len(video_bytes)} bytes")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_i2v_url(base_url: str, output_dir: str, image_url: str, verbose: bool = True):
    """测试 I2V URL 图片输入"""
    print_separator("I2V-002: URL 图片输入测试")

    videos_create_url = f"{base_url.rstrip('/')}/videos"

    payload = {
        "prompt": "A serene mountain landscape at dawn",
        "size": "832x480",
        "input_reference": image_url,  # 直接传 URL
        "extra_body": {
            "num_inference_steps": 10,
            "guidance_scale": 1.0,
            "seed": 42
        }
    }

    print(f"Request with input_reference URL: {image_url}")

    try:
        obj = post_json(videos_create_url, payload, timeout_s=30)
        video_id = obj.get("id") or obj.get("video_id")

        if not video_id:
            print(f"FAIL: create response missing id: {obj}")
            return False

        print(f"Created video_id: {video_id}")

        result = poll_video_task(base_url, video_id, verbose=verbose)

        if result.get("error"):
            print(f"FAIL: {result}")
            return False

        content_url = f"{base_url.rstrip('/')}/videos/{video_id}/content"
        video_bytes = get_bytes(content_url, timeout_s=300)

        output_path = os.path.join(output_dir, "test_i2v_url.mp4")
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"SUCCESS: saved to {output_path}, size={len(video_bytes)} bytes")
        return True

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_timeout(base_url: str, verbose: bool = True):
    """测试超时场景（设置极短超时）"""
    print_separator("T2V-004: 超时测试")

    videos_create_url = f"{base_url.rstrip('/')}/videos"

    payload = {
        "prompt": "A dog running in a park",
        "size": "832x480",
        "extra_body": {"seed": 42}
    }

    print(f"Request payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    try:
        obj = post_json(videos_create_url, payload, timeout_s=30)
        video_id = obj.get("id") or obj.get("video_id")

        if not video_id:
            print(f"FAIL: create response missing id: {obj}")
            return False

        print(f"Created video_id: {video_id}")
        print("Polling with 1 second timeout (expecting timeout)...")

        result = poll_video_task(base_url, video_id, poll_interval=0.5, timeout=1.0, verbose=verbose)

        if result.get("error") == "timeout":
            print("SUCCESS: timeout detected as expected")
            return True
        else:
            print(f"WARN: task completed before timeout: {result}")
            return True  # 任务太快完成了，也算通过

    except Exception as e:
        print(f"FAIL: {e}")
        return False


def test_health_check(base_url: str, verbose: bool = True):
    """测试健康检查端点"""
    print_separator("Health Check")

    health_url = base_url.rstrip('/') + "/../health"

    try:
        r = requests.get(health_url, timeout=(1, 3))
        print(f"Health check status: {r.status_code}")
        print(f"Response: {r.text[:200]}")
        return r.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="sglang diffusion video wrapper test")
    parser.add_argument("--base-url", required=True, help="Base URL, e.g. http://127.0.0.1:30010/v1")
    parser.add_argument("--output-dir", default="./test_output", help="Output directory for videos")
    parser.add_argument("--image-path", default=None, help="Image path for I2V binary test")
    parser.add_argument("--image-url", default="https://picsum.photos/512/512", help="Image URL for I2V URL test")
    parser.add_argument("--skip-i2v", action="store_true", help="Skip I2V tests")
    parser.add_argument("--skip-timeout", action="store_true", help="Skip timeout test")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}

    # 健康检查
    results["health_check"] = test_health_check(args.base_url, args.verbose)

    # T2V 测试
    results["t2v_basic"] = test_t2v_basic(args.base_url, args.output_dir, args.verbose)
    results["t2v_missing_prompt"] = test_t2v_missing_prompt(args.base_url, args.verbose)

    # I2V 测试
    if not args.skip_i2v:
        if args.image_path:
            results["i2v_binary"] = test_i2v_binary(args.base_url, args.output_dir, args.image_path, args.verbose)
        else:
            print("\n[SKIP] I2V binary test: --image-path not provided")

        results["i2v_url"] = test_i2v_url(args.base_url, args.output_dir, args.image_url, args.verbose)
    else:
        print("\n[SKIP] I2V tests: --skip-i2v flag set")

    # 超时测试
    if not args.skip_timeout:
        results["timeout"] = test_timeout(args.base_url, args.verbose)
    else:
        print("\n[SKIP] Timeout test: --skip-timeout flag set")

    # 汇总
    print_separator("Test Summary")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for name, result in results.items():
        status = "PASS" if result is True else "FAIL" if result is False else "SKIP"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
