#!/usr/bin/env python
# coding:utf-8

import argparse
import json
import sys
import time
from typing import Any, Dict, Optional

import requests


def post_json(url: str, payload: Dict[str, Any], timeout_s: int = 30) -> Dict[str, Any]:
    r = requests.post(url, json=payload, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.json()


def get_json(url: str, timeout_s: int = 30) -> Dict[str, Any]:
    r = requests.get(url, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.json()


def get_bytes(url: str, timeout_s: int = 300) -> bytes:
    r = requests.get(url, timeout=(3, timeout_s))
    r.raise_for_status()
    return r.content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="例如 http://127.0.0.1:30010/v1")
    ap.add_argument("--model", default=None, help="可选，部分实现会忽略")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--poll-interval", type=float, default=1.5)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default="out.mp4")
    args = ap.parse_args()

    videos_create = args.base_url.rstrip("/") + "/videos"
    payload: Dict[str, Any] = {
        "prompt": args.prompt,
        "size": args.size,
        "extra_body": {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance,
            "seed": args.seed,
        },
    }
    if args.model:
        payload["model"] = args.model

    print("create payload:", json.dumps(payload, ensure_ascii=False))
    obj = post_json(videos_create, payload, timeout_s=30)
    video_id = obj.get("id") or obj.get("video_id")
    if not video_id:
        print("create response missing id:", obj)
        return 2
    print("video_id:", video_id, "status:", obj.get("status"), "progress:", obj.get("progress"))

    retrieve_url = args.base_url.rstrip("/") + f"/videos/{video_id}"
    content_url = args.base_url.rstrip("/") + f"/videos/{video_id}/content"
    deadline = time.time() + args.timeout

    last = None
    while True:
        if time.time() > deadline:
            print("timeout")
            return 3
        robj = get_json(retrieve_url, timeout_s=30)
        status = robj.get("status")
        progress = robj.get("progress")
        if (status, progress) != last:
            print("retrieve:", "status=", status, "progress=", progress)
            last = (status, progress)

        if status in ("succeeded", "completed"):
            break
        if status in ("failed", "cancelled"):
            print("task ended:", robj)
            return 4
        time.sleep(args.poll_interval)

    b = get_bytes(content_url, timeout_s=300)
    with open(args.out, "wb") as f:
        f.write(b)
    print("saved:", args.out, "bytes:", len(b))
    return 0


if __name__ == "__main__":
    sys.exit(main())

