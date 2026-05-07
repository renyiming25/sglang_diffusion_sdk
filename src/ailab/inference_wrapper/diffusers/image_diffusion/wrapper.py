#!/usr/bin/env python
# coding:utf-8
import os
import logging
import json
import threading
import asyncio
import queue
import uuid
import subprocess
import socket
import time
import base64
import requests
from openai import OpenAI

from typing import Dict

from aiges.core.types import *
try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls, SessionCreateResponse, callback  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls, SessionCreateResponse, callback

from aiges.sdk import WrapperBase
from aiges.utils.log import getFileLogger

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TASK_TYPE_CONFIGS = ["t2i", "i2i"]
RESOLUTIONS_TOSIZE = {
    "512P": ["512x512", "768x512", "512x768"],
    "1024P": ["1024x1024", "1536x1024", "1024x1536", "1280x720", "720x1280"],
}

# Error codes
ERROR_DOWNSTREAM_FAILED = 1001
ERROR_INVALID_PARAMS = 1002
ERROR_SERVER_NOT_READY = 1003
ERROR_POLL_TIMEOUT = 1004
ERROR_IMAGE_READ_FAILED = 1005


# ---------- module-level helpers (aligned with video_diffusion pattern) ----------

def _get_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    _, port = s.getsockname()
    s.close()
    return int(port)


def _write_port_file(path: str, port: int, filelogger) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(port).strip() + "\n")
        filelogger.info(f"write maas_port {port} to {path}")
    except Exception as e:
        filelogger.error(f"write maas_port file failed: {e}")
        raise


def _wait_server_ready(server_url: str, filelogger, timeout_s: int = 300) -> None:
    deadline = time.time() + timeout_s
    while True:
        if time.time() > deadline:
            raise TimeoutError(f"sglang server not ready within {timeout_s}s: {server_url}")
        try:
            r = requests.get(server_url + "/health", timeout=(1, 3))
            if r.status_code == 200:
                filelogger.info(f"{server_url} ready")
                return
        except Exception as e:
            filelogger.debug(f"{server_url} health exception: {e}")
        time.sleep(2)


def _launch_sglang_serve(model_path: str, port: int, filelogger) -> subprocess.Popen:
    extra_args = os.environ.get("SGLANG_CMD_EXTRA_ARGS", "").strip()
    extra_args = extra_args if extra_args else ""
    command = (
        f"sglang serve "
        f"--model-path {model_path} "
        f"--port {port} "
        f"{extra_args}"
    ).strip()
    filelogger.info(f"sglang command: {command}")
    return subprocess.Popen(command, shell=True)


def _setup_health_status_probe(port: int, filelogger) -> None:
    """
    参考 wrapper_text_generate 的 /var/run/wrapper_status 约定：
    - 0: healthy
    - -1: unhealthy
    """

    def run():
        os.makedirs("/var/run", exist_ok=True)
        while True:
            try:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=(1, 3))
                code = "0" if r.status_code == 200 else "-1"
            except Exception:
                code = "-1"
            try:
                with open("/var/run/wrapper_status", "w", encoding="utf-8") as h:
                    h.write(code)
            except Exception as e:
                filelogger.debug(f"write wrapper_status failed: {e}")
            time.sleep(3)

    threading.Thread(target=run, daemon=True).start()


def _normalize_url(url: str, filelogger) -> str:
    """
    规范化 url，支持两种输入格式：
    1. http(s) 图像链接 — 原样返回
    2. base64 数据（无 data: 前缀） — 补全 data:image/png;base64, 前缀
    """
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("data:"):
        return url
    filelogger.info("url is raw base64, prepended data:image/png;base64, prefix")
    return f"data:image/png;base64,{url}"


def _parse_raw_req(reqData: DataListCls, filelogger) -> dict:
    """从 reqData 中解析 raw_req JSON 字段"""
    raw_req_node = reqData.get("raw_req")
    if not raw_req_node:
        return {}
    raw_req_data = raw_req_node.data
    if isinstance(raw_req_data, bytes):
        raw_req_data = raw_req_data.decode("utf-8")
    try:
        parsed = json.loads(raw_req_data)
        if isinstance(parsed, str):
            filelogger.warning("raw_req is double-encoded JSON, parsing again")
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            filelogger.error(f"raw_req is not a dict, got {type(parsed).__name__}")
            return {}
        return parsed
    except Exception as e:
        filelogger.error(f"Failed to parse raw_req JSON: {e}")
        return {}


# ---------- data classes ----------

class RequestInfo:
    def __init__(self, sid: str, params: dict, user_tag: str = ""):
        self.handle = str(uuid.uuid4().hex)
        self.sid = sid
        self.user_tag = user_tag
        self.params = params
        self.raw_req = None  # parsed dict from reqData


class PromptInferenceInfo:
    def __init__(self, wrapper,
                 prompt: str,
                 requestInfo: RequestInfo,
                 url: str = None):
        self.wrapper = wrapper
        self.requestInfo = requestInfo
        self.prompt = prompt
        self.url = url
        self.request_id = str(uuid.uuid4().hex)


# ---------- thread pool ----------

class ThreadPool:
    def __init__(self, num_threads, wrapper):
        self.num_threads = num_threads
        self.threads = {}
        self.task_queues = {}
        self.lock = threading.Lock()
        self.wrapper = wrapper
        LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
        self.filelogger = getFileLogger(level=LEVEL)

        for i in range(num_threads):
            process_id = os.getpid()
            thread_id = "process-{}-thread-{}".format(str(process_id), str(i))
            task_queue = queue.Queue()
            self.threads[thread_id] = threading.Thread(target=self.worker, args=(thread_id, task_queue))
            self.task_queues[thread_id] = task_queue

        for thread in self.threads.values():
            thread.start()

    def worker(self, thread_id, task_queue):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.task_loop(thread_id, task_queue))

    async def task_loop(self, thread_id, task_queue):
        self.filelogger.debug(f"task_loop {thread_id} enter")
        while True:
            if not task_queue.empty():
                task: PromptInferenceInfo = task_queue.get_nowait()
                if task is None:
                    break
                asyncio.create_task(task.wrapper.create_image_task(task))
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.01)
        self.filelogger.info(f"task_loop {thread_id} end")

    def alloc_min_thread(self) -> str:
        with self.lock:
            min_thread_id = min(self.threads, key=lambda thread_id: self.task_queues[thread_id].qsize())
        return min_thread_id

    def put_task(self, thread_id, task):
        with self.lock:
            self.task_queues[thread_id].put(task)

    def wait_completion(self):
        for task_queue in self.task_queues.values():
            task_queue.put(None)
        for thread in self.threads.values():
            thread.join()


# ---------- wrapper ----------

class Wrapper(WrapperBase):
    version = "v1"
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
        self.filelogger = getFileLogger(level=LEVEL)
        self.base_model: str = None
        self.server_url: str = None
        self.model_name: str = None
        self.task_type: str = None
        self.client: OpenAI = None
        self.thread_pool: ThreadPool = None
        self.thread_pool_size: int = 1
        self.supported_resolutions: dict = {}

    def wrapperInit(self, config: Dict) -> int:
        self.filelogger.info("Initializing ...")
        try:
            if not config:
                self.filelogger.info("Config is empty, using environment variables")
            else:
                os.environ['SGLANG_CLOUD_STORAGE_TYPE'] = config.get("sglStorageType", "s3")
                os.environ['SGLANG_S3_ENDPOINT_URL'] = config.get("sglS3EndpointURL", "")
                os.environ['SGLANG_S3_BUCKET_NAME'] = config.get("sglS3BucketName", "")
                os.environ['SGLANG_S3_SECRET_ACCESS_KEY'] = config.get("sglS3SecretKey", "")
                os.environ['SGLANG_S3_ACCESS_KEY_ID'] = config.get("sglS3AccessKey", "")

                self.model_name = config.get("modelName", "")
                self.task_type = config.get("modelTaskType", "t2i")
                supported_resolutions = config.get("supportedResolutions", {})
                if isinstance(supported_resolutions, str):
                    try:
                        supported_resolutions = json.loads(supported_resolutions)
                    except Exception as e:
                        self.filelogger.error(f"Failed to parse supportedResolutions JSON: {e}")
                        supported_resolutions = {}
                self.supported_resolutions = supported_resolutions

            # 使用环境变量 FULL_MODEL_PATH 作为模型路径
            self.base_model = os.environ.get("FULL_MODEL_PATH")

            if not self.base_model:
                self.filelogger.error("FULL_MODEL_PATH is not set.")
                return -1

            if not self.model_name:
                self.filelogger.error("modelName is not set in config.")
                return -1

            if not self.task_type or self.task_type not in TASK_TYPE_CONFIGS:
                self.filelogger.error(f"Unsupported task_type: {self.task_type}, currently only support {TASK_TYPE_CONFIGS}")
                return -1

            if not self.supported_resolutions:
                self.filelogger.error("supportedResolutions is not set in config.")
                return -1

            self.filelogger.info(f"base_model: {self.base_model}, model_name: {self.model_name}, task_type: {self.task_type}")

            port = _get_free_port()
            self.server_url = f"http://127.0.0.1:{port}"

            _setup_health_status_probe(port, self.filelogger)
            _write_port_file(os.environ.get("MAAS_PORT_FILE", "/home/aiges/maas_port"), port, self.filelogger)

            self.sglang_proc = _launch_sglang_serve(self.base_model, port, self.filelogger)
            try:
                _wait_server_ready(self.server_url, self.filelogger, timeout_s=int(os.environ.get("SGLANG_READY_TIMEOUT_S", "600")))
            except Exception as e:
                self.filelogger.error(f"sglang server not ready: {e}")
                return -1

            self.client = OpenAI(base_url=self.server_url + "/v1", api_key="maas")
            self.thread_pool = ThreadPool(num_threads=self.thread_pool_size, wrapper=self)
            self.filelogger.info("WrapperInit successfully!")
            return 0

        except Exception as e:
            self.filelogger.error(f"WrapperInit error: {e}")
            return -1

    def wrapperOnceExecAsync(self, params: Dict, reqData: DataListCls, usrTag: str = "9527", persId: int = 0) -> int:
        """非流式异步接口：接收请求后立即返回，图片生成完成后通过 callback 返回结果"""
        try:
            # 解析 raw_req
            raw_req = _parse_raw_req(reqData, self.filelogger)
            if not raw_req:
                self.filelogger.error("raw_req is empty or invalid")
                res = Response()
                res = res.response_err(ERROR_INVALID_PARAMS)
                callback(res, usrTag)
                return -1

            prompt = raw_req.get("prompt")
            if not prompt:
                self.filelogger.error("prompt is required in raw_req")
                res = Response()
                res = res.response_err(ERROR_INVALID_PARAMS)
                callback(res, usrTag)
                return -1

            url = raw_req.get("url") or None

            requestInfo = RequestInfo("", params, usrTag)
            requestInfo.raw_req = raw_req

            thread_id = self.thread_pool.alloc_min_thread()
            inferenceInfo = PromptInferenceInfo(self, prompt, requestInfo, url)

            self.thread_pool.put_task(thread_id, inferenceInfo)

            self.filelogger.info(f"wrapperOnceExecAsync queued, request_id:{inferenceInfo.request_id}, usrTag:{usrTag}")
            return 0

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"wrapperOnceExecAsync failed: {e}")
            res = Response()
            res = res.response_err(ERROR_DOWNSTREAM_FAILED)
            callback(res, usrTag)
            return -1

    async def create_image_task(self, inferenceInfo: PromptInferenceInfo):
        """线程池 worker：调用 sglang images API，完成后通过 callback 返回"""
        requestInfo = inferenceInfo.requestInfo
        request_id = inferenceInfo.request_id
        user_tag = requestInfo.user_tag
        raw_req = requestInfo.raw_req

        try:
            prompt = inferenceInfo.prompt

            # 解析参数
            size = raw_req.get("size", "1024x1024")
            negative_prompt = raw_req.get("negative_prompt") or None
            url = inferenceInfo.url
            if url:
                url = _normalize_url(url, self.filelogger)

            seed = raw_req.get("seed")
            if seed is not None and seed != "":
                seed = int(seed)
            else:
                seed = None
            num_inference_steps = raw_req.get("num_inference_steps")
            if num_inference_steps is not None and num_inference_steps != "":
                num_inference_steps = int(num_inference_steps)
            else:
                num_inference_steps = None
            guidance_scale = raw_req.get("guidance_scale")
            if guidance_scale is not None and guidance_scale != "":
                guidance_scale = float(guidance_scale)
            else:
                guidance_scale = None

            # 校验尺寸
            supported_sizes = set()
            model_resolutions = self.supported_resolutions.get(self.model_name, [])
            for res in model_resolutions:
                supported_sizes.update(RESOLUTIONS_TOSIZE.get(res, []))

            if supported_sizes and size not in supported_sizes:
                ori_size = size
                size = "1024x1024"
                self.filelogger.warning(f"{self.model_name} unsupported size {ori_size}, use {size} instead.")

            # 构建 API 请求参数
            req_params = {
                "model": self.model_name,
                "prompt": prompt,
                "n": 1,
                "size": size,
                "response_format": "b64_json",
            }
            extra_body = {}
            if negative_prompt:
                extra_body["negative_prompt"] = negative_prompt
            if seed is not None:
                extra_body["seed"] = seed
            if num_inference_steps is not None:
                extra_body["num_inference_steps"] = num_inference_steps
            if guidance_scale is not None:
                extra_body["guidance_scale"] = guidance_scale
            req_params["extra_body"] = extra_body

            # 调用 OpenAI SDK
            if self.task_type == "i2i":
                # I2I: images.edit — 需要 url 作为 image 参数
                if not url:
                    self.filelogger.error("i2i task requires url in raw_req")
                    res = Response()
                    res = res.response_err(ERROR_INVALID_PARAMS)
                    callback(res, user_tag)
                    return
                req_params["image"] = url
                self.filelogger.info(f"Creating i2i (image-edit) task, prompt: {prompt[:100]}...")
                self.filelogger.debug(f"i2i task params: {str(req_params)[:1000]}")
                rsp = await asyncio.to_thread(self.client.images.edit, **req_params)
            else:
                # T2I: images.generate
                self.filelogger.info(f"Creating t2i (text-to-image) task, prompt: {prompt[:100]}...")
                self.filelogger.debug(f"t2i task params: {str(req_params)[:1000]}")
                rsp = await asyncio.to_thread(self.client.images.generate, **req_params)

            if not rsp.data or len(rsp.data) == 0:
                self.filelogger.error("Image generation API returned empty data")
                res = Response()
                res = res.response_err(ERROR_DOWNSTREAM_FAILED)
                callback(res, user_tag)
                return

            # 解码 b64_json → 图片二进制
            b64_str = rsp.data[0].b64_json
            if not b64_str:
                self.filelogger.error("Image generation API returned no b64_json")
                res = Response()
                res = res.response_err(ERROR_DOWNSTREAM_FAILED)
                callback(res, user_tag)
                return

            img_bytes = base64.b64decode(b64_str)

            # 构建返回 Response
            res = Response()
            resd = ResponseData()
            resd.key = "raw_resp"
            resd.setDataType(DataImage)
            resd.status = Once
            resd.setData(img_bytes)
            res.list = [resd]

            ret = callback(res, user_tag)
            if ret not in (0, None):
                self.filelogger.error(f"callback failed for {user_tag}, ret code:{ret}")

            self.filelogger.info(f"Image generation completed, request_id:{request_id}, size:{len(img_bytes)} bytes")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"create_image_task failed: {e}")
            res = Response()
            res = res.response_err(ERROR_DOWNSTREAM_FAILED)
            callback(res, user_tag)

    def wrapperOnceExec(self, params: Dict, reqData: DataListCls, usrTag: str = "", persId: int = 0) -> Response:
        pass

    def wrapperCreate(self, params: {}, sid: str, persId: int = 0, usrTag: str = "") -> SessionCreateResponse:
        pass

    def wrapperWrite(self, handle: str, req: DataListCls) -> int:
        pass

    def wrapperDestroy(self, handle: str) -> int:
        pass

    def wrapperRead(self, handle: str):
        pass

    def wrapperFini(self) -> int:
        if self.thread_pool is not None:
            self.thread_pool.wait_completion()
        return 0

    @classmethod
    def wrapperError(cls, ret: int) -> str:
        error_messages = {
            ERROR_DOWNSTREAM_FAILED: "Downstream call failed (images.generate)",
            ERROR_INVALID_PARAMS: "Invalid parameters (missing prompt, bad size)",
            ERROR_SERVER_NOT_READY: "sglang server not ready",
            ERROR_POLL_TIMEOUT: "Poll timeout",
            ERROR_IMAGE_READ_FAILED: "Image read failed (URL/path/binary)",
        }
        return error_messages.get(ret, f"Unknown error: {ret}")

    @classmethod
    def wrapperTestFunc(cls, data: list, respData: list):
        pass


if __name__ == '__main__':
    m = Wrapper()
    m.run()
