#!/usr/bin/env python
# coding:utf-8
import os
import logging
import json
import threading
import asyncio
import queue
import enum
import uuid
import subprocess
import socket
import time
import requests

from typing import Dict
from openai import OpenAI

from aiges.core.types import *
try:
    from aiges_embed import ResponseData, Response, DataListNode, SessionCreateResponse, DataListCls, callback # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, SessionCreateResponse, DataListCls, callback

from aiges.sdk import WrapperBase
from aiges.utils.log import getFileLogger

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TASK_TYPE_CONFIGS = ["t2v", "i2v"]
RESOLUTIONS_TOSIZE = {
    "480P": ["832x480", "480x832", "624x624"],
    "720P": ["1280x720", "720x1280", "960x960", "1088x832", "832x1088"]
}

# Error codes
ERROR_DOWNSTREAM_FAILED = 1001
ERROR_INVALID_PARAMS = 1002
ERROR_SERVER_NOT_READY = 1003
ERROR_POLL_TIMEOUT = 1004
ERROR_IMAGE_READ_FAILED = 1005

# Constants
DataNone = -1
DataBegin = 0
DataContinue = 1
DataEnd = 2

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
    # shell=True 与参考实现保持一致（便于 extra args），由部署环境控制注入内容
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


def _normalize_reference_url(reference_url: str, filelogger) -> str:
    """
    规范化 reference_url，支持两种输入格式：
    1. http(s) 图像链接 — 原样返回
    2. base64 数据（无 data: 前缀） — 补全 data:image/png;base64, 前缀
    """
    if reference_url.startswith("http://") or reference_url.startswith("https://"):
        return reference_url
    if reference_url.startswith("data:"):
        return reference_url
    # 裸 base64 字符串，补全 data URI 前缀
    filelogger.info("reference_url is raw base64, prepended data:image/png;base64, prefix")
    return f"data:image/png;base64,{reference_url}"

def resp_content(status, output_json: dict):
    resd = ResponseData()
    resd.key = "raw_resp"
    resd.setDataType(DataText)
    resd.status = status
    resd.setData(json.dumps(output_json, ensure_ascii=False).encode("utf-8"))
    return resd


class RequestMode(enum.Enum):
    ONCE = "once"
    ONCE_ASYNC = "once_async"
    STREAM = "stream"

class RequestInfo:
    def __init__(self, sid: str, params: dict, user_tag: str = ""):
        self.handle = str(uuid.uuid4().hex)
        self.sid = sid
        self.user_tag = user_tag
        self.params = params
        self.requests = []
        self.raw_req_chunks = []        # raw_req 分片缓存
        self.raw_req = None             # 解析后的 raw_req dict
        self.stop_event = threading.Event()  # 取消信号（替代 stop_q + cancelled）
        self.finished_event = threading.Event()  # 任务已进入终态
        self.end_sent_event = threading.Event()  # 已发送 DataEnd（防止重复）
        self._end_sent_lock = threading.Lock()  # _send_end_response 原子化锁
        self.out_q = queue.Queue(maxsize=10)  # 有限大小，防止内存膨胀
        self._destroy_pending = False  # wrapperDestroy 已调用，待清理
        self.task_status = "idle"  # idle / queued / processing / completed / failed

class PromptInferenceInfo:
    def __init__(self, wrapper,
                 thread_id: str,
                 mode: RequestMode,
                 prompt: str,
                 requestInfo: RequestInfo,
                 result_q: queue.Queue = None):
        self.wrapper = wrapper
        self.requestInfo = requestInfo
        self.thread_id = thread_id
        self.mode = mode
        self.prompt = prompt
        self.request_id = str(uuid.uuid4().hex)
        self.result_q = result_q

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
        # 创建并运行事件循环
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
                asyncio.create_task(task.wrapper.create_video_task(task))
                # 让出cpu
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

# 定义服务推理逻辑
class Wrapper(WrapperBase):
    version = "v1"
    model = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        LEVEL = getattr(logging, LOG_LEVEL, logging.INFO)
        self.filelogger = getFileLogger(level=LEVEL)
        self.base_model: str = None
        self.base_url: str = None
        self.api_key: str = None
        self.model_name: str = None
        self.task_type: str = None
        self.thread_pool: ThreadPool = None
        self.request_map: dict[str, RequestInfo] = {}
        self.request_map_lock: threading.Lock = threading.Lock()
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
                os.environ['POLL_INTERVAL_MS'] = config.get("pollIntervalMs", "5000")
                
                self.model_name = config.get("modelName", "")
                self.task_type = config.get("modelTaskType", "t2v")
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
                self.filelogger.error(f"supportedResolutions is not set in config.")
                return -1
            self.filelogger.info(f"base_model: {self.base_model}, model_name: {self.model_name}, task_type: {self.task_type}")

            port = _get_free_port()
            self.server_port = port
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

    def wrapperCreate(self, params: {}, sid: str, persId: int = 0, usrTag: str = "") -> SessionCreateResponse:
        self.filelogger.info(f"start wrapperCreate {params}")
        requestInfo = RequestInfo(sid, params, usrTag)
        with self.request_map_lock:
            self.request_map[requestInfo.handle] = requestInfo

        s = SessionCreateResponse()
        s.handle = requestInfo.handle
        s.error_code = 0
        self.filelogger.debug(f"success wrapperCreate, handle: {s.handle}, params: {params}")
        return s

    def wrapperWrite(self, handle: str, reqData: DataListCls) -> int:
        self.filelogger.debug(f"start wrapperWrite, handle: {handle}, reqData: {reqData}")
        try:
            with self.request_map_lock:
                requestInfo = self.request_map.get(handle)
                if not requestInfo:
                    self.filelogger.error(f"handle not found: {handle}")
                    return -1

            # --- raw_req 流式接收 ---
            raw_req_node = reqData.get("raw_req")
            if raw_req_node:
                raw_req_data = raw_req_node.data
                raw_req_status = raw_req_node.status

                # 缓存本次收到的 raw_req 分片
                # aiges 框架保证同一 handle 的 wrapperWrite 串行调用，无需加锁
                # data 可能是 str 或 bytes，统一转为 str
                if isinstance(raw_req_data, bytes):
                    raw_req_data = raw_req_data.decode("utf-8")
                self.filelogger.debug(f"raw_req_data: {raw_req_data}")
                requestInfo.raw_req_chunks.append(raw_req_data)

                # raw_req 尚未传输完毕，仅缓存
                if raw_req_status != DataEnd:
                    self.filelogger.debug(
                        f"raw_req chunk received, status={raw_req_status}, "
                        f"accumulated chunks={len(requestInfo.raw_req_chunks)}, handle: {handle}")
                    return 0

                # DataEnd: raw_req 传输完毕，拼合并解析 JSON
                raw_req_str = ''.join(requestInfo.raw_req_chunks)
                requestInfo.raw_req_chunks.clear()
                self.filelogger.debug(f"raw_req_str: {raw_req_str[:500]}, handle: {handle}")
                try:
                    parsed = json.loads(raw_req_str)
                    # 处理双重 JSON 编码：如果解析结果仍是字符串，再解析一次
                    if isinstance(parsed, str):
                        self.filelogger.warning(f"raw_req is double-encoded JSON, parsing again, handle: {handle}")
                        parsed = json.loads(parsed)
                    requestInfo.raw_req = parsed
                    self.filelogger.info(f"raw_req parsed successfully, handle: {handle}")
                    if not isinstance(requestInfo.raw_req, dict):
                        self.filelogger.error(f"raw_req is not a dict, got {type(requestInfo.raw_req).__name__}: {str(requestInfo.raw_req)[:200]}, handle: {handle}")
                        requestInfo.task_status = "failed"
                        self._send_end_response(requestInfo, {"status": "failed", "error": f"raw_req is not a dict, got {type(requestInfo.raw_req).__name__}"})
                        return -1
                except Exception as e:
                    self.filelogger.error(f"Failed to parse raw_req JSON: {e}")
                    requestInfo.task_status = "failed"
                    self._send_end_response(requestInfo, {"status": "failed", "error": f"Failed to parse raw_req JSON: {e}"})
                    return -1
            else:
                # 非 raw_req 帧，跳过（数据尚未到达）
                self.filelogger.debug(f"wrapperWrite: no raw_req node in this frame, handle: {handle}")
                return 0

            # 校验 raw_req
            prompt = requestInfo.raw_req.get("prompt")
            if not prompt:
                self.filelogger.error("prompt is required in raw_req")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": "prompt is required in raw_req"})
                return -1

            thread_id = self.thread_pool.alloc_min_thread()
            inferenceInfo = PromptInferenceInfo(self, thread_id, RequestMode.STREAM, prompt, requestInfo)

            # 先修改 requestInfo 内部字段，再投递任务（保证可见性）
            requestInfo.handle = handle
            requestInfo.requests.append(inferenceInfo.request_id)

            self.thread_pool.put_task(thread_id, inferenceInfo)

            self.filelogger.debug(
                f"success wrapperWrite handle:{handle}, thread_id:{thread_id},request_id:{inferenceInfo.request_id}")
            return 0

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"wrapperWrite failed: {e}")
            return -1

    def wrapperRead(self, handle: str) -> Response:
        """
        同步读取结果, 对应 aiges 配置 asyncMode = false
        阻塞等待 out_q 数据: 有结果立即返回, 60s 超时返回异常
        """
        # time.sleep(5)
        with self.request_map_lock:
            requestInfo = self.request_map.get(handle)
        if not requestInfo:
            self.filelogger.error(f"wrapperRead handle not found: {handle}")
            r = Response()
            r = r.response_err(ERROR_INVALID_PARAMS)
            return r

        # 检查是否已取消（wrapperDestroy 已调用）
        if requestInfo.stop_event.is_set() and not requestInfo.end_sent_event.is_set():
            # 发送 cancelled DataEnd 兜底
            self._send_end_response(requestInfo, {"status": "cancelled", "error": "Session destroyed"})

        # 阻塞等待最多 60s，等 out_q 有数据立即返回
        try:
            rs = requestInfo.out_q.get(timeout=60)
        except queue.Empty:
            # 60s 超时无数据，返回超时异常
            self.filelogger.error(f"wrapperRead 60s timeout, handle: {handle}")
            r = Response()
            r = r.response_err(ERROR_POLL_TIMEOUT)
            return r

        if not isinstance(rs, Response):
            self.filelogger.error(f"wrapperRead invalid response type: {type(rs)}")
            r = Response()
            r = r.response_err(ERROR_DOWNSTREAM_FAILED)
            return r

        # DataEnd 表示本次会话结果读取完毕，清理 request_map
        is_end = any(rd.status == DataEnd for rd in rs.list) if rs.list else False
        if is_end:
            self.filelogger.info(f"wrapperRead session done, handle: {handle}")
            with self.request_map_lock:
                self.request_map.pop(handle, None)

        return rs

    def wrapperDestroy(self, handle: str) -> int:
        self.filelogger.debug(f"start wrapperDestroy, handle: {handle}")
        with self.request_map_lock:
            ri = self.request_map.get(handle)
            if ri is None:
                # handle 已被 wrapperRead 在 DataEnd 后清理，会话正常结束
                self.filelogger.info(f"wrapperDestroy: handle already cleaned up, handle: {handle}")
                return 0
            # 软取消：通知 worker 停止
            ri.stop_event.set()
            ri._destroy_pending = True
            # 如果 worker 已完成，直接删除；否则保留给 wrapperRead 兜底
            if ri.finished_event.is_set():
                del self.request_map[handle]
                self.filelogger.info(f"wrapperDestroy, worker already finished, handle: {handle}")
        self.filelogger.info(f"success wrapperDestroy, handle: {handle}")
        return 0


    def wrapperOnceExecAsync(self, params: Dict, reqData: DataListCls, usrTag: str = "9527", persId: int = 0):
        pass

    def wrapperOnceExec(self, params: Dict, reqData: DataListCls, usrTag: str = "", persId: int = 0) -> Response:
        pass

    def _response_err(self, error_code: int, message: str = "") -> Response:
        """创建错误响应"""
        res = Response()
        res = res.response_err(error_code)
        return res

    def _send_response(self, res: Response, requestInfo: RequestInfo):
        """推送到 out_q 供 wrapperRead 同步读取，队列满时丢弃旧进度数据"""
        try:
            requestInfo.out_q.put_nowait(res)
        except queue.Full:
            # 队列满：丢弃最旧的 DataContinue 进度数据，保留 DataBegin/DataEnd
            for _ in range(requestInfo.out_q.qsize() + 1):
                try:
                    old = requestInfo.out_q.get_nowait()
                    # 如果是终态数据，重新放回
                    if isinstance(old, Response) and old.list and any(rd.status == DataEnd for rd in old.list):
                        try:
                            requestInfo.out_q.put_nowait(old)
                        except queue.Full:
                            self.filelogger.error(f"out_q full when putting back DataEnd, handle: {requestInfo.handle}")
                except queue.Empty:
                    break
            try:
                requestInfo.out_q.put_nowait(res)
            except queue.Full:
                self.filelogger.warning(f"out_q still full after drain, dropping response for handle: {requestInfo.handle}")

    def _send_end_response(self, requestInfo: RequestInfo, video_info: dict):
        """发送 DataEnd 终态响应，保证只发一次（原子化 check-and-set）"""
        # end_sent_event 初始为 False，set() 返回 True 表示由本线程设置
        # 如果已经是 True，说明其他线程已发送过，直接返回
        if not requestInfo._end_sent_lock.acquire(blocking=False):
            # 另一个线程正在发送，等待其完成后返回
            requestInfo._end_sent_lock.acquire()
            requestInfo._end_sent_lock.release()
            return
        try:
            if requestInfo.end_sent_event.is_set():
                return
            requestInfo.end_sent_event.set()
            requestInfo.finished_event.set()
            res = Response()
            content = resp_content(DataEnd, video_info)
            res.list = [content]
            self._send_response(res, requestInfo)
        finally:
            requestInfo._end_sent_lock.release()

    async def create_video_task(self, inferenceInfo: PromptInferenceInfo):
        requestInfo = inferenceInfo.requestInfo
        request_id = inferenceInfo.request_id
        user_tag = requestInfo.user_tag
        sid = requestInfo.sid

        try:
            # 检查是否已取消
            if requestInfo.stop_event.is_set():
                self.filelogger.info(f"====>inference abort before infer, {request_id}")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": "Task aborted"})
                return

            raw_req = requestInfo.raw_req
            if not isinstance(raw_req, dict):
                self.filelogger.error(f"raw_req is not dict, type={type(raw_req).__name__}, value={str(raw_req)[:200]}")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": f"raw_req is not a dict, got {type(raw_req).__name__}"})
                return
            prompt = inferenceInfo.prompt

            # 从 raw_req 解析各字段
            size = raw_req.get("size", "1280x720")
            negative_prompt = raw_req.get("negative_prompt") or None
            reference_url = raw_req.get("reference_url") or None
            # 规范化 reference_url：支持 http(s) 链接 和 base64 数据
            if reference_url:
                reference_url = _normalize_reference_url(reference_url, self.filelogger)
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
            seconds = raw_req.get("seconds")
            if seconds is not None and seconds != "":
                seconds = int(seconds)
            else:
                seconds = None
            fps = raw_req.get("fps")
            if fps is not None and fps != "":
                fps = int(fps)
            else:
                fps = None

            # 校验尺寸
            supported_sizes = set()
            model_resolutions = self.supported_resolutions.get(self.model_name, [])
            for res in model_resolutions:
                supported_sizes.update(RESOLUTIONS_TOSIZE.get(res, []))

            if size not in supported_sizes:
                ori_size = size
                size = "1280x720" if self.model_name not in ["wan2.1-t2v-1.3b"] else "832x480"
                self.filelogger.warning(f"{self.model_name} unsupported size {ori_size}, use {size} instead.")

            self.filelogger.info(f"Creating video generation task, prompt: {prompt[:100]}...")
            requestInfo.task_status = "queued"
            try:
                req_params = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "size": size,
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
                if fps is not None:
                    extra_body["fps"] = fps

                if seconds is not None:
                    req_params["seconds"] = seconds
                req_params["extra_body"] = extra_body

                # I2V 需要传入图片
                if self.task_type == "i2v":
                    if not reference_url:
                        self.filelogger.error("i2v task requires reference_url in raw_req")
                        requestInfo.task_status = "failed"
                        self._send_end_response(requestInfo, {"status": "failed", "error": "i2v task requires reference_url in raw_req"})
                        return
                    extra_body["reference_url"] = reference_url

                # 再次检查取消
                if requestInfo.stop_event.is_set():
                    self.filelogger.info(f"Task cancelled before API call, {request_id}")
                    requestInfo.task_status = "failed"
                    self._send_end_response(requestInfo, {"status": "failed", "error": "Task cancelled"})
                    return

                self.filelogger.debug(f"video task params: {str(req_params)[:1000]}")
                rsp = await asyncio.to_thread(self.client.videos.create, **req_params)

                if rsp.status == "failed":
                    self.filelogger.error(
                        f"Failed to create task, ase_sid:{sid}, ase_request_id:{request_id}, request_id:{rsp.id}."
                    )
                    requestInfo.task_status = "failed"
                    self._send_end_response(requestInfo, {"status": "failed", "error": "Failed to create video task"})
                    return

                task_id = rsp.id
                requestInfo.task_status = rsp.status
                self.filelogger.info(f"Video task created, task_id: {task_id}")

                # 发送 DataBegin 状态
                res_begin = Response()
                video_info = {"request_id": request_id}
                video_info.update(rsp.model_dump())
                content = resp_content(DataBegin, video_info)
                res_begin.list = [content]
                self._send_response(res_begin, requestInfo)

                # 轮询获取结果（终态已由 poll 内部通过 _send_end_response 发送）
                await self.poll_video_task_async(task_id, sid, request_id, requestInfo)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.filelogger.error(f"An error occurred when infer: {e}, ret {ERROR_DOWNSTREAM_FAILED}")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": str(e)})

        except Exception as e:
            # 外层兜底：任何未预期的异常
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"Unexpected error in create_video_task: {e}", exc_info=True)
            requestInfo.task_status = "failed"
            self._send_end_response(requestInfo, {"status": "failed", "error": f"Unexpected error: {e}"})

    async def poll_video_task_async(self, task_id: str, sid: str, request_id: str, requestInfo: RequestInfo) -> Response:
        """
        异步轮询视频任务状态，直到完成或超时
        注意：此函数返回的 Response 由调用方(_send_response)发送，终态 DataEnd 由 _send_end_response 统一处理
        """
        self.filelogger.info(f"Waiting video task result, task_id: {task_id}")

        poll_interval_ms = int(os.environ.get("POLL_INTERVAL_MS", "5000"))
        poll_timeout_s = int(os.environ.get("POLL_TIMEOUT_S", "1800"))
        deadline = time.time() + poll_timeout_s

        last_status = None
        while True:
            # 检查取消
            if requestInfo.stop_event.is_set():
                self.filelogger.info(f"Task cancelled during poll, task_id: {task_id}")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": "Task cancelled"})
                return None  # 已由 _send_end_response 发送终态

            # 检查超时
            if time.time() > deadline:
                self.filelogger.error(f"Poll timeout for task_id: {task_id}")
                requestInfo.task_status = "failed"
                self._send_end_response(requestInfo, {"status": "failed", "error": f"Poll timeout after {poll_timeout_s}s"})
                return None

            try:
                rsp = await asyncio.to_thread(self.client.videos.retrieve, video_id=task_id)

                current_status = rsp.status
                progress = getattr(rsp, 'progress', 0)
                requestInfo.task_status = current_status

                if current_status != last_status:
                    self.filelogger.info(f"Task {task_id} status: {current_status}, progress: {progress}")
                    last_status = current_status

                if current_status == "completed" and str(progress) == "100":
                    requestInfo.task_status = "completed"
                    self.filelogger.info(f"Video generation completed, task_id: {task_id}, inference_time: {getattr(rsp, 'inference_time_s', None)}")

                    video_info = {"request_id": request_id}
                    video_info.update(rsp.model_dump())

                    self._send_end_response(requestInfo, video_info)
                    return None  # 已由 _send_end_response 发送终态

                elif current_status == "failed":
                    requestInfo.task_status = "failed"
                    error_msg = getattr(rsp, 'error', None) or "unknown"
                    self.filelogger.error(f"Video task failed, task_id: {task_id}, error: {error_msg}")
                    self._send_end_response(requestInfo, {"status": "failed", "error": f"Video generation failed: {error_msg}"})
                    return None

                else:
                    # 发送进度更新 DataContinue
                    res_continue = Response()
                    video_info = {"request_id": request_id}
                    video_info.update(rsp.model_dump())

                    content = resp_content(DataContinue, video_info)
                    res_continue.list = [content]
                    self._send_response(res_continue, requestInfo)

            except Exception as e:
                self.filelogger.error(f"Error polling task {task_id}: {e}")

            await asyncio.sleep(poll_interval_ms / 1000.0)


    def wrapperFini(self) -> int:
        if self.thread_pool is not None:
            self.thread_pool.wait_completion()
        return 0

    @classmethod
    def wrapperError(cls, ret: int) -> str:
        error_messages = {
            ERROR_DOWNSTREAM_FAILED: "Downstream call failed (videos.create/retrieve)",
            ERROR_INVALID_PARAMS: "Invalid parameters (missing prompt, bad size)",
            ERROR_SERVER_NOT_READY: "sglang server not ready",
            ERROR_POLL_TIMEOUT: "Poll timeout",
            ERROR_IMAGE_READ_FAILED: "Image read failed (URL/path/binary)",
        }
        return error_messages.get(ret, f"Unknown error: {ret}")

    '''
        此函数保留测试用，不可删除
    '''

    @classmethod
    def wrapperTestFunc(cls, data: list, respData: list):
        pass


if __name__ == '__main__':
    m = Wrapper()
    m.run()
