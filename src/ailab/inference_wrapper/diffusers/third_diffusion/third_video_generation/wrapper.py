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
import base64
import io
import dashscope

from PIL import Image
from http import HTTPStatus
from typing import List, Dict
from dashscope import VideoSynthesis

from aiges.core.types import *
try:
    from aiges_embed import ResponseData, Response, DataListNode, DataListCls, callback  # c++
except:
    from aiges.dto import Response, ResponseData, DataListNode, DataListCls, callback

from aiges.sdk import WrapperBase
from aiges.utils.log import getFileLogger

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

TASK_TYPE_CONFIGS = ["t2v", "i2v"]
WAN_CONFIGS = ["wan2.6-t2v", "wan2.5-t2v-preview", "wan2.2-t2v-plus", "wan2.6-i2v", "wan2.5-i2v-preview", "wan2.2-i2v-flash", "wan2.2-i2v-plus"]
RESOLUTIONS_TOSIZE = {
    "480P": ["832*480", "480*832", "624*624"],
    "720P": ["1280*720", "720*1280", "960*960", "1088*832", "832*1088"],
    "1080P": ["1920*1080", "1080*1920", "1440*1440", "1632*1248", "1248*1632"]
}
SUPPORTED_RESOLUTIONS = {
    "wan2.6-t2v": ["720P", "1080P"],
    "wan2.5-t2v-preview": ["480P", "720P", "1080P"],
    "wan2.2-t2v-plus": ["480P", "1080P"],
    "wan2.6-i2v": ["720P", "1080P"],
    "wan2.5-i2v-preview": ["480P", "720P", "1080P"],
    "wan2.2-i2v-flash": ["480P", "720P", "1080P"],
    "wan2.2-i2v-plus": ["480P", "1080P"]
}

def get_payload_text(reqData: DataListCls, filelogger):
    try:
        text = reqData.get("video_description").data.decode("utf-8")
        return text
    except Exception as e:
        filelogger.info(f"get video_description error: {e}")
        return None

def get_payload_image_url(reqData: DataListCls, log):
    try:
        image_bytes = reqData.get("image_url").data
        mime_type = get_image_format(image_bytes, log)
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/{mime_type};base64,{image_base64}"
        log.info(f"get image_url success: {str(data_url)[:200]}")
        return data_url
    except Exception as e:
        log.error(f"get image_url error: {e}")
        return None

def get_image_format(image_bytes, log) -> str:
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = img.format.lower()
        fmt = "jpeg" if fmt == "jpg" else fmt
        return fmt
    except Exception as e:
        log.error(f"Cannot determine image format: {e}")
        return "png"

def get_params_width(params: Dict):
    return int(params.get("width", 1280))

def get_params_height(params: Dict):
    return int(params.get("height", 720))

def get_params_duration(params: Dict):
    return int(params.get("duration", 5))

def get_params_resolution(params: Dict):
    return str(params.get("resolution", "720P"))

def get_params_prompt_extend(params: Dict):
    return str(params.get("prompt_extend", True)).lower() == "true"

def get_params_waterDisable(params: Dict):
    return str(params.get("waterDisable", True)).lower() == "true"

def get_params_shot_type(params: Dict):
    return str(params.get("shot_type", "single")).lower()

def get_params_seed(params: Dict):
    return int(params.get("seed", 1234))

def get_params_audio_url(params: Dict):
    return params.get("audio_url", None)


def resp_content(status, output_json: dict):
    resd = ResponseData()
    resd.key = "video"
    resd.setDataType(DataText)
    resd.status = status
    resd.setData(json.dumps(output_json, ensure_ascii=False).encode("utf-8"))
    return resd

def resp_usage(status, usage_json: dict):
    resd = ResponseData()
    resd.key = "usage"
    resd.setDataType(DataText)
    resd.status = status
    resd.setData(json.dumps(usage_json, ensure_ascii=False).encode("utf-8"))
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
        self.stop_q = queue.Queue()

class PromptInferenceInfo:
    def __init__(self, wrapper,
                 thread_id: str,
                 mode: RequestMode,
                 prompt: str,
                 requestInfo: RequestInfo,
                 img_url: str = None,
                 result_q: queue.Queue = None):
        self.wrapper = wrapper
        self.requestInfo = requestInfo
        self.thread_id = thread_id
        self.mode = mode
        self.prompt = prompt
        self.img_url = img_url
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
        self.task_type: str = None
        self.thread_pool: ThreadPool = None
        self.thread_pool_size: int = 1
    
    def wrapperInit(self, config: Dict) -> int:
        self.filelogger.info("Initializing ...")
        try:
            if not config:
                self.filelogger.warning("Config is empty, using default values")
                return 0
            
            if "base_model" in config:
                self.base_model = config["base_model"]
            if not self.base_model or self.base_model not in WAN_CONFIGS:
                self.filelogger.error(f"Unsupported base_model: {self.base_model}")
                return -1
            self.filelogger.info(f"base_model: {self.base_model}")

            if "task_type" in config:
                self.task_type = config["task_type"]
            if not self.task_type or self.task_type not in TASK_TYPE_CONFIGS:
                self.filelogger.error(f"Unsupported task_type: {self.task_type}, currently only support {TASK_TYPE_CONFIGS}")
                return -1
            
            if "base_url" in config:
                self.base_url = config["base_url"]
            
            if "api_key" in config:
                self.api_key = config["api_key"]

            dashscope.base_http_api_url = self.base_url
            self.thread_pool = ThreadPool(num_threads=self.thread_pool_size, wrapper=self)
            self.filelogger.info("WrapperInit successfully!")
            return 0
            
        except Exception as e:
            self.filelogger.error(f"WrapperInit error: {e}")
            return -1


    def wrapperOnceExecAsync(self, params: Dict, reqData: DataListCls, usrTag: str = "9527", persId: int = 0):
        try:
            prompt = get_payload_text(reqData, self.filelogger)
            img_url = None
            if "i2v" in self.task_type:
                img_url = get_payload_image_url(reqData, self.filelogger)
            requestInfo = RequestInfo("", params, usrTag)
            thread_id = self.thread_pool.alloc_min_thread()
            inferenceInfo = PromptInferenceInfo(self, thread_id, RequestMode.ONCE_ASYNC, prompt, requestInfo, img_url)
            self.filelogger.info(f"start wrapperOnceExecAsync params:{params}, request_id:{inferenceInfo.request_id}")

            self.thread_pool.put_task(thread_id, inferenceInfo)
            self.filelogger.debug(
                f"success wrapperOnceExecAsync, params:{params}, prompt:{prompt}, request_id:{inferenceInfo.request_id}")
            return 0

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"wrapperOnceExecAsync failed: {e}")
            return -1

    def wrapperOnceExec(self, params: Dict, reqData: DataListCls, usrTag: str = "", persId: int = 0) -> Response:
        pass

    async def create_video_task(self, inferenceInfo: PromptInferenceInfo):
        requestInfo = inferenceInfo.requestInfo
        request_id = inferenceInfo.request_id
        user_tag = requestInfo.user_tag
        sid = requestInfo.sid
        is_stoped = False
        if not requestInfo.stop_q.empty():
            is_stoped = requestInfo.stop_q.get_nowait()
        if is_stoped:
            state = Once
            content = resp_content(state, {})
            usage = resp_usage(state, {})
            res = Response()
            res.list = [content, usage]
            if inferenceInfo.mode == RequestMode.ONCE_ASYNC:
                callback(res, user_tag)
            self.filelogger.info(f"====>inference abort before infer, {request_id}")
            return 0

        params = requestInfo.params
        prompt = inferenceInfo.prompt
        img_url = inferenceInfo.img_url

        width = get_params_width(params)
        height = get_params_height(params)
        size = f"{width}*{height}"

        supported_sizes = set()
        for res in SUPPORTED_RESOLUTIONS[self.base_model]:
            supported_sizes.update(RESOLUTIONS_TOSIZE[res])

        if size not in supported_sizes:
            ori_size = size
            size = "1280*720" if self.base_model not in ["wan2.2-t2v-plus"] else "832*480"
            self.filelogger.warning(f"{self.base_model} unsupported size {ori_size}, use {size} instead.")
        
        resolution = get_params_resolution(params)
        if resolution not in SUPPORTED_RESOLUTIONS[self.base_model]:
            ori_resolution = resolution
            resolution = "720P" if self.base_model not in ["wan2.2-i2v-plus"] else "480P"
            self.filelogger.warning(f"{self.base_model} unsupported resolution {ori_resolution}, use {resolution} instead.")

        negative_prompt = params.get("negative_prompt", None)
        duration = get_params_duration(params)
        prompt_extend = get_params_prompt_extend(params)
        waterDisable = get_params_waterDisable(params)
        shot_type = get_params_shot_type(params)
        seed = get_params_seed(params)
        audio_url = get_params_audio_url(params)

        self.filelogger.info("Creating video generation task...")
        try:
            req_params = {
                "api_key": self.api_key,
                "model": self.base_model,
                "prompt": prompt,
                "duration": duration,
                "prompt_extend": prompt_extend,
                "watermark": not waterDisable,
                "seed": seed
            }
            if self.task_type == "t2v":
                req_params["size"] = size
                if audio_url is not None and self.base_model not in ["wan2.2-t2v-plus"]:
                    req_params["audio_url"] = audio_url
                if self.base_model in ["wan2.6-t2v"]:
                    req_params["shot_type"] = shot_type
                if negative_prompt:
                    req_params["negative_prompt"] = str(negative_prompt)
                self.filelogger.debug(f"video task params: {req_params}")
                rsp = VideoSynthesis.async_call(**req_params)
            
            elif self.task_type == "i2v":
                req_params["resolution"] = resolution
                if self.base_model in ["wan2.6-i2v"]:
                    req_params["shot_type"] = shot_type
                if negative_prompt:
                    req_params["negative_prompt"] = str(negative_prompt)
                if audio_url is not None and self.base_model not in ["wan2.2-i2v-flash", "wan2.2-i2v-plus"]:
                    req_params["audio_url"] = audio_url
                req_params["img_url"] = img_url
                self.filelogger.debug(f"video task params: {str(req_params)[:1000]}")
                rsp = VideoSynthesis.async_call(**req_params)
            else:
                raise RuntimeError(f"Unsupported task type:{self.task_type}, currently only support {TASK_TYPE_CONFIGS}")

            if rsp.status_code != HTTPStatus.OK:
                raise RuntimeError(
                    f"Failed to create task, ase_sid:{sid}, ase_request_id:{request_id}, request_id:{rsp.request_id}, code:{rsp.code}, message:{rsp.message}."
                )
            
            task_id = rsp.output.task_id
            self.filelogger.info(f"Video task created, task_id: {task_id}")
            ## 获取结果
            res = await self.poll_video_task_async(task_id, sid, request_id)
            ret = callback(res, user_tag)
            if ret != 0:
                self.filelogger.error(f"wrapperOnceExecAsync callback failed, ret code: {ret}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.filelogger.error(f"An error occurred when infer: {e}, ret 1001")
            res = Response()
            res = res.response_err(1001)
            callback(res, user_tag)
    
    async def poll_video_task_async(self, task_id: str, sid: str, request_id: str) -> Response:
        """
        异步版本的 poll_video_task，使用线程池执行阻塞调用，避免阻塞事件循环
        """
        self.filelogger.info(f"Waiting video task result, task_id: {task_id}")

        try:
            # 使用 asyncio.to_thread 在线程池中执行阻塞调用，避免阻塞事件循环
            # 这样可以让多个任务真正并发执行
            loop = asyncio.get_event_loop()
            rsp = await loop.run_in_executor(
                None,
                lambda: VideoSynthesis.wait(
                    api_key=self.api_key,
                    task=task_id
                )
            )

            if rsp.status_code != HTTPStatus.OK or rsp.output.task_status != "SUCCEEDED":
                self.filelogger.error(
                    f"Failed to create task, ase_sid: {sid}, ase_request_id: {request_id}, request_id: {rsp.request_id}, code: {rsp.output.code}, message: {rsp.output.message}."
                )
            self.filelogger.debug(f"Video task result, rsp: {rsp}")
            res = Response()
            status = Once
            content = resp_content(status, rsp.output)
            usage = resp_usage(status, rsp.usage)
            res.list = [content, usage]
            self.filelogger.info(
                f"""Video generation successfully, ase_sid:{sid}, ase_request_id:{request_id}, request_id:{rsp.request_id}, task_status: {rsp.output.task_status}, video_url:{rsp.output.video_url}""")
            return res
        except Exception as e:
            self.filelogger.error(f"Video generation task failed, error: {e}")
            raise

    def poll_video_task(self, task_id: str, sid: str = "", request_id: str = "") -> Response:
        self.filelogger.info(f"Waiting video task result, task_id: {task_id}")

        try:
            rsp = VideoSynthesis.wait(
                api_key=self.api_key,
                task=task_id
            )

            if rsp.status_code != HTTPStatus.OK or rsp.output.task_status != "SUCCEEDED":
                self.filelogger.error(
                    f"Failed to create task, ase_sid: {sid}, ase_request_id: {request_id}, request_id: {rsp.request_id}, code: {rsp.output.code}, message: {rsp.output.message}."
                )
            self.filelogger.debug(f"Video task result, rsp: {rsp}")
            res = Response()
            status = Once
            content = resp_content(status, rsp.output)
            usage = resp_usage(status, rsp.usage)
            res.list = [content, usage]
            self.filelogger.info(
                f"""Video generation successfully, ase_sid:{sid}, ase_request_id:{request_id}, request_id:{rsp.request_id}, task_status: {rsp.output.task_status}, video_url:{rsp.output.video_url}""")
            return res
        except Exception as e:
            self.filelogger.error(f"Video generation task failed, error: {e}")
            return None


    def wrapperFini(self) -> int:
        self.thread_pool.wait_completion()
        return 0

    @classmethod
    def wrapperError(cls, ret: int) -> str:
        if ret == 100:
            return "user error defined here"
        return ""

    '''
        此函数保留测试用，不可删除
    '''

    @classmethod
    def wrapperTestFunc(cls, data: list, respData: list):
        pass


if __name__ == '__main__':
    m = Wrapper()
    m.run()
