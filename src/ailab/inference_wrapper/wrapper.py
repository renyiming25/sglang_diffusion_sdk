import enum
import os

try:
    from aiges_embed import Response, DataListCls, SessionCreateResponse  # c++
except:
    from aiges.dto import Response, DataListCls, SessionCreateResponse

from aiges.utils.log import log


class WTaskType(enum.Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    AUDIO_GENERATION = "audio_generation"
    THIRD_VIDEO_GENERATION = "third_video_generation"


def GetTaskType(model: str, model_task_type:str, backend="sglang") -> WTaskType:
    if model_task_type == "image_generation":
        return WTaskType.IMAGE_GENERATION
    elif model_task_type == "video_generation":
        return WTaskType.VIDEO_GENERATION
    elif model_task_type == "audio_generation":
        return WTaskType.AUDIO_GENERATION
    elif model_task_type == "third_video_generation":
        return WTaskType.THIRD_VIDEO_GENERATION
    else:
        raise ValueError(f"unsupported model_task_type {model_task_type}, model {model}")


class Wrapper:
    def __init__(self, *args, **kwargs):
        pretrained_model = os.environ.get("PRETRAINED_MODEL_NAME")
        log.info(f"Wrapper init pretrained_model:{pretrained_model}")
        # if pretrained_model is None:
        #     raise ValueError("pretrained_model env not set")

        model_task_type = os.environ.get("MODEL_TASK_TYPE", "image_generation")
        backend = os.environ.get("MAAS_BACKEND","sglang")

        task = GetTaskType(pretrained_model, model_task_type, backend)
        log.info(f"Wrapper init task: {model_task_type}, pretrained_model: {pretrained_model}")

        if task == WTaskType.IMAGE_GENERATION:
            from ailab.inference_wrapper.diffusers.image_diffusion.wrapper import Wrapper as OlmWrapper
        elif task == WTaskType.VIDEO_GENERATION:
            from ailab.inference_wrapper.diffusers.video_diffusion.wrapper import Wrapper as OlmWrapper
        elif task == WTaskType.AUDIO_GENERATION:
            from ailab.inference_wrapper.diffusers.audio_diffusion.wrapper import Wrapper as OlmWrapper
        elif task == WTaskType.THIRD_VIDEO_GENERATION:
            from ailab.inference_wrapper.diffusers.third_diffusion.third_video_generation.wrapper import Wrapper as OlmWrapper
        self._wrapper = OlmWrapper(*args, **kwargs)

    def wrapperInit(self, config: {}) -> int:
        return self._wrapper.wrapperInit(config)

    def wrapperLoadRes(self, reqData: DataListCls, patch_id: int) -> int:
        return self._wrapper.wrapperLoadRes(reqData, patch_id)

    def wrapperUnloadRes(self, patch_id: int) -> int:
        return self._wrapper.wrapperUnloadRes(patch_id)

    def wrapperOnceExec(self, params: {}, reqData: DataListCls, usrTag: str = "", persId: int = 0) -> Response:
        return self._wrapper.wrapperOnceExec(params, reqData, usrTag, persId)

    def wrapperOnceExecAsync(self, params: {}, reqData: DataListCls, usrTag: str, persId: int = 0) -> int:
        return self._wrapper.wrapperOnceExecAsync(params, reqData, usrTag, persId)

    def wrapperFini(self) -> int:
        return self._wrapper.wrapperFini()

    def wrapperError(self, ret: int) -> str:
        return self._wrapper.wrapperError(ret)

    # def wrapperWrite(self, handle: str, req: DataListCls sid: str) -> int:
    def wrapperWrite(self, handle: str, req: DataListCls) -> int:
        return self._wrapper.wrapperWrite(handle, req)

    def wrapperCreate(self, params: {}, sid: str, persId: int = 0, usrTag: str = "") -> SessionCreateResponse:
        return self._wrapper.wrapperCreate(params, sid, persId, usrTag)

    def wrapperDestroy(self, handle: str) -> int:
        return self._wrapper.wrapperDestroy(handle)

    def wrapperRead(self, handle: str):
        return self._wrapper.wrapperRead(handle)

    def wrapperTestFunc(self, data: [], respData: []):
        return self._wrapper.wrapperTestFunc(data, respData)
