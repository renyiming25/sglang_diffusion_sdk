from aiges.sdk import WrapperBase

try:
    from aiges_embed import Response, DataListCls, SessionCreateResponse  # c++
except:
    from aiges.dto import Response, DataListCls, SessionCreateResponse

class Wrapper(WrapperBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from ailab.inference_wrapper.wrapper import Wrapper as OlmWrapper
        self._wrapper = OlmWrapper(*args, **kwargs)

    def wrapperInit(self, config: {}) -> int:
        return self._wrapper.wrapperInit(config)

    def wrapperLoadRes(self, reqData: DataListCls, patch_id: int) -> int:
        return self._wrapper.wrapperLoadRes(reqData, patch_id)
    
    def wrapperUnloadRes(self, patch_id: int) -> int:
        return self._wrapper.wrapperUnloadRes(patch_id)
    
    def wrapperOnceExec(self, params: {}, reqData: DataListCls, usrTag:str="",persId: int = 0) -> Response:
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

    def wrapperCreate(self, params: {}, sid: str, persId: int = 0, usrTag: str="") -> SessionCreateResponse:
        return self._wrapper.wrapperCreate(params, sid, persId, usrTag)

    def wrapperDestroy(self, handle: str) -> int:
        return self._wrapper.wrapperDestroy(handle)

    def wrapperRead(self, handle: str):
        return self._wrapper.wrapperRead(handle)

    def wrapperTestFunc(self, data: [], respData: []):
        return self._wrapper.wrapperTestFunc(data, respData)