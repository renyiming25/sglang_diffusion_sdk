FROM artifacts.iflytek.com/docker-private/maas/inference:22.04-cuda128-dev-py311

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    LD_LIBRARY_PATH=/home/aiges/library:/home/aiges \
    UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple

# 安装 aiges
RUN pip install aiges==0.16.1 \
    --index-url https://repo-maas.cn-huabei-1.xf-yun.com/api/packages/administrator/pypi/simple \
    --extra-index-url https://pypi.mirrors.ustc.edu.cn/simple/

# 添加 aiges 的配置
COPY ./docker/log.py /usr/local/lib/python3.11/site-packages/aiges/utils/
COPY ./docker/aiservice_3.1.0.bin/ /home/aiges/

# 安装本地 sglang（可选你是否需要 --no-deps 或开发模式） 安装 nc
RUN apt-get update && apt-get install -y --no-install-recommends netcat curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv poetry boto3 \
    && uv pip install "sglang[diffusion]" --prerelease=allow --system

COPY . /build/sglang_diffusion_sdk
RUN cd /build/sglang_diffusion_sdk && make build

WORKDIR /home/aiges
# 清除缓存
RUN rm -rf ~/.cache/pip
