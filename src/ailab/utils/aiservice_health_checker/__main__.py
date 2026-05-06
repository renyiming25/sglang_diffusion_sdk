#!/usr/bin/env python
# coding:utf-8
import re
import subprocess
import sys

EXIT_SUCCESS = 0
EXIT_PORT_FILE_NOT_FOUND = 3   # 端口文件不存在
EXIT_PORT_FORMAT_INVALID = 4   # 端口号格式非法
EXIT_MAAS_PORT_FAIL = 5        # MAAS 端口不可用
EXIT_UNKNOWN_ERROR = 255       # 未知异常

MAAS_PORT_PATH = "/home/aiges/maas_port"


def check_port_alive(ip: str, port: int) -> bool:
    """检查指定 ip:port 是否可连通，返回 True/False。"""
    result = subprocess.run(
        ["nc", "-zv", ip, str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def main() -> None:
    try:
        ip = "127.0.0.1"

        # 读取端口文件
        try:
            with open(MAAS_PORT_PATH, "r") as f:
                port_str = f.read().strip()
        except FileNotFoundError:
            print(f"Port file not found: {MAAS_PORT_PATH}")
            sys.exit(EXIT_PORT_FILE_NOT_FOUND)

        # 校验端口号格式
        if not re.fullmatch(r"[1-9]\d{0,4}", port_str):
            print(f"Invalid port format in {MAAS_PORT_PATH}: '{port_str}'")
            sys.exit(EXIT_PORT_FORMAT_INVALID)
        port = int(port_str)
        if not (1 <= port <= 65535):
            print(f"Port out of range (1-65535): {port}")
            sys.exit(EXIT_PORT_FORMAT_INVALID)

        # 检查端口
        print(f"Checking MaaS service address: {ip}:{port}")
        if not check_port_alive(ip, port):
            print(f"MaaS port {port} is not reachable.")
            sys.exit(EXIT_MAAS_PORT_FAIL)

        print(f"MaaS port {port} is alive.")
        sys.exit(EXIT_SUCCESS)

    except SystemExit:
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(EXIT_UNKNOWN_ERROR)


if __name__ == "__main__":
    main()