#!/usr/bin/env python
# coding:utf-8
import subprocess
import sys

EXIT_SUCCESS = 0          # 成功
EXIT_INVALID_FORMAT = 3   # .status 格式错误
EXIT_RPC_PORT_FAIL = 4    # RPC端口不可用
EXIT_MAAS_PORT_FAIL = 5   # MAAS service 端口不可用
EXIT_UNKNOWN_ERROR = 255  # 未知错误（异常）

def checkPortAlive() -> None:
    status_path = "/home/aiges/.status"
    maas_port_path = "/home/aiges/maas_port"

    try:
        with open(status_path, "r") as file:
            content = file.read().strip()
        with open(maas_port_path, "r") as file:
            maas_port = file.read().strip()

        if ":" not in content:
            print("Invalid format in .status file (expected 'ip:port').")
            sys.exit(EXIT_INVALID_FORMAT)

        ip, port = content.split(":", 1)
        print(f"Checking RPC address: {ip}:{port}")

        # 检查 RPC 端口
        rpc_cmd = ["nc", "-zv", ip, port]
        result = subprocess.run(rpc_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"RPC port {port} check failed:\n{result.stderr.decode().strip()}")
            sys.exit(EXIT_RPC_PORT_FAIL)
        print(f"RPC port {port} is alive.")

        # 检查 vLLM 端口
        print(f"Checking vLLM address: {ip}:{maas_port}")
        maas_cmd = ["nc", "-zv", ip, maas_port]
        result = subprocess.run(maas_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"maas port {maas_port} check failed:\n{result.stderr.decode().strip()}")
            sys.exit(EXIT_MAAS_PORT_FAIL)
        print(f"maas port {maas_port} is alive.")

        print("All ports are alive.")
        sys.exit(EXIT_SUCCESS)

    except Exception as e:
        print(f"nexpected error in checkPortAlive: {e}")
        sys.exit(EXIT_UNKNOWN_ERROR)

if __name__ == '__main__':
    checkPortAlive()