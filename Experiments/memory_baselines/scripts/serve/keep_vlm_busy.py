#!/usr/bin/env python3
"""Discover this user's local VLM endpoints and keep a selected one busy.

Examples:
  bash keep_vlm_busy.sh --list
  bash keep_vlm_busy.sh --port 8170 --target-gpu-util 60
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import random
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Endpoint:
    port: int
    address: str
    pids: tuple[int, ...]
    models: tuple[str, ...]
    listening: bool = True


def is_vlm_server(command: str) -> bool:
    value = command.lower()
    known_servers = (
        "vllm serve",
        "vllm.entrypoints.openai.api_server",
        "sglang.launch_server",
        "lmdeploy serve",
        "ollama serve",
        "aphrodite run",
        "tritonserver",
    )
    if any(marker in value for marker in known_servers):
        return True
    has_model = any(name in value for name in ("qwen", "llava", "internvl", "minicpm-v", "vision-language"))
    has_api_server = any(name in value for name in ("uvicorn", "fastapi", "api_server"))
    return has_model and has_api_server


def model_from_command(command: str) -> str:
    match = re.search(r"--served-model-name(?:=|\s+)([^\s]+)", command)
    if match:
        return match.group(1).strip("'\"")
    match = re.search(r"\bvllm\s+serve\s+([^\s]+)", command)
    if match:
        return match.group(1).strip("'\"").rstrip("/").rsplit("/", 1)[-1]
    match = re.search(r"(?:--model(?:-path)?)(?:=|\s+)([^\s]+)", command)
    if match:
        return match.group(1).strip("'\"").rstrip("/").rsplit("/", 1)[-1]
    return "VLM API"


def discover_endpoints() -> list[Endpoint]:
    if not shutil_which("ss"):
        raise RuntimeError("找不到 ss 命令；请安装 iproute2。")
    uid = os.getuid()
    process_result = subprocess.run(
        ["ps", "-eo", "pid=,uid=,comm=,args="], text=True, capture_output=True, check=True
    )
    processes: dict[int, str] = {}
    for row in process_result.stdout.splitlines():
        parts = row.strip().split(None, 3)
        if len(parts) != 4:
            continue
        try:
            pid, process_uid = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if process_uid == uid and parts[2] not in {"bash", "sh", "dash", "zsh", "ps", "rg", "grep", "timeout"} and is_vlm_server(parts[3]):
            processes[pid] = parts[3]

    socket_result = subprocess.run(
        ["ss", "-H", "-ltnp"], text=True, capture_output=True, check=True
    )
    found: dict[tuple[int, str], dict[int, str]] = {}
    for row in socket_result.stdout.splitlines():
        fields = row.split()
        if len(fields) < 5 or fields[0] != "LISTEN":
            continue
        local = fields[3]
        try:
            address, port_text = local.rsplit(":", 1)
            port = int(port_text)
        except ValueError:
            continue
        pids = {int(value) for value in re.findall(r"pid=(\d+)", row)}
        matches = pids.intersection(processes)
        if not matches:
            continue
        key = (port, address.strip("[]"))
        bucket = found.setdefault(key, {})
        for pid in matches:
            bucket[pid] = model_from_command(processes[pid])

    endpoints = [
        Endpoint(port, address, tuple(sorted(items)), tuple(sorted(set(items.values()))))
        for (port, address), items in found.items()
    ]
    listening_ports = {item.port for item in endpoints}
    for pid, command in processes.items():
        port_match = re.search(r"--port(?:=|\s+)(\d+)", command)
        if not port_match:
            continue
        port = int(port_match.group(1))
        if port in listening_ports:
            continue
        host_match = re.search(r"--host(?:=|\s+)([^\s]+)", command)
        address = host_match.group(1).strip("'\"") if host_match else "127.0.0.1"
        endpoints.append(Endpoint(port, address, (pid,), (model_from_command(command),), False))
    return sorted(endpoints, key=lambda item: (item.port, item.address))


def shutil_which(command: str) -> Optional[str]:
    from shutil import which

    return which(command)


def print_endpoints(endpoints: list[Endpoint]) -> None:
    if not endpoints:
        print("当前容器中没有发现由当前 UID 启动的 VLM 监听端口。")
        return
    print("当前容器中检测到的 VLM 端口（仅当前 UID）：")
    print(f"{'PORT':>6}  {'ADDRESS':<24} {'PID':<16} {'STATUS':<12} MODEL")
    for item in endpoints:
        print(
            f"{item.port:>6}  {item.address:<24} "
            f"{','.join(map(str, item.pids)):<16} "
            f"{'LISTENING' if item.listening else 'STARTING':<12} {', '.join(item.models)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="发现本容器当前用户运行的 VLM 服务，并持续发送图文请求。"
    )
    parser.add_argument("--port", type=int, help="目标 VLM 服务端口；不指定时交互选择")
    parser.add_argument("--host", default="127.0.0.1", help="API 地址主机，默认 127.0.0.1")
    parser.add_argument("--model", help="服务模型 ID；默认从 /v1/models 自动获取")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"), help="API key；默认读取 OPENAI_API_KEY")
    parser.add_argument("--concurrency", type=int, default=1, help="初始并发请求数，默认 1")
    parser.add_argument("--max-concurrency", type=int, default=8, help="利用率控制允许的最大并发数，默认 8")
    parser.add_argument("--interval", type=float, default=0.2, help="每个 worker 请求完成后的等待秒数，默认 0.2")
    parser.add_argument("--duration", type=float, default=0, help="运行秒数；0 表示持续运行，按 Ctrl-C 停止")
    parser.add_argument("--max-tokens", type=int, default=256, help="每次生成的最大 token 数，默认 256")
    parser.add_argument("--timeout", type=float, default=180, help="单次 API 请求超时秒数")
    parser.add_argument("--image", help="可选：使用指定本地图像；默认生成变化的测试图")
    parser.add_argument("--target-gpu-util", type=int, help="可选：目标 GPU 利用率百分比；自动识别目标服务使用的 GPU 并自适应并发")
    parser.add_argument("--gpu-index", help="可选：手动指定 GPU 索引，可逗号分隔；默认从目标服务进程自动识别")
    parser.add_argument("--gpu-tolerance", type=int, default=8, help="利用率目标容差，默认 ±8%%")
    parser.add_argument("--min-interval", type=float, default=0.05, help="自适应控制最小请求间隔")
    parser.add_argument("--max-interval", type=float, default=5.0, help="自适应控制最大请求间隔")
    parser.add_argument("--list", action="store_true", help="只列出发现的端口，不发送请求")
    return parser.parse_args()


def select_port(args: argparse.Namespace, endpoints: list[Endpoint]) -> int:
    listening = [item for item in endpoints if item.listening]
    if args.port is not None:
        if not 1 <= args.port <= 65535:
            raise ValueError("端口必须在 1 到 65535 之间。")
        if not any(item.port == args.port for item in listening):
            pending = any(item.port == args.port for item in endpoints)
            if pending:
                raise ValueError(f"VLM 进程已配置端口 {args.port}，但端口还未开始监听；等待服务启动后重试。")
            available = ", ".join(str(item.port) for item in listening) or "无"
            raise ValueError(
                f"端口 {args.port} 不在当前 UID 的 VLM 监听列表中；发现的端口：{available}。"
            )
        return args.port
    if not listening:
        raise ValueError("没有可选择的 VLM 端口。")
    if len(listening) == 1:
        print(f"自动选择唯一监听端口：{listening[0].port}")
        return listening[0].port
    if not sys.stdin.isatty():
        raise ValueError("发现多个服务；请用 --port 指定目标端口。")
    choices = sorted({item.port for item in listening})
    while True:
        value = input(f"选择目标端口 {choices}: ").strip()
        try:
            port = int(value)
        except ValueError:
            continue
        if port in choices:
            return port


def make_test_image(seed: int) -> str:
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    image = Image.new("RGB", (512, 512), (245, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, 494, 78), fill=(28, 54, 92))
    draw.text((34, 37), f"VLM LOAD TEST {seed:06d}", fill="white")
    for index in range(20):
        x = 24 + (index % 5) * 94
        y = 104 + (index // 5) * 94
        color = tuple(rng.randint(35, 225) for _ in range(3))
        draw.rounded_rectangle((x, y, x + 76, y + 68), radius=8, fill=color)
        draw.text((x + 8, y + 25), f"{index + 1:02d}", fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def read_image(path: str) -> str:
    from PIL import Image

    buffer = io.BytesIO()
    with Image.open(path) as image:
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def process_gpu_visibility(pids: tuple[int, ...]) -> Optional[str]:
    """Read GPU selection from the VLM process or its launcher parent."""
    for initial_pid in pids:
        pid = initial_pid
        visited: set[int] = set()
        for _ in range(12):
            if pid <= 1 or pid in visited:
                break
            visited.add(pid)
            proc = f"/proc/{pid}"
            try:
                entries = open(f"{proc}/environ", "rb").read().split(b"\0")
                values = {}
                for entry in entries:
                    if b"=" not in entry:
                        continue
                    key, value = entry.split(b"=", 1)
                    if key in (b"CUDA_VISIBLE_DEVICES", b"GPU_INDEX"):
                        values[key.decode()] = value.decode(errors="replace").strip()
                visible = values.get("CUDA_VISIBLE_DEVICES")
                if visible and visible.lower() not in {"all", "-1"}:
                    return visible
                gpu_index = values.get("GPU_INDEX")
                if gpu_index and gpu_index.lower() not in {"all", "-1"}:
                    return gpu_index
                status = open(f"{proc}/status", encoding="utf-8").read()
                ppid_line = next(line for line in status.splitlines() if line.startswith("PPid:"))
                pid = int(ppid_line.split()[1])
            except (OSError, StopIteration, ValueError):
                break
    return None


def resolve_gpu_indices(endpoint: Endpoint, override: Optional[str]) -> tuple[int, ...]:
    if override:
        values = override.split(",")
    else:
        visible = process_gpu_visibility(endpoint.pids)
        if not visible:
            raise RuntimeError(
                "无法从目标 VLM 服务进程读取 GPU 配置；请通过 --gpu-index 手动指定。"
            )
        values = visible.split(",")

    try:
        rows = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"无法查询 nvidia-smi GPU 列表：{exc}") from exc
    uuid_to_index = {}
    valid_indices = set()
    for row in rows:
        columns = [value.strip() for value in row.split(",", 1)]
        if len(columns) == 2 and columns[0].isdigit():
            valid_indices.add(int(columns[0]))
            uuid_to_index[columns[1].lower()] = int(columns[0])

    indices = []
    for value in values:
        value = value.strip()
        if value.isdigit():
            index = int(value)
        elif value.lower() in uuid_to_index:
            index = uuid_to_index[value.lower()]
        else:
            raise RuntimeError(f"无法将 GPU 标识 {value!r} 映射到 nvidia-smi 索引；请传入 --gpu-index。")
        if index not in valid_indices:
            raise RuntimeError(f"GPU 索引 {index} 不存在；请传入有效的 --gpu-index。")
        if index not in indices:
            indices.append(index)
    if not indices:
        raise RuntimeError("没有识别到 GPU；如服务运行在 CPU 上，请不要启用 GPU 利用率目标控制。")
    return tuple(indices)


def gpu_utilization(gpu_indices: tuple[int, ...]) -> tuple[float, list[int]]:
    values = []
    for index in gpu_indices:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        values.append(int(result.stdout.strip().splitlines()[0]))
    return sum(values) / len(values), values


def main() -> int:
    args = parse_args()
    if args.concurrency < 1 or args.max_concurrency < args.concurrency:
        print("并发需满足 1 <= concurrency <= max-concurrency。", file=sys.stderr)
        return 2
    if args.interval < 0 or args.duration < 0 or args.max_tokens < 1:
        print("interval、duration 不能小于 0，max-tokens 必须大于 0。", file=sys.stderr)
        return 2
    if args.min_interval < 0 or args.max_interval < args.min_interval:
        print("请求间隔范围无效，须满足 0 <= min-interval <= max-interval。", file=sys.stderr)
        return 2
    if args.target_gpu_util is not None:
        if not 1 <= args.target_gpu_util <= 100:
            print("目标 GPU 利用率须为 1–100。", file=sys.stderr)
            return 2

    try:
        endpoints = discover_endpoints()
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        print(f"端口发现失败：{exc}", file=sys.stderr)
        return 2
    print_endpoints(endpoints)
    if args.list:
        return 0
    try:
        port = select_port(args, endpoints)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    gpu_indices = None
    if args.target_gpu_util is not None:
        selected_endpoint = next(item for item in endpoints if item.port == port and item.listening)
        try:
            gpu_indices = resolve_gpu_indices(selected_endpoint, args.gpu_index)
            initial_util, initial_values = gpu_utilization(gpu_indices)
        except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, IndexError) as exc:
            print(f"无法读取目标 VLM 使用的 GPU：{exc}", file=sys.stderr)
            return 2
        per_gpu = ", ".join(f"GPU {i}={v}%" for i, v in zip(gpu_indices, initial_values))
        print(f"GPU 自动识别：{per_gpu}，平均利用率 {initial_util:.0f}%")

    try:
        from openai import OpenAI
        from PIL import Image, ImageDraw  # noqa: F401
    except ImportError:
        print("缺少 openai 或 Pillow；请使用 vllm_env 环境运行此脚本。", file=sys.stderr)
        return 2

    base_url = f"http://{args.host}:{port}/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key, timeout=args.timeout)
    try:
        model = args.model or client.models.list().data[0].id
    except Exception as exc:
        print(f"无法连接 {base_url} 或读取模型列表：{exc}", file=sys.stderr)
        return 1

    try:
        static_image = read_image(args.image) if args.image else None
    except Exception as exc:
        print(f"读取图像失败：{exc}", file=sys.stderr)
        return 2

    print(
        f"开始调用 {base_url}，model={model}，初始并发={args.concurrency}，"
        f"max_tokens={args.max_tokens}。持续运行时按 Ctrl-C 停止。",
        flush=True,
    )
    stop = threading.Event()
    stats_lock = threading.Lock()
    control_lock = threading.Lock()
    stats = {"ok": 0, "errors": 0, "latency": 0.0, "started": time.monotonic()}
    control = {"active": args.concurrency, "interval": args.interval}

    def stop_handler(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, stop_handler)

    def worker(worker_id: int) -> None:
        failures = 0
        while not stop.is_set():
            with control_lock:
                active = control["active"]
                delay = control["interval"]
            if worker_id >= active:
                stop.wait(0.2)
                continue

            started = time.monotonic()
            seed = int(started * 1_000_000) + worker_id
            image_url = static_image or make_test_image(seed)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"读取图片顶部的测试编号，并简要描述彩色方块布局。请求序号 {seed}。"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
            try:
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=0.2,
                )
                elapsed = time.monotonic() - started
                with stats_lock:
                    stats["ok"] += 1
                    stats["latency"] += elapsed
                failures = 0
            except Exception as exc:
                with stats_lock:
                    stats["errors"] += 1
                failures += 1
                print(f"worker {worker_id}: 请求失败 ({failures}/8): {exc}", flush=True)
                if failures >= 8:
                    stop.set()
                stop.wait(min(2**min(failures, 4), 15))
                continue
            stop.wait(delay)

    executor = ThreadPoolExecutor(max_workers=args.max_concurrency, thread_name_prefix="vlm-load")
    futures = [executor.submit(worker, worker_id) for worker_id in range(args.max_concurrency)]
    next_report = time.monotonic() + 5
    next_control = time.monotonic() + 5
    try:
        while not stop.is_set():
            now = time.monotonic()
            if args.duration and now - stats["started"] >= args.duration:
                break
            if now >= next_report:
                with stats_lock:
                    elapsed = max(now - stats["started"], 0.001)
                    ok, errors = stats["ok"], stats["errors"]
                    avg_latency = stats["latency"] / ok if ok else 0.0
                with control_lock:
                    active, delay = control["active"], control["interval"]
                gpu_note = ""
                if gpu_indices is not None:
                    try:
                        current_util, current_values = gpu_utilization(gpu_indices)
                        gpu_note = ", " + ", ".join(
                            f"GPU {i}={v}%" for i, v in zip(gpu_indices, current_values)
                        ) + f"，均值={current_util:.0f}%"
                    except Exception as exc:
                        print(f"GPU 利用率读取失败，停止负载：{exc}", file=sys.stderr)
                        stop.set()
                        break
                print(
                    f"运行 {elapsed:.0f}s | 成功 {ok} | 失败 {errors} | "
                    f"平均延迟 {avg_latency:.2f}s | "
                    f"并发 {active}/{args.max_concurrency} | 间隔 {delay:.2f}s{gpu_note}",
                    flush=True,
                )
                next_report = now + 5

            if gpu_indices is not None and now >= next_control:
                try:
                    current, _ = gpu_utilization(gpu_indices)
                except Exception as exc:
                    print(f"GPU 利用率读取失败，停止负载：{exc}", file=sys.stderr)
                    stop.set()
                    break
                with control_lock:
                    active, delay = control["active"], control["interval"]
                    low = args.target_gpu_util - args.gpu_tolerance
                    high = args.target_gpu_util + args.gpu_tolerance
                    if current < low:
                        control["active"] = min(args.max_concurrency, active + 1)
                        control["interval"] = max(args.min_interval, delay * 0.75)
                    elif current > high:
                        control["active"] = max(1, active - 1)
                        control["interval"] = min(args.max_interval, max(delay * 1.25, delay + 0.05))
                next_control = now + 5
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("收到 Ctrl-C，等待正在执行的请求结束…", flush=True)
    finally:
        stop.set()
        executor.shutdown(wait=True, cancel_futures=False)
        with stats_lock:
            elapsed = max(time.monotonic() - stats["started"], 0.001)
            print(
                f"结束：运行 {elapsed:.1f}s，成功 {stats['ok']}，失败 {stats['errors']}，"
                f"平均吞吐 {stats['ok'] / elapsed:.2f} 请求/秒。",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
