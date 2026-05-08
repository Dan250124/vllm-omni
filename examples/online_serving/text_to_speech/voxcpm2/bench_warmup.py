"""Benchmark script for VoxCPM2 first-request latency (warmup optimization).

Measures first-request latency across multiple server restarts.  The warmup
optimization shifts torch.compile + CUDA Graph capture from the first user
request to server startup, so the first request should be ~2s instead of ~28s.

Usage:
    # Auto mode: script manages server lifecycle, restarts between rounds
    python bench_warmup.py --rounds 3

    # Manual mode: use your own running server (only measures 1 round)
    python bench_warmup.py --api-base http://localhost:8000

    # Custom server command
    python bench_warmup.py --rounds 3 --server-cmd "vllm-omni serve openbmb/VoxCPM2 --omni --port 8000"

Output example:
    Hardware: NVIDIA GeForce RTX 5090 D x1, CUDA 13.1
    Round 1/3:  2.01s  (1,048,576 bytes)
    Round 2/3:  1.95s  (1,048,576 bytes)
    Round 3/3:  2.03s  (1,048,576 bytes)
    First-request (p50):  2.01s | (p95):  2.03s
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time

import httpx

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "sk-empty"
DEFAULT_TEXT = "Hello, this is a warmup benchmark for VoxCPM2."
DEFAULT_SERVER_CMD = "vllm-omni serve openbmb/VoxCPM2 --omni --host 0.0.0.0 --port 8000"
SERVER_READY_TIMEOUT = 600  # 10 minutes max wait for server startup


def get_hardware_info() -> str:
    """Collect GPU model, count, and CUDA version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpus = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
        gpu_str = f"{gpus[0]} x{len(gpus)}" if gpus else "unknown"

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        driver = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else "unknown"

        # Use nvidia-smi CUDA version (max CUDA runtime supported by the driver)
        # rather than nvcc (CUDA Toolkit compiler version), which may differ.
        cuda = "unknown"
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if "CUDA Version" in line:
                cuda = line.split("CUDA Version:")[-1].strip()
                break
        return f"{gpu_str}, Driver {driver}, CUDA {cuda}"
    except Exception:
        return "hardware info unavailable"


def wait_for_server(api_base: str, timeout: int = SERVER_READY_TIMEOUT) -> bool:
    """Poll /health until the server is ready or timeout."""
    url = f"{api_base}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(url)
            if resp.status_code == 200:
                return True
        except (httpx.ConnectError, httpx.ConnectTimeout):
            pass
        time.sleep(2)
    return False


def run_request(api_base: str, text: str, api_key: str) -> tuple[float, int]:
    """Send a single TTS request and return (latency_seconds, response_bytes)."""
    payload = {
        "model": "voxcpm2",
        "input": text,
        "voice": "default",
        "response_format": "wav",
    }
    url = f"{api_base}/v1/audio/speech"
    t0 = time.monotonic()
    with httpx.Client(timeout=300) as client:
        resp = client.post(url, json=payload, headers={"Authorization": f"Bearer {api_key}"})
    elapsed = time.monotonic() - t0
    if resp.status_code != 200:
        print(f"  Error {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return elapsed, 0
    return elapsed, len(resp.content)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    return sorted(data)[int(len(data) * p / 100)] if len(data) > 1 else data[0]


def run_auto_bench(args: argparse.Namespace) -> None:
    """Auto mode: manage server lifecycle, restart between rounds."""
    server_proc: subprocess.Popen | None = None
    latencies: list[float] = []

    def kill_server() -> None:
        nonlocal server_proc
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            server_proc = None
            time.sleep(3)  # wait for port release

    try:
        for i in range(1, args.rounds + 1):
            if i > 1:
                print("  Restarting server...")
                kill_server()

            # Start server
            server_proc = subprocess.Popen(
                args.server_cmd,
                shell=True,
                stdout=sys.stdout,
                stderr=sys.stderr,
                preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
            )

            print("  Waiting for server ready...", end="", flush=True)
            if not wait_for_server(args.api_base):
                print(f"\n  ERROR: Server did not become ready within {SERVER_READY_TIMEOUT}s", file=sys.stderr)
                kill_server()
                return
            print(" done")

            # First request
            latency, nbytes = run_request(args.api_base, args.text, args.api_key)
            latencies.append(latency)
            tag = f"Round {i}/{args.rounds}: {latency:6.2f}s"
            if nbytes:
                tag += f"  ({nbytes:,} bytes)"
            print(tag)

            kill_server()

    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        kill_server()

    if latencies:
        print()
        print(f"First-request (p50): {percentile(latencies, 50):6.2f}s | (p95): {percentile(latencies, 95):6.2f}s")


def run_manual_bench(args: argparse.Namespace) -> None:
    """Manual mode: measure one first request against an already running server."""
    print("Measuring first request against running server...")
    latency, nbytes = run_request(args.api_base, args.text, args.api_key)
    tag = f"First request: {latency:6.2f}s"
    if nbytes:
        tag += f"  ({nbytes:,} bytes)"
    print(tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM2 warmup latency benchmark")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT)
    parser.add_argument("--rounds", type=int, default=3, help="Number of server restart cycles (auto mode)")
    parser.add_argument("--server-cmd", type=str, default=DEFAULT_SERVER_CMD, help="Command to start the server")
    parser.add_argument("--manual", action="store_true", help="Manual mode: use an already running server")
    args = parser.parse_args()

    hw = get_hardware_info()
    print(f"Hardware: {hw}")

    if args.manual:
        print("Mode: manual")
        print()
        run_manual_bench(args)
    else:
        print(f"Mode: auto (rounds={args.rounds})")
        print(f"Server cmd: {args.server_cmd}")
        print()
        run_auto_bench(args)


if __name__ == "__main__":
    main()
