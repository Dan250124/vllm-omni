"""Benchmark client for VoxCPM2 TTS via /v1/audio/speech endpoint.

Reports RTF (Real-Time Factor) and latency statistics.
RTF = synthesis_time / audio_duration (lower is better, < 1.0 means realtime).

Examples:
    # 10 concurrent requests
    python bench.py --concurrency 10

    # 20 concurrent requests, 50 total
    python bench.py --concurrency 20 --total 50

    # With custom API base
    python bench.py --concurrency 10 --api-base http://192.168.1.100:8000
"""

from __future__ import annotations

import argparse
import asyncio
import struct
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_API_KEY = "sk-empty"

TEXTS = [
    "你好，这是VoxCPM2语音合成测试。",
    "人工智能正在深刻地改变着我们的生活方式。",
    "今天天气真不错，适合出去走走。",
    "请在下一个路口左转，然后直行三百米。",
    "知识就是力量，但热情才是推动力的源泉。",
    "会议已经改到下周四下午两点举行。",
    "图书馆位于主楼的第三层，欢迎随时前来借阅。",
    "恭喜你以优异的成绩通过了这次考试。",
    "请记得多喝水，保证充足的睡眠。",
    "列车将于下午三点半从七号站台出发。",
    "在一个遥远的星系中，曾经存在着一颗和平的星球。",
    "每天清晨，她都会在海边捡拾贝壳。",
    "开发团队昨晚成功完成了系统升级部署。",
    "音乐能够表达那些无法用言语描述的情感。",
    "请系好安全带，飞机即将起飞。",
    "时间如白驹过隙，转眼间又是一年。",
    "每一次伟大的成就，都曾被认为是不可能的。",
    "春天的风吹过田野，带来了花朵的芬芳。",
    "科技的发展让我们的生活变得更加便捷和高效。",
    "无论遇到什么困难，都要保持乐观向上的心态。",
    "没关系，时间会让你改变的。",
    "那我们就走着瞧。",
    "别再任性了，乖乖听话，不然我不介意让别人看到你狼狈的样子。",
    "没有不开心呀，有老婆在身边，我开心得很呢。是不是老婆觉得我哪里表现得不好？",
    "以后不许去，听到没？",
    "啊、啊、啊、啊、啊……好了吗？",
    "俩？除了小爷，还有谁？",
    "什么大哥？是你欠我钱！警察同志，你们可算是来了。",
    "……啊……啊……啊……啊……啊……啊~这样好累，人家不想动了。",
    "啊……啊……啊……啊……啊……啊~",
    "真的吗？你唱的什么歌呀，好不好听？",
    "嗯，那吃完饭去看看有什么新片。",
]


def parse_wav_duration(data: bytes) -> float | None:
    """Parse duration from WAV PCM data (bytes) by reading RIFF header."""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    # fmt chunk: sample_rate at offset 24 (4 bytes, little-endian)
    sample_rate = struct.unpack_from("<I", data, 24)[0]
    if sample_rate == 0:
        return None
    # data chunk size at offset 40
    data_size = struct.unpack_from("<I", data, 40)[0]
    # bytes per sample: bits_per_sample / 8, at offset 34
    bits_per_sample = struct.unpack_from("<H", data, 34)[0]
    bytes_per_sample = max(bits_per_sample // 8, 1)
    num_channels = struct.unpack_from("<H", data, 22)[0]
    num_samples = data_size // (num_channels * bytes_per_sample)
    return num_samples / sample_rate


def percentile(sorted_list: list[float], p: float) -> float:
    n = len(sorted_list)
    if n == 0:
        return 0.0
    idx = min(int(n * p), n - 1)
    return sorted_list[idx]


@dataclass
class Result:
    success: bool
    latency: float
    audio_duration: float = 0.0
    rtf: float = 0.0
    audio_bytes: int = 0
    status_code: int = 0
    error: str = ""


@dataclass
class BenchStats:
    total: int = 0
    success: int = 0
    failed: int = 0
    latencies: list[float] = field(default_factory=list)
    rtfs: list[float] = field(default_factory=list)
    audio_durations: list[float] = field(default_factory=list)
    audio_bytes_total: int = 0

    def add(self, r: Result) -> None:
        self.total += 1
        if r.success:
            self.success += 1
            self.latencies.append(r.latency)
            if r.rtf > 0:
                self.rtfs.append(r.rtf)
            if r.audio_duration > 0:
                self.audio_durations.append(r.audio_duration)
            self.audio_bytes_total += r.audio_bytes
        else:
            self.failed += 1

    def _print_percentiles(self, label: str, values: list[float], fmt: str = "{:.3f}") -> None:
        if not values:
            return
        s = sorted(values)
        print(f"\n  {label}:")
        print(f"    Min:              {fmt.format(s[0])}")
        print(f"    Mean:             {fmt.format(statistics.mean(s))}")
        print(f"    Median (P50):     {fmt.format(percentile(s, 0.50))}")
        print(f"    P90:              {fmt.format(percentile(s, 0.90))}")
        print(f"    P95:              {fmt.format(percentile(s, 0.95))}")
        print(f"    P99:              {fmt.format(percentile(s, 0.99))}")
        print(f"    Max:              {fmt.format(s[-1])}")

    def report(self) -> None:
        print("\n" + "=" * 55)
        print("  VoxCPM2 TTS Benchmark Results")
        print("=" * 55)
        print(f"  Total requests:    {self.total}")
        print(f"  Successful:        {self.success}")
        print(f"  Failed:            {self.failed}")
        if self.total:
            print(f"  Success rate:      {self.success / self.total * 100:.1f}%")

        self._print_percentiles("RTF (Real-Time Factor, lower=better)", self.rtfs)

        self._print_percentiles("Latency (seconds)", self.latencies)

        if self.audio_durations:
            s = sorted(self.audio_durations)
            print(f"\n  Audio duration (seconds):")
            print(f"    Min:              {s[0]:.3f}")
            print(f"    Mean:             {statistics.mean(s):.3f}")
            print(f"    Max:              {s[-1]:.3f}")
            print(f"    Total:            {sum(self.audio_durations):.2f}")

        if self.audio_bytes_total:
            print(f"\n  Total audio bytes:  {self.audio_bytes_total:,}")

        print("=" * 55)


VOICES = ["alice", "mix_voice_60"]


def build_payload(text: str, model: str, response_format: str) -> dict:
    return {
        "model": model,
        "input": text,
        "voice": random.choice(VOICES),
        "response_format": response_format,
    }


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    api_key: str,
    payload: dict,
    request_id: int,
    text: str,
) -> Result:
    t0 = time.monotonic()
    try:
        resp = await asyncio.wait_for(
            client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
            ),
            timeout=4.0,
        )
        latency = time.monotonic() - t0
        if resp.status_code == 200:
            audio_data = resp.content
            audio_bytes = len(audio_data)
            audio_dur = parse_wav_duration(audio_data) or 0.0
            rtf = latency / audio_dur if audio_dur > 0 else 0.0
            print(f"  [#{request_id}] OK  {latency:.3f}s  "
                  f"audio: {audio_dur:.2f}s  RTF: {rtf:.3f}  "
                  f"text: {text[:40]}")
            return Result(success=True, latency=latency,
                          audio_duration=audio_dur, rtf=rtf,
                          audio_bytes=audio_bytes,
                          status_code=resp.status_code)
        else:
            print(f"  [#{request_id}] ERR {resp.status_code} {latency:.3f}s  "
                  f"{resp.text[:120]}")
            return Result(success=False, latency=latency,
                          status_code=resp.status_code, error=resp.text[:200])
    except asyncio.TimeoutError:
        latency = time.monotonic() - t0
        print(f"  [#{request_id}] TIMEOUT {latency:.3f}s  (exceeded 4s)  text: {text[:40]}")
        return Result(success=False, latency=latency, error="Timeout after 4s")
    except Exception as e:
        latency = time.monotonic() - t0
        print(f"  [#{request_id}] EXC {latency:.3f}s  {e}")
        return Result(success=False, latency=latency, error=str(e))


async def run_bench(
    concurrency: int,
    total: int,
    api_base: str,
    api_key: str,
    model: str,
    response_format: str,
) -> None:
    url = f"{api_base}/v1/audio/speech"
    texts = TEXTS[:]

    sem = asyncio.Semaphore(concurrency)
    stats = BenchStats()
    request_id = 0
    wall_start = time.monotonic()

    async with httpx.AsyncClient(timeout=300) as client:
        async def worker() -> None:
            nonlocal request_id
            while True:
                async with sem:
                    rid = request_id
                    request_id += 1
                    if rid >= total:
                        return
                    text = random.choice(texts)
                    payload = build_payload(text, model, response_format)
                    r = await send_request(client, url, api_key, payload, rid, text)
                    stats.add(r)

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*tasks)

    wall_time = time.monotonic() - wall_start
    stats.report()
    print(f"\n  Wall time:             {wall_time:.2f}s")
    if wall_time > 0 and stats.total:
        print(f"  Throughput:            {stats.total / wall_time:.2f} req/s")
    if wall_time > 0 and stats.success:
        print(f"  Success throughput:    {stats.success / wall_time:.2f} req/s")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxCPM2 TTS benchmark")
    parser.add_argument(
        "--concurrency", "-c", type=int, default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--total", "-n", type=int, default=None,
        help="Total number of requests (default: 5 * concurrency)",
    )
    parser.add_argument("--model", type=str, default="OpenBMB/VoxCPM2")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY)
    parser.add_argument("--response-format", type=str, default="wav")
    args = parser.parse_args()

    total = args.total if args.total else 5 * args.concurrency

    print(f"VoxCPM2 TTS Benchmark")
    print(f"  Target:       {args.api_base}/v1/audio/speech")
    print(f"  Model:        {args.model}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Total reqs:   {total}")
    print(f"  Texts pool:   {len(TEXTS)}")
    print(f"  Voices:       {VOICES}")
    print()

    asyncio.run(run_bench(
        concurrency=args.concurrency,
        total=total,
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        response_format=args.response_format,
    ))


if __name__ == "__main__":
    main()