"""Monitor physical memory (RSS) and virtual memory (VSZ) of processes whose command line contains 'vllm'.

Samples every 60 seconds and appends results to vllm_mem.log.
Press Ctrl-C to stop.
"""

import subprocess
import time
from datetime import datetime

LOG_FILE = "vllm_mem.log"


def get_vllm_processes() -> list[tuple[str, int, float, float]]:
    """Return list of (name, pid, rss_mb, vsz_mb) for processes matching 'vllm'."""
    # pid, vsz (KB), rss (KB), full command
    result = subprocess.run(
        ["ps", "-eo", "pid,vsz,rss,args"],
        capture_output=True,
        text=True,
    )
    entries: list[tuple[str, int, float, float]] = []
    for line in result.stdout.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, vsz, rss, cmd = parts
        if "vllm" not in cmd.lower():
            continue
        entries.append((cmd.split()[0] if cmd else "unknown", int(pid), int(rss) / 1024, int(vsz) / 1024))
    return entries


def main() -> None:
    print(f"Monitoring vllm process RSS & VSZ, logging to {LOG_FILE} (Ctrl-C to stop)")
    with open(LOG_FILE, "a") as f:
        while True:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            procs = get_vllm_processes()
            if not procs:
                f.write(f"[{now}] No vllm processes found\n")
            else:
                total_rss = sum(rss for _, _, rss, _ in procs)
                total_vsz = sum(vsz for _, _, _, vsz in procs)
                f.write(f"[{now}] {'PID':>8}  {'VSZ (MB)':>10}  {'RSS (MB)':>10}  Command\n")
                f.write("-" * 75 + "\n")
                for name, pid, rss, vsz in procs:
                    f.write(f"  {pid:>6}  {vsz:>10.1f}  {rss:>10.1f}  {name}\n")
                f.write(f"  {'TOTAL':>6}  {total_vsz:>10.1f}  {total_rss:>10.1f}\n\n")
            f.flush()
            total_rss = sum(rss for _, _, rss, _ in procs)
            total_vsz = sum(vsz for _, _, _, vsz in procs)
            print(f"[{now}] sampled, total RSS: {total_rss:.1f} MB, total VSZ: {total_vsz:.1f} MB")
            time.sleep(60)


if __name__ == "__main__":
    main()
