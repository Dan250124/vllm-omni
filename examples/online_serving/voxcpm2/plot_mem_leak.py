"""Plot RSS memory growth for given PIDs from vllm_mem.log."""

import argparse
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CLI ---
parser = argparse.ArgumentParser(description="Plot RSS memory growth from vllm_mem.log")
parser.add_argument(
    "--pids", type=int, nargs="+", required=True,
    help="Target PIDs to plot (e.g., --pids 3629 4859)",
)
parser.add_argument(
    "--log-file",
    default=__file__.replace("plot_mem_leak.py", "vllm_mem.log"),
    help="Path to vllm_mem.log (default: vllm_mem.log in same dir)",
)
parser.add_argument(
    "--out",
    default=None,
    help="Output image path (default: mem_leak_plot.png in same dir)",
)
args = parser.parse_args()

LOG_FILE = args.log_file
TARGET_PIDS = set(args.pids)

# Color palette for up to 6 PIDs
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

# --- Parse log ---
timestamp_re = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
data_re = re.compile(r"^\s+(\d+)\s+[\d.]+\s+([\d.]+)\s+(.+)$")

pid_rss: dict[int, list[float]] = defaultdict(list)
pid_cmd: dict[int, str] = {}

with open(LOG_FILE) as f:
    current_ts = None
    for line in f:
        ts_match = timestamp_re.search(line)
        if ts_match:
            current_ts = ts_match.group(1)
            continue

        if current_ts is None:
            continue

        m = data_re.match(line)
        if not m:
            if line.strip().startswith("TOTAL") or line.strip() == "":
                current_ts = None
            continue

        pid = int(m.group(1))
        rss = float(m.group(2))
        cmd = m.group(3).strip()

        if pid in TARGET_PIDS:
            pid_rss[pid].append(rss)
            pid_cmd[pid] = cmd

# --- Validate ---
if not pid_rss:
    print(f"ERROR: None of the target PIDs {TARGET_PIDS} found in {LOG_FILE}")
    print("Available PIDs in log:")
    all_pids = set()
    with open(LOG_FILE) as f:
        for line in f:
            m = data_re.match(line)
            if m:
                pid = int(m.group(1))
                cmd = m.group(3).strip()
                if pid not in all_pids:
                    all_pids.add(pid)
                    print(f"  PID {pid:>6}  {cmd}")
    exit(1)

# --- Align snapshots ---
snapshots_per_pid = {pid: len(v) for pid, v in pid_rss.items()}
n_snapshots = max(snapshots_per_pid.values())

unique_timestamps = []
with open(LOG_FILE) as f:
    for line in f:
        ts_match = timestamp_re.search(line)
        if ts_match:
            unique_timestamps.append(
                datetime.strptime(ts_match.group(1), "%Y-%m-%d %H:%M:%S")
            )

unique_timestamps = unique_timestamps[:n_snapshots]

# Pad shorter series with NaN
for pid in TARGET_PIDS:
    while len(pid_rss[pid]) < n_snapshots:
        pid_rss[pid].insert(0, float("nan"))

# --- Build per-PID labels and colors ---
sorted_pids = sorted(pid_rss.keys())
pid_color = {pid: COLORS[i % len(COLORS)] for i, pid in enumerate(sorted_pids)}
pid_label = {}
for pid in sorted_pids:
    cmd = pid_cmd.get(pid, f"PID {pid}")
    short = cmd.split("/")[-1]
    pid_label[pid] = f"PID {pid} — {short}"

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

pid_list = ", ".join(str(p) for p in sorted_pids)
fig.suptitle(f"vLLM-Omni Memory — PID {pid_list}", fontsize=14)

# Top: absolute RSS
for pid in sorted_pids:
    rss = pid_rss[pid]
    ax1.plot(
        unique_timestamps, rss,
        color=pid_color[pid],
        label=pid_label[pid],
        linewidth=1.5,
    )
    valid = [v for v in rss if v == v]
    if valid:
        growth = valid[-1] - valid[0]
        ax1.annotate(
            f"start {valid[0]:.1f} → end {valid[-1]:.1f} MB  (+{growth:.1f})",
            xy=(unique_timestamps[-1], valid[-1]),
            fontsize=9,
            color=pid_color[pid],
            fontweight="bold",
            ha="right",
            va="bottom",
        )

ax1.set_ylabel("RSS (MB)", fontsize=12)
ax1.set_title("Absolute RSS", fontsize=13)
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(True, alpha=0.3)

# Bottom: relative change from first sample
for pid in sorted_pids:
    rss = pid_rss[pid]
    valid = [(i, v) for i, v in enumerate(rss) if v == v]
    if not valid:
        continue
    base = valid[0][1]
    ts = [unique_timestamps[i] for i, _ in valid]
    delta = [v - base for _, v in valid]
    ax2.plot(ts, delta, color=pid_color[pid], linewidth=2)
    ax2.annotate(
        f"+{delta[-1]:.1f} MB",
        xy=(ts[-1], delta[-1]),
        fontsize=11,
        color=pid_color[pid],
        fontweight="bold",
        ha="right",
        va="bottom",
    )

ax2.set_xlabel("Time", fontsize=12)
ax2.set_ylabel("RSS Change (MB)", fontsize=12)
ax2.set_title("Memory Growth Relative to Start", fontsize=13)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax2.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color="grey", linewidth=0.8, linestyle="--")
fig.autofmt_xdate()
plt.tight_layout()

out_path = args.out or LOG_FILE.replace("vllm_mem.log", "mem_leak_plot.png")
fig.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
plt.show()
