#!/usr/bin/env python3
"""Monitor all running/recent HF Jobs with live status updates.

Usage:
    uv run python scripts/monitor_jobs.py
    uv run python scripts/monitor_jobs.py --interval 15
    uv run python scripts/monitor_jobs.py --logs  # also stream logs from running jobs
"""

import argparse
import time
from datetime import datetime
from datetime import timezone

from huggingface_hub import fetch_job_logs
from huggingface_hub import list_jobs

STAGE_ICONS = {
    "RUNNING": "🟢",
    "COMPLETED": "✅",
    "ERROR": "❌",
    "CANCELED": "⚪",
    "DELETED": "🗑️",
}


def format_duration(start: datetime) -> str:
    now = datetime.now(timezone.utc)
    delta = now - start
    minutes = int(delta.total_seconds() // 60)
    seconds = int(delta.total_seconds() % 60)
    if minutes >= 60:
        hours = minutes // 60
        minutes = minutes % 60
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{seconds:02d}s"


def get_shard_info(command: list[str]) -> str:
    """Extract shard info from job command."""
    cmd = " ".join(command) if command else ""
    if "--shard-id" in cmd:
        parts = cmd.split()
        for i, p in enumerate(parts):
            if p == "--shard-id" and i + 1 < len(parts):
                shard_id = parts[i + 1]
                num_shards = "?"
                for j, q in enumerate(parts):
                    if q == "--num-shards" and j + 1 < len(parts):
                        num_shards = parts[j + 1]
                return f"shard {shard_id}/{num_shards}"
    if "label.py" in cmd:
        return "label"
    if "preprocess.py" in cmd:
        return "preprocess"
    if "train.py" in cmd:
        return "train"
    return "unknown"


def print_dashboard(show_logs: bool = False) -> None:
    jobs = list(list_jobs())

    # Filter to recent acestep jobs (last 24h, or all running)
    relevant = []
    for job in jobs:
        stage = str(job.status.stage).split(".")[-1]
        cmd = " ".join(job.command) if job.command else ""
        if "acestep" not in cmd:
            continue
        relevant.append(job)

    if not relevant:
        print("No acestep jobs found.")
        return

    # Group by stage
    running = [j for j in relevant if str(j.status.stage).split(".")[-1] == "RUNNING"]
    completed = [j for j in relevant if str(j.status.stage).split(".")[-1] == "COMPLETED"]
    errored = [j for j in relevant if str(j.status.stage).split(".")[-1] == "ERROR"]
    other = [
        j
        for j in relevant
        if str(j.status.stage).split(".")[-1] not in ("RUNNING", "COMPLETED", "ERROR")
    ]

    print(f"ACE-Step Jobs — {datetime.now().strftime('%H:%M:%S')}")
    print(f"Running: {len(running)}  Completed: {len(completed)}  Errors: {len(errored)}\n")

    for job in running + errored + completed + other:
        stage = str(job.status.stage).split(".")[-1]
        icon = STAGE_ICONS.get(stage, "?")
        shard = get_shard_info(job.command)
        duration = format_duration(job.created_at)
        short_id = job.id[:12]
        msg = f"  {icon} {short_id}  {shard:<16s}  {stage:<10s}  {duration}"
        if stage == "ERROR" and job.status.message:
            msg += f"  {job.status.message[:50]}"
        print(msg)

    if show_logs and running:
        print(f"\n{'─' * 60}")
        print(f"Latest logs from {len(running)} running job(s):\n")
        for job in running[:3]:  # limit to 3 to avoid flooding
            shard = get_shard_info(job.command)
            lines = list(fetch_job_logs(job_id=job.id))
            tail = lines[-5:] if lines else ["  (no logs yet)"]
            print(f"  [{shard}]")
            for line in tail:
                print(f"    {line.rstrip()}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Monitor ACE-Step HF Jobs")
    parser.add_argument("--interval", type=int, default=30, help="Refresh interval (seconds)")
    parser.add_argument("--logs", action="store_true", help="Show tail of logs from running jobs")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if args.once:
        print_dashboard(show_logs=args.logs)
        return

    try:
        while True:
            print("\033[2J\033[H", end="")
            print_dashboard(show_logs=args.logs)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
