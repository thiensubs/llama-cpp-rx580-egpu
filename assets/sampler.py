#!/usr/bin/env python3
"""Sample host RAM, GPU memory and utilisation of the llama-server on a port.

usage: sampler.py <port> <out.csv> <duration_s> [interval_s]

Rows: t_s, rss_mb, cpu_pct, vram_used_mb, vram_free_mb, gpu_pct
The pid is re-resolved every tick so a restart is captured, and a failed probe
records an empty field instead of ending the run.
"""
import csv
import re
import subprocess
import sys
import time


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20).stdout
    except subprocess.TimeoutExpired:
        return ""


def server_pid(port):
    pids = [p for p in sh(f"lsof -tiTCP:{port} -sTCP:LISTEN").split() if p.isdigit()]
    return pids[0] if pids else None


def main():
    port, out, dur = sys.argv[1], sys.argv[2], float(sys.argv[3])
    interval = float(sys.argv[4]) if len(sys.argv) > 4 else 4.0

    t0 = time.time()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "rss_mb", "cpu_pct", "vram_used_mb", "vram_free_mb", "gpu_pct"])
        while time.time() - t0 < dur:
            pid = server_pid(port)
            rss = sh(f"ps -o rss= -p {pid}").strip() if pid else ""
            cpu = sh(f"ps -o %cpu= -p {pid}").strip() if pid else ""
            io = sh("ioreg -r -d 1 -w 0 -c IOAccelerator")
            vram = [int(x) for x in re.findall(r'"inUseVidMemoryBytes"=(\d+)', io)]
            vfree = [int(x) for x in re.findall(r'"vramFreeBytes"=(\d+)', io)]
            gpu = [int(x) for x in re.findall(r'"GPU Activity\(%\)"=(\d+)', io)]
            try:
                w.writerow([
                    round(time.time() - t0, 1),
                    round(int(rss) / 1024) if rss else "",
                    cpu or "",
                    round(max(vram) / 1048576) if vram else "",
                    round(min(vfree) / 1048576) if vfree else "",
                    max(gpu) if gpu else "",
                ])
            except ValueError:
                w.writerow([round(time.time() - t0, 1), "", "", "", "", ""])
            fh.flush()
            time.sleep(interval)


if __name__ == "__main__":
    main()
