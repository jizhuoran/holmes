"""Per-station GPU utilisation from an nvidia-smi sampler aligned with ``[station-ts]`` lines (HOLMES_STATION_TS=1).

    nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv,noheader -lms 50 > util.csv
    python bench/cmp/gpu_util_report.py run.log util.csv [--layer-index 1]

``utilization.gpu`` is the fraction of the sampling period with at least one kernel running (coarse: the driver's
own period is 1/6 .. 1 s), so station means are a smoothed lower bound on idle time; a station at 60 % means the
device waited on the host for roughly 40 % of its wall.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("csv")
    ap.add_argument("--layer-index", type=int, default=-1, help="which occurrence of the station sequence (default last)")
    args = ap.parse_args()
    samples = []
    for line in open(args.csv):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3 or not parts[0][:2].isdigit():
            continue
        ts = dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        samples.append((ts, float(parts[1].split()[0]), float(parts[2].split()[0])))
    spans = []
    starts = {}
    for line in open(args.log):
        m = re.match(r"\[station-ts\] (start|end) (\S+) ([0-9.]+)(?: ([0-9.]+))?", line.strip())
        if not m:
            continue
        kind, name, ts = m.group(1), m.group(2), float(m.group(3))
        if kind == "start":
            starts[name] = ts
        elif name in starts:
            spans.append((name, starts.pop(name), ts))
    # group into layers by the first station name
    layers, cur = [], []
    first = spans[0][0] if spans else None
    for span in spans:
        if span[0] == first and cur:
            layers.append(cur)
            cur = []
        cur.append(span)
    if cur:
        layers.append(cur)
    layer = layers[args.layer_index]
    print(f"{'station':16s} {'wall':>7s} {'gpu%':>6s} {'mem%':>6s} {'idle_s':>7s}  samples")
    tot_wall = tot_idle = 0.0
    for name, a, b in layer:
        sel = [s for s in samples if a <= s[0] <= b]
        if not sel:
            print(f"{name:16s} {b - a:7.3f}  (no samples)")
            continue
        gpu = sum(s[1] for s in sel) / len(sel)
        mem = sum(s[2] for s in sel) / len(sel)
        idle = (b - a) * (1 - gpu / 100.0)
        tot_wall += b - a
        tot_idle += idle
        print(f"{name:16s} {b - a:7.3f} {gpu:6.1f} {mem:6.1f} {idle:7.3f}  {len(sel)}")
    print(f"{'total':16s} {tot_wall:7.3f} {'':6s} {'':6s} {tot_idle:7.3f}")


if __name__ == "__main__":
    main()
