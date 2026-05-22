"""Run the official OpenVINO toolkit `sync_benchmark.py` sample across our
6 (model x device) pairs and compare to our bench_app.py results.

This wraps the UNMODIFIED stock sample from:
  openvino_toolkit_windows_2026.3.0.dev20260520_x86_64/samples/python/benchmark/sync_benchmark/sync_benchmark.py

That stock sample only takes (model_path, device). It internally:
  * compile_model with PERFORMANCE_HINT=LATENCY
  * fill input tensors with random data
  * one warmup infer
  * loop until (>= 10 s AND >= 10 iters) -- hardcoded
  * reports Median / Average / Min / Max / FPS

We launch it per pair, parse stdout, and emit a comparison table vs the
bench_report.json produced by bench_app.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
TOOLKIT = Path(
    r"C:\working\commercial\clip\openvino_toolkit_windows_2026.3.0.dev20260520_x86_64"
)
SYNC_SAMPLE = TOOLKIT / "samples" / "python" / "benchmark" / "sync_benchmark" / "sync_benchmark.py"

MODELS = {
    "vision": ROOT / "ov_models" / "clip-vit-base-patch32_vision_static_fully_opt.xml",
    "text":   ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml",
}

RE = {
    "count":  re.compile(r"Count:\s+(\d+)\s+iterations"),
    "dur":    re.compile(r"Duration:\s+([\d\.]+)\s+ms"),
    "median": re.compile(r"Median:\s+([\d\.]+)\s+ms"),
    "avg":    re.compile(r"Average:\s+([\d\.]+)\s+ms"),
    "min":    re.compile(r"Min:\s+([\d\.]+)\s+ms"),
    "max":    re.compile(r"Max:\s+([\d\.]+)\s+ms"),
    "fps":    re.compile(r"Throughput:\s+([\d\.]+)\s+FPS"),
    "build":  re.compile(r"Build\s+\.+\s+(\S+)"),
}


def parse(text: str) -> dict:
    out = {}
    for k, rx in RE.items():
        m = rx.search(text)
        out[k] = m.group(1) if m else None
    for k in ("dur", "median", "avg", "min", "max", "fps"):
        out[k] = float(out[k]) if out[k] is not None else None
    if out["count"] is not None:
        out["count"] = int(out["count"])
    return out


def run_one(model_xml: Path, device: str) -> dict:
    cmd = [sys.executable, "-u", str(SYNC_SAMPLE), str(model_xml.resolve()), device]
    print("  cmd: " + " ".join(f'"{a}"' if " " in a else a for a in cmd))
    t0 = time.time()
    cp = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    wall = time.time() - t0
    m = parse(cp.stdout + "\n" + cp.stderr)
    m["returncode"] = cp.returncode
    m["wallclock_s"] = round(wall, 2)
    m["raw_tail"] = (cp.stdout + cp.stderr)[-1500:]
    return m


def fmt(v, suf=""):
    return "n/a" if v is None else (f"{v:.2f}{suf}" if isinstance(v, float) else f"{v}{suf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--save-json", default="bench_official_report.json")
    ap.add_argument("--compare", default="bench_report.json",
                    help="Path to bench_app.py JSON to compare against.")
    args = ap.parse_args()

    print("=== Official OpenVINO sync_benchmark.py sweep ===")
    print(f"  OpenVINO dist : {os.environ.get('INTEL_OPENVINO_DIR', '<not set>')}")
    print(f"  Sample script : {SYNC_SAMPLE}")
    print(f"  Devices       : {args.devices}")
    print(f"  Models        : {args.models}")
    print(f"  Compare JSON  : {args.compare}")
    print("  Sample is hardcoded to >=10s & >=10 iters, LATENCY hint, no CACHE_DIR.")
    print("  Models on disk:")
    for mname in args.models:
        x = MODELS[mname].resolve()
        b = x.with_suffix(".bin")
        print(f"    [{mname}] {x}")
        print(f"           xml={x.stat().st_size/1024:.1f} KB  "
              f"bin={b.stat().st_size/(1024*1024):.1f} MB")

    results: dict[str, dict] = {}
    for mname in args.models:
        for dev in args.devices:
            key = f"{mname}@{dev}"
            print(f"\n=== {key} ===")
            print(f"  model : {MODELS[mname].resolve()}")
            m = run_one(MODELS[mname], dev)
            results[key] = m
            if m["returncode"] != 0:
                print(f"  FAIL exit={m['returncode']}\n  tail:\n{m['raw_tail']}")
                continue
            print(f"  build={m['build']}  count={m['count']}  duration={fmt(m['dur'],' ms')}")
            print(f"  median={fmt(m['median'],' ms')}  avg={fmt(m['avg'],' ms')}  "
                  f"min={fmt(m['min'],' ms')}  max={fmt(m['max'],' ms')}  fps={fmt(m['fps'],' FPS')}")

    Path(args.save_json).write_text(json.dumps(results, indent=2))

    # Comparison vs bench_app.py
    cmp_path = ROOT / args.compare
    ours = {}
    if cmp_path.exists():
        ours = json.loads(cmp_path.read_text())

    print("\n\n=== COMPARISON  (official sync_benchmark.py  vs  bench_app.py) ===")
    hdr = ["pair", "off.avg_ms", "ours.avg_ms", "off.med_ms", "ours.med_ms",
           "off.fps", "ours.fps", "off.count"]
    line = "{:<14} {:>11} {:>11} {:>11} {:>11} {:>10} {:>10} {:>9}"
    print(line.format(*hdr))
    for k in results:
        off = results[k]
        ou = ours.get(k, {})
        print(line.format(
            k,
            fmt(off.get("avg")), fmt(ou.get("avg")),
            fmt(off.get("median")), fmt(ou.get("median")),
            fmt(off.get("fps")), fmt(ou.get("thr")),
            fmt(off.get("count")),
        ))


if __name__ == "__main__":
    main()
