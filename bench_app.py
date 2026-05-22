"""Run OpenVINO benchmark_app on the CLIP text & vision models across CPU/GPU/NPU.

Captures per (model, device):
  * load time (read+compile)
  * first-inference latency
  * average latency
  * throughput
  * cache .blob size (after compile, with fresh cache dir)
  * RSS memory footprint delta (peak during run - baseline before compile)

Memory and cache stats are produced by re-running the same model in a child
process with a fresh cache dir while polling psutil.Process.memory_info().rss
from the parent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).parent.resolve()
CACHE_ROOT = ROOT / "bench_cache"

VISION_XML = ROOT / "ov_models" / "clip-vit-base-patch32_vision_static_fully_opt.xml"
TEXT_XML = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml"

MODELS = {
    "vision": {"xml": VISION_XML, "shape": "pixel_values[1,3,224,224]"},
    "text":   {"xml": TEXT_XML,   "shape": "input_ids[1,128],attention_mask[1,128]"},
}

NITER = 100        # number of inference iterations per measurement
DURATION = 0       # use niter, not time
API = "sync"       # latency-mode

RE = {
    "read":  re.compile(r"Read model took ([\d\.]+) ms"),
    "load":  re.compile(r"Compile model took ([\d\.]+) ms"),
    "first": re.compile(r"First inference took ([\d\.]+) ms"),
    "avg":   re.compile(r"Average:\s+([\d\.]+) ms"),
    "min":   re.compile(r"Min:\s+([\d\.]+) ms"),
    "max":   re.compile(r"Max:\s+([\d\.]+) ms"),
    "median":re.compile(r"Median:\s+([\d\.]+) ms"),
    "thr":   re.compile(r"Throughput:\s+([\d\.]+) FPS"),
}


def parse_metrics(text: str) -> dict:
    out = {}
    for k, rx in RE.items():
        m = rx.search(text)
        out[k] = float(m.group(1)) if m else None
    return out


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _device_config(device: str) -> dict:
    """Device-specific compile properties.

    On OpenVINO 2026.1 official, the plugin-bundled NPU MLIR compiler rejects
    the int8 attention-mask operand emitted by SDPA in the multilingual text
    model ("'IE.SDPA' op operand #3 must be ranked tensor of 16-bit float or
    32-bit float values, but got 'tensor<1x1x128x128xi8>'"). Switching to the
    driver-side compiler (NPU driver >= 1004778 carries a newer compiler that
    accepts i8 masks) sidesteps the bug. The override is harmless for the
    vision model."""
    if device.upper() == "NPU":
        return {"NPU_COMPILER_TYPE": "DRIVER"}
    return {}


def run_benchmark(model_xml: Path, device: str, shape: str, cache_dir: Path,
                  niter: int | None = None, duration: int | None = None) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    launcher = ROOT / "_bench_launcher.py"
    cmd = [
        sys.executable, "-u", str(launcher),
        "-m", str(model_xml.resolve()),
        "-d", device,
        "-api", API,
        "-hint", "latency",
        "-shape", shape,
        "-cdir", str(cache_dir.resolve()),
        "-report_type", "no_counters",
    ]
    dev_cfg = _device_config(device)
    if dev_cfg:
        cfg_path = cache_dir / "_bench_load_config.json"
        cfg_path.write_text(json.dumps({device.upper(): dev_cfg}))
        cmd += ["-load_config", str(cfg_path.resolve())]
    if duration is not None and duration > 0:
        cmd += ["-t", str(duration)]
    if niter is not None and niter > 0:
        cmd += ["-niter", str(niter)]

    print("  cmd: " + " ".join(f'"{a}"' if " " in a else a for a in cmd))

    # Launch child & monitor RSS while a background thread drains stdout
    import threading
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(ROOT),
    )
    chunks: list[str] = []

    def _drain():
        assert proc.stdout is not None
        for line in proc.stdout:
            chunks.append(line)

    drainer = threading.Thread(target=_drain, daemon=True)
    drainer.start()

    p = psutil.Process(proc.pid)
    peak_rss = 0
    baseline_rss = None
    try:
        while proc.poll() is None:
            try:
                rss = p.memory_info().rss
                for ch in p.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.Error:
                        pass
                if baseline_rss is None:
                    baseline_rss = rss
                peak_rss = max(peak_rss, rss)
            except psutil.Error:
                pass
            time.sleep(0.05)
        drainer.join(timeout=5)
    except Exception:
        proc.kill()
        raise

    text = "".join(chunks)
    metrics = parse_metrics(text)
    metrics["cache_bytes"] = dir_size(cache_dir)
    metrics["rss_baseline_mb"] = (baseline_rss or 0) / (1024 * 1024)
    metrics["rss_peak_mb"] = peak_rss / (1024 * 1024)
    metrics["rss_delta_mb"] = (peak_rss - (baseline_rss or 0)) / (1024 * 1024)
    metrics["returncode"] = proc.returncode
    metrics["raw_tail"] = text[-2000:]
    return metrics


def fmt(v, suffix=""):
    return "n/a" if v is None else f"{v:.2f}{suffix}"


INPUT_DTYPES = {
    "vision": {"pixel_values": ("uint8", (1, 3, 224, 224))},
    "text":   {"input_ids":     ("int64", (1, 128)),
               "attention_mask":("int64", (1, 128))},
}


def inproc_latency(model_xml: Path, device: str, model_kind: str, cache_dir: Path,
                   warmup: int = 5, niter: int | None = None,
                   duration: int | None = None) -> dict:
    """Fallback latency measurement using OpenVINO Python API (bypasses benchmark_app
    NPU property-query crash in this dev build). Reuses the same cache_dir so the
    .blob produced earlier is loaded instead of re-compiling from scratch.

    Stop condition: whichever of (niter, duration) is hit first; if both None
    falls back to niter=100."""
    import openvino as ov  # local import to keep parent process light
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})
    t0 = time.perf_counter()
    model = core.read_model(model_xml)
    read_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    cm = core.compile_model(model, device, _device_config(device))
    compile_ms = (time.perf_counter() - t0) * 1000

    feed = {}
    for n, (dt, sh) in INPUT_DTYPES[model_kind].items():
        if dt == "uint8":
            feed[n] = np.zeros(sh, dtype=np.uint8)
        else:
            feed[n] = np.ones(sh, dtype=np.int64)

    req = cm.create_infer_request()
    t0 = time.perf_counter()
    req.infer(feed)
    first_ms = (time.perf_counter() - t0) * 1000
    for _ in range(warmup):
        req.infer(feed)

    if not niter and not duration:
        niter = 100
    samples = []
    deadline = (time.perf_counter() + duration) if duration else None
    i = 0
    while True:
        if niter and i >= niter:
            break
        if deadline and time.perf_counter() >= deadline:
            break
        t0 = time.perf_counter()
        req.infer(feed)
        samples.append((time.perf_counter() - t0) * 1000)
        i += 1
    arr = np.array(samples)
    return {
        "read": read_ms,
        "load": compile_ms,
        "first": first_ms,
        "avg": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "thr": float(1000.0 / arr.mean()),
        "_inproc": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--niter", type=int, default=0,
                    help="Number of inference iterations. 0 = use --duration only.")
    ap.add_argument("-t", "--duration", type=int, default=60,
                    help="Test duration in seconds (default 60). 0 disables duration cap.")
    ap.add_argument("--no-wipe-cache", action="store_true",
                    help="Keep existing bench_cache (warm-cache run).")
    ap.add_argument("--save-json", default="bench_report.json")
    args = ap.parse_args()

    print("=== Benchmark configuration ===")
    print(f"  OpenVINO dist : {os.environ.get('INTEL_OPENVINO_DIR', '<not set>')}")
    print(f"  cwd           : {ROOT}")
    print(f"  cache root    : {CACHE_ROOT}")
    print(f"  devices       : {args.devices}")
    print(f"  models        : {args.models}")
    print(f"  duration (s)  : {args.duration}")
    print(f"  niter         : {args.niter}")
    print(f"  api           : {API}")
    print(f"  hint          : latency")
    print(f"  wipe cache    : {not args.no_wipe_cache}")
    print("  models on disk:")
    for mname in args.models:
        x = MODELS[mname]["xml"].resolve()
        b = x.with_suffix(".bin")
        xml_kb = x.stat().st_size / 1024 if x.exists() else 0
        bin_kb = b.stat().st_size / 1024 if b.exists() else 0
        print(f"    [{mname}] {x}")
        print(f"           shape={MODELS[mname]['shape']}  xml={xml_kb:.1f} KB  bin={bin_kb/1024:.1f} MB")

    # wipe cache root for clean per-pair measurement
    if not args.no_wipe_cache and CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)

    results = {}
    for mname in args.models:
        mcfg = MODELS[mname]
        for dev in args.devices:
            key = f"{mname}@{dev}"
            print(f"\n=== {key} ===")
            print(f"  model : {mcfg['xml'].resolve()}")
            print(f"  shape : {mcfg['shape']}")
            cache_dir = CACHE_ROOT / mname / dev
            print(f"  cdir  : {cache_dir.resolve()}")
            t0 = time.time()
            m = run_benchmark(mcfg["xml"], dev, mcfg["shape"], cache_dir,
                              niter=args.niter or None,
                              duration=args.duration or None)
            m["wallclock_s"] = round(time.time() - t0, 2)
            m["args"] = {"niter": args.niter, "duration": args.duration,
                         "api": API, "hint": "latency",
                         "shape": mcfg["shape"],
                         "model": str(mcfg["xml"].resolve()),
                         "cdir": str(cache_dir.resolve())}
            results[key] = m
            print(
                f"  read={fmt(m['read'],' ms')} compile={fmt(m['load'],' ms')} "
                f"first={fmt(m['first'],' ms')} avg={fmt(m['avg'],' ms')} "
                f"median={fmt(m['median'],' ms')} thr={fmt(m['thr'],' FPS')}"
            )
            cache_files = []
            cd = CACHE_ROOT / mname / dev
            if cd.exists():
                cache_files = [(f.name, f.stat().st_size) for f in cd.iterdir() if f.is_file()]
            print(f"  cache: {m['cache_bytes']/1024:.1f} KB  files={cache_files}")
            print(f"  RSS  baseline={m['rss_baseline_mb']:.1f} MB  peak={m['rss_peak_mb']:.1f} MB  delta={m['rss_delta_mb']:.1f} MB")
            if m["returncode"] != 0:
                print(f"  WARN benchmark_app exit={m['returncode']} (likely NPU property query bug). Running in-process fallback...")
            if m["avg"] is None:
                try:
                    fb = inproc_latency(mcfg["xml"], dev, mname, cache_dir,
                                        niter=args.niter or None,
                                        duration=args.duration or None)
                    # keep cache bytes/memory from subprocess run; overwrite latency fields
                    for k2 in ("read", "load", "first", "avg", "median", "min", "max", "thr"):
                        m[k2] = fb[k2]
                    m["fallback"] = "inproc"
                    print(
                        f"  [inproc] read={fmt(m['read'],' ms')} compile={fmt(m['load'],' ms')} "
                        f"first={fmt(m['first'],' ms')} avg={fmt(m['avg'],' ms')} "
                        f"median={fmt(m['median'],' ms')} thr={fmt(m['thr'],' FPS')}"
                    )
                except Exception as e:
                    m["fallback"] = "inproc_failed"
                    m["fallback_error"] = str(e).splitlines()[0][:300]
                    print(f"  [inproc] FAILED: {m['fallback_error']}")

    Path(args.save_json).write_text(json.dumps(results, indent=2))

    # Pretty summary table
    print("\n\n=== SUMMARY ===")
    hdr = ["pair", "read_ms", "compile_ms", "first_ms", "avg_ms", "median_ms",
           "thr_fps", "cache_KB", "rss_delta_MB"]
    print(("{:<14} " * len(hdr)).format(*hdr))
    for k, m in results.items():
        row = [
            k,
            fmt(m["read"]),
            fmt(m["load"]),
            fmt(m["first"]),
            fmt(m["avg"]),
            fmt(m["median"]),
            fmt(m["thr"]),
            f"{m['cache_bytes']/1024:.1f}",
            f"{m['rss_delta_mb']:.1f}",
        ]
        print(("{:<14} " * len(row)).format(*row))


if __name__ == "__main__":
    main()
