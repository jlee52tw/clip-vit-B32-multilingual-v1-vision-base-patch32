"""Clean per-pair measurement: wipe cache, compile, then measure steady-state
inference RSS (after compile + warmup, so only inference allocations count).

Outputs per pair:
  * read_ms, compile_ms (cold cache)
  * first_ms (first .infer call)
  * avg_ms, median_ms, min_ms (steady-state)
  * fps
  * cache_kb (only this pair's .blob, since cache dir was wiped)
  * inference_rss_delta_mb (peak RSS during steady-state run - RSS after warmup)
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov
import psutil

ROOT = Path(__file__).parent.resolve()
CACHE_ROOT = ROOT / "bench_cache"

VISION_XML = ROOT / "ov_models" / "clip-vit-base-patch32_vision_static_fully_opt.xml"
TEXT_XML = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml"
TEXT_XML_NPU = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt_npu.xml"

MODELS = {
    "vision": {
        "xml_default": VISION_XML,
        "xml_npu": VISION_XML,
        "inputs": lambda: {"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.uint8)},
    },
    "text": {
        "xml_default": TEXT_XML,
        "xml_npu": TEXT_XML_NPU,
        "inputs": lambda: {
            "input_ids": np.ones((1, 128), dtype=np.int64),
            "attention_mask": np.ones((1, 128), dtype=np.int64),
        },
    },
}


def _resolve_xml(mname: str, device: str) -> Path:
    cfg = MODELS[mname]
    return cfg["xml_npu"] if device.upper() == "NPU" else cfg["xml_default"]


def _ensure_text_npu_ir() -> None:
    if TEXT_XML_NPU.exists() and TEXT_XML_NPU.with_suffix(".bin").exists():
        return
    print(f"[setup] generating {TEXT_XML_NPU.name} ...")
    import subprocess
    subprocess.check_call([sys.executable, str(ROOT / "fix_text_ir_for_npu.py")])


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def measure_pair(mname: str, device: str, duration: float, warmup: int,
                 use_driver: bool) -> dict:
    cache_dir = CACHE_ROOT / mname / device
    # CLEAN: nuke any existing cache for this pair so cache_kb counts only this run
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    xml = _resolve_xml(mname, device)
    feed = MODELS[mname]["inputs"]()

    cfg: dict = {}
    if device.upper() == "NPU" and use_driver:
        cfg["NPU_COMPILER_TYPE"] = "DRIVER"

    proc = psutil.Process()
    gc.collect()

    core = ov.Core()
    core.set_property({"CACHE_DIR": str(cache_dir)})

    t0 = time.perf_counter()
    model = core.read_model(xml)
    read_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    cm = core.compile_model(model, device, cfg)
    compile_ms = (time.perf_counter() - t0) * 1000

    req = cm.create_infer_request()

    # First inference latency
    t0 = time.perf_counter()
    req.infer(feed)
    first_ms = (time.perf_counter() - t0) * 1000

    # Warmup (not measured)
    for _ in range(warmup):
        req.infer(feed)

    # Settle, then take the steady-state RSS baseline
    gc.collect()
    time.sleep(0.2)
    base_rss = proc.memory_info().rss

    # Sample RSS in a background thread during the steady-state run
    samples_rss: list[int] = []
    stop_evt = threading.Event()

    def _mon():
        while not stop_evt.is_set():
            try:
                samples_rss.append(proc.memory_info().rss)
            except psutil.Error:
                pass
            time.sleep(0.02)

    mon_t = threading.Thread(target=_mon, daemon=True)
    mon_t.start()

    lat_samples: list[float] = []
    end_at = time.perf_counter() + duration
    while time.perf_counter() < end_at:
        s = time.perf_counter()
        req.infer(feed)
        lat_samples.append((time.perf_counter() - s) * 1000)

    stop_evt.set()
    mon_t.join(timeout=1)

    arr = np.array(lat_samples)
    peak_rss = max(samples_rss) if samples_rss else base_rss
    cache_bytes = dir_size(cache_dir)

    # release before next pair
    del req, cm, model, core
    gc.collect()

    return {
        "model": mname,
        "device": device,
        "xml": str(xml.relative_to(ROOT)),
        "read_ms": read_ms,
        "compile_ms": compile_ms,
        "first_ms": first_ms,
        "iters": len(lat_samples),
        "avg_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": float(1000.0 / arr.mean()),
        "cache_kb": cache_bytes / 1024,
        "inference_rss_baseline_mb": base_rss / 1024 / 1024,
        "inference_rss_peak_mb": peak_rss / 1024 / 1024,
        "inference_rss_delta_mb": (peak_rss - base_rss) / 1024 / 1024,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="+", default=["CPU", "GPU", "NPU"])
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--use-driver-compiler", action="store_true")
    ap.add_argument("--save-json", default="bench_clean.json")
    args = ap.parse_args()

    _ensure_text_npu_ir()

    print(f"OV: {ov.get_version()}")
    print(f"models   : {args.models}")
    print(f"devices  : {args.devices}")
    print(f"duration : {args.duration}s steady-state per pair (+{args.warmup} warmup)")
    print(f"NPU compiler: {'DRIVER (forced)' if args.use_driver_compiler else 'PREFER_PLUGIN (default)'}\n")

    results = []
    for mname in args.models:
        for dev in args.devices:
            print(f"=== {mname}@{dev} ===", flush=True)
            r = measure_pair(mname, dev, args.duration, args.warmup,
                             args.use_driver_compiler)
            results.append(r)
            print(
                f"  xml          : {r['xml']}\n"
                f"  read         : {r['read_ms']:8.2f} ms\n"
                f"  compile      : {r['compile_ms']:8.2f} ms (cold cache)\n"
                f"  first        : {r['first_ms']:8.2f} ms\n"
                f"  avg          : {r['avg_ms']:8.3f} ms  ({r['iters']} iters)\n"
                f"  median       : {r['median_ms']:8.3f} ms\n"
                f"  fps          : {r['fps']:8.2f}\n"
                f"  cache .blob  : {r['cache_kb']:8.1f} KB ({r['cache_kb']/1024:.2f} MB)\n"
                f"  inf RSS base : {r['inference_rss_baseline_mb']:8.1f} MB\n"
                f"  inf RSS peak : {r['inference_rss_peak_mb']:8.1f} MB\n"
                f"  inf RSS delta: {r['inference_rss_delta_mb']:8.1f} MB  (steady-state inference only)\n",
                flush=True,
            )

    Path(args.save_json).write_text(json.dumps(results, indent=2))

    print("\n=== SUMMARY (steady-state inference, clean cache) ===")
    hdr = ("pair", "read_ms", "compile_ms", "first_ms", "avg_ms",
           "med_ms", "fps", "cache_MB", "inf_rss_d_MB")
    print(("{:<14}" * len(hdr)).format(*hdr))
    print("-" * (14 * len(hdr)))
    for r in results:
        print((
            f"{r['model'] + '@' + r['device']:<14}"
            f"{r['read_ms']:<14.2f}"
            f"{r['compile_ms']:<14.2f}"
            f"{r['first_ms']:<14.2f}"
            f"{r['avg_ms']:<14.3f}"
            f"{r['median_ms']:<14.3f}"
            f"{r['fps']:<14.2f}"
            f"{r['cache_kb']/1024:<14.2f}"
            f"{r['inference_rss_delta_mb']:<14.1f}"
        ))


if __name__ == "__main__":
    sys.exit(main())
