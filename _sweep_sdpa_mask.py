"""比較多種 SDPA mask 重編碼方式對 NPU 內建編譯器 (PREFER_PLUGIN) 的影響。

優先順序：FPS > memory delta > compile time。

針對每種變體：
  * 修補一份 IR 並另存
  * cold-cache 編譯
  * 60s sync 推論測 FPS / latency
  * 監控父+子 RSS delta
  * 比對輸出與原 IR 在 CPU 上的差異 (cos sim)

mask 變體：
  A. f32_large_bias : f32 加性 bias，遮蔽位置 = -1e30
  B. f16_neg65504   : f16 加性 bias，遮蔽位置 = -65504 (f16 最小正規數)
  C. select_neg_inf : Select(cond_bool, 0.0_f32, -1e30_f32)
  D. select_f16     : Select(cond_bool, 0.0_f16, -65504_f16)

每個變體分別測：no_hint / hint=f16
"""
from __future__ import annotations

import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import openvino as ov
import psutil
from openvino import opset13 as ops

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml"
OUT_DIR = ROOT / "ov_models"
CACHE_ROOT = ROOT / "bench_cache" / "_sdpa_sweep"

DUR_SEC = 60.0
WARMUP = 10

LARGE_F32 = 1e30
NEG_F16 = -65504.0  # f16 最小正規數的絕對值


def _wrap_sdpa_to_f16(node) -> None:
    """把整個 SDPA 的 Q/K/V/scale convert 成 f16，輸出 convert 回 f32。"""
    # Q/K/V/scale → f16
    for idx in (0, 1, 2, 4):
        if idx >= len(node.inputs()):
            continue
        port = node.input(idx)
        src = port.get_source_output()
        if src.get_element_type() == ov.Type.f16:
            continue
        cvt = ops.convert(src, ov.Type.f16)
        port.replace_source_output(cvt.output(0))
    # 重新推 type
    node.validate_and_infer_types()
    # 在輸出後插 Convert(f32)
    out = node.output(0)
    consumers = list(out.get_target_inputs())
    cvt_back = ops.convert(out, ov.Type.f32)
    for c in consumers:
        c.replace_source_output(cvt_back.output(0))


def _patch(model: ov.Model, mode: str) -> int:
    n = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "ScaledDotProductAttention":
            continue
        if len(node.inputs()) < 4:
            continue

        target_is_f16 = mode.endswith("_f16")
        if target_is_f16:
            _wrap_sdpa_to_f16(node)

        target_dtype = ov.Type.f16 if target_is_f16 else ov.Type.f32
        np_dtype = np.float16 if target_is_f16 else np.float32
        large_neg = NEG_F16 if target_is_f16 else -LARGE_F32

        mask_in = node.input(3)
        src = mask_in.get_source_output()

        if mode in ("f32_large_bias", "f16_large_bias_f16"):
            fp = ops.convert(src, target_dtype)
            one = ops.constant(np.array(1.0, dtype=np_dtype))
            inverted = ops.subtract(one, fp)
            neg = ops.constant(np.array(large_neg, dtype=np_dtype))
            bias = ops.multiply(inverted, neg)
        elif mode in ("select_neg_inf", "select_neg_f16"):
            cond = ops.convert(src, ov.Type.boolean)
            zero = ops.constant(np.array(0.0, dtype=np_dtype))
            neg = ops.constant(np.array(large_neg, dtype=np_dtype))
            bias = ops.select(cond, zero, neg)
        elif mode == "muladd_f32":
            # bias = src*L - L   (allow:0, mask:-L)。同樣等價、少一個 op。
            fp = ops.convert(src, ov.Type.f32)
            L = ops.constant(np.array(LARGE_F32, dtype=np.float32))
            negL = ops.constant(np.array(-LARGE_F32, dtype=np.float32))
            bias = ops.add(ops.multiply(fp, L), negL)
        else:
            raise ValueError(mode)

        bias.set_friendly_name(node.get_friendly_name() + f"/mask_{mode}")
        mask_in.replace_source_output(bias.output(0))
        n += 1
    return n


@dataclass
class Result:
    variant: str
    hint: str
    cold_compile_ms: float = 0.0
    avg_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    fps: float = 0.0
    iters: int = 0
    rss_baseline_mb: float = 0.0
    rss_peak_mb: float = 0.0
    rss_delta_mb: float = 0.0
    cos_sim: float = 0.0
    max_abs_diff: float = 0.0
    err: str = ""


def measure(variant: str, ir_path: Path, hint: str, ref_out: np.ndarray | None) -> Result:
    r = Result(variant=variant, hint=hint)
    cfg = {}
    if hint == "f16":
        cfg["INFERENCE_PRECISION_HINT"] = "f16"

    cache_dir = CACHE_ROOT / f"{variant}_{hint}"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # RSS 監控
    proc = psutil.Process()
    stop = threading.Event()
    samples = []
    baseline_mb = proc.memory_info().rss / 1024 / 1024

    def mon():
        while not stop.is_set():
            try:
                rss = proc.memory_info().rss
                for ch in proc.children(recursive=True):
                    try:
                        rss += ch.memory_info().rss
                    except psutil.Error:
                        pass
                samples.append(rss)
            except psutil.Error:
                pass
            time.sleep(0.05)

    t_mon = threading.Thread(target=mon, daemon=True)
    t_mon.start()

    try:
        core = ov.Core()
        core.set_property({"CACHE_DIR": str(cache_dir)})
        model = core.read_model(ir_path)
        t0 = time.perf_counter()
        cm = core.compile_model(model, "NPU", cfg)
        r.cold_compile_ms = (time.perf_counter() - t0) * 1000

        ir = cm.create_infer_request()
        ids = np.zeros((1, 128), dtype=np.int64)
        ids[0, :10] = [101, 2023, 2003, 1037, 1010, 1996, 4248, 2829, 4419, 102]
        mask = np.zeros((1, 128), dtype=np.int64)
        mask[0, :10] = 1
        feed = {"input_ids": ids, "attention_mask": mask}

        for _ in range(WARMUP):
            ir.infer(feed)

        out = ir.get_output_tensor(0).data.copy()
        if ref_out is not None:
            diff = np.abs(out - ref_out)
            r.max_abs_diff = float(diff.max())
            r.cos_sim = float(np.dot(out[0], ref_out[0]) / (np.linalg.norm(out[0]) * np.linalg.norm(ref_out[0])))

        times = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < DUR_SEC:
            s = time.perf_counter()
            ir.infer(feed)
            times.append((time.perf_counter() - s) * 1000)
        r.iters = len(times)
        arr = np.array(times)
        r.avg_ms = float(arr.mean())
        r.median_ms = float(np.median(arr))
        r.min_ms = float(arr.min())
        r.fps = 1000.0 / r.avg_ms

    except Exception as e:
        r.err = str(e).splitlines()[0][:200]
    finally:
        stop.set()
        t_mon.join(timeout=1)
        if samples:
            r.rss_baseline_mb = baseline_mb
            r.rss_peak_mb = max(samples) / 1024 / 1024
            r.rss_delta_mb = r.rss_peak_mb - baseline_mb

    return r


def main():
    print(f"OV: {ov.get_version()}")
    print(f"src IR: {SRC}\n")

    core = ov.Core()
    # 先建立 CPU 參考輸出
    print("[ref] computing CPU reference output from ORIGINAL IR ...")
    cm_ref = core.compile_model(SRC, "CPU")
    ids = np.zeros((1, 128), dtype=np.int64)
    ids[0, :10] = [101, 2023, 2003, 1037, 1010, 1996, 4248, 2829, 4419, 102]
    mask = np.zeros((1, 128), dtype=np.int64)
    mask[0, :10] = 1
    ref_out = cm_ref({"input_ids": ids, "attention_mask": mask})[0].copy()
    del cm_ref
    print(f"  ref shape={ref_out.shape}  ref[0,:4]={ref_out[0, :4]}\n")

    variants = [
        "f32_large_bias",       # baseline (現在 NPU IR 使用)
        "muladd_f32",           # f32 但 src*L-L 形式
        "select_neg_inf",       # f32 Select(bool, 0, -L)
        "f16_large_bias_f16",   # Q/K/V/scale/mask 全 f16，(1-x)*-65504
        "select_neg_f16",       # Q/K/V/scale 全 f16，Select(bool, 0, -65504)
    ]
    hints = ["default", "f16"]
    results: list[Result] = []

    for v in variants:
        ir_path = OUT_DIR / f"_sweep_text_{v}.xml"
        m = core.read_model(SRC)
        n = _patch(m, v)
        m.validate_nodes_and_infer_types()
        ov.save_model(m, ir_path, compress_to_fp16=True)
        bin_mb = ir_path.with_suffix(".bin").stat().st_size / 1024 / 1024
        print(f"=== {v} === patched {n} SDPA  (bin={bin_mb:.1f} MB)")

        for h in hints:
            print(f"  -- hint={h} ...", flush=True)
            r = measure(v, ir_path, h, ref_out)
            results.append(r)
            if r.err:
                print(f"     FAIL: {r.err}")
            else:
                print(f"     compile={r.cold_compile_ms:7.1f} ms  fps={r.fps:6.2f}  avg={r.avg_ms:.2f}  rss_delta={r.rss_delta_mb:6.1f} MB  cos={r.cos_sim:.6f}")

    print("\n\n=== SUMMARY (sorted by FPS desc) ===")
    print(f"{'variant':<18}{'hint':<10}{'compile_ms':>12}{'fps':>10}{'avg_ms':>10}{'med_ms':>10}{'rss_d_MB':>10}{'cos_sim':>12}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: -x.fps):
        if r.err:
            print(f"{r.variant:<18}{r.hint:<10}  FAIL: {r.err}")
        else:
            print(f"{r.variant:<18}{r.hint:<10}{r.cold_compile_ms:>12.1f}{r.fps:>10.2f}{r.avg_ms:>10.2f}{r.median_ms:>10.2f}{r.rss_delta_mb:>10.1f}{r.cos_sim:>12.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
