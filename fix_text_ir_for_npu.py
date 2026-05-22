"""把既有 text IR 的 SDPA attn_mask（i8）轉成 f16，產生新的 IR：
   ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt_npu.xml

這樣 OpenVINO 2026.1 內建 NPU MLIR 編譯器（PREFER_PLUGIN / MLIR）就能接受。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt.xml"
DST = ROOT / "ov_models" / "clip-ViT-B-32-multilingual-v1_text_static_opt_npu.xml"


def fix_sdpa_mask_dtype(model: ov.Model) -> int:
    """在每個 ScaledDotProductAttention 的 attn_mask 輸入前做語意正確的轉換：
       SDPA 對 i8/bool mask 的語意是「1=允許 attend、0=遮蔽」（內部會把 0 位置加上 -inf）；
       對 float mask 的語意則是「直接加到 attention score」。
       因此把 i8/bool 直接 Convert 成 float 會破壞數值。
       正確等價轉換：fp = (1 - i8) * (-LARGE)，允許→0，遮蔽→-LARGE。
       Q (input #0) 的型別決定目標型別（通常 f32）。
       回傳被修補的節點數量。"""
    LARGE = 1e30  # CPU/GPU 的內建實作通常用 -3.4e38，1e30 已足夠把 softmax 中該位置壓到 0
    patched = 0
    for node in model.get_ordered_ops():
        if node.get_type_name() != "ScaledDotProductAttention":
            continue
        if len(node.inputs()) < 4:
            continue  # 沒有 attn_mask 輸入
        target_dtype = node.input(0).get_source_output().get_element_type()
        mask_in = node.input(3)
        src = mask_in.get_source_output()
        src_dtype = src.get_element_type()
        if src_dtype == target_dtype:
            continue  # 本來就是 float，不必修補

        # i8/bool → float 的語意正確轉換
        as_fp = ops.convert(src, target_dtype)
        one = ops.constant(np.array(1.0, dtype=np.float32).astype(
            np.float16 if target_dtype == ov.Type.f16 else np.float32))
        inverted = ops.subtract(one, as_fp)  # 1->0 (允許), 0->1 (遮蔽)
        neg_large = ops.constant(np.array(-LARGE, dtype=np.float32).astype(
            np.float16 if target_dtype == ov.Type.f16 else np.float32))
        bias = ops.multiply(inverted, neg_large)  # 允許→0, 遮蔽→-LARGE
        bias.set_friendly_name(node.get_friendly_name() + f"/mask_bool_to_{target_dtype.get_type_name()}_bias")
        mask_in.replace_source_output(bias.output(0))
        patched += 1
    return patched


def main():
    print(f"OV: {ov.get_version()}")
    print(f"src: {SRC}")
    if not SRC.exists():
        print("ERROR: source IR not found")
        return 1

    core = ov.Core()
    model = core.read_model(SRC)
    print(f"  inputs : {[(i.get_any_name(), i.get_partial_shape(), i.get_element_type()) for i in model.inputs]}")
    print(f"  outputs: {[(o.get_any_name(), o.get_partial_shape(), o.get_element_type()) for o in model.outputs]}")

    n = fix_sdpa_mask_dtype(model)
    print(f"patched {n} ScaledDotProductAttention nodes (mask i8 -> f16)")
    if n == 0:
        print("nothing to patch; exiting")
        return 0

    model.validate_nodes_and_infer_types()
    ov.save_model(model, DST, compress_to_fp16=True)
    print(f"saved : {DST}  ({DST.stat().st_size} bytes xml, {DST.with_suffix('.bin').stat().st_size} bytes bin)")

    # 數值驗證：原 IR vs 修補後 IR 在 CPU 上的輸出差異
    print("\n[verify] running both IRs on CPU with same inputs ...")
    ids = np.zeros((1, 128), dtype=np.int64)
    ids[0, :5] = [101, 2023, 2003, 1037, 102]
    mask = np.zeros((1, 128), dtype=np.int64)
    mask[0, :5] = 1
    feed = {"input_ids": ids, "attention_mask": mask}

    cm_old = core.compile_model(SRC, "CPU")
    cm_new = core.compile_model(DST, "CPU")
    out_old = cm_old(feed)[0]
    out_new = cm_new(feed)[0]
    diff = np.abs(out_old - out_new)
    print(f"  output shape           : {out_old.shape}")
    print(f"  max |old - new|        : {diff.max():.3e}")
    print(f"  mean |old - new|       : {diff.mean():.3e}")
    print(f"  cos sim                : {float(np.dot(out_old[0], out_new[0]) / (np.linalg.norm(out_old[0]) * np.linalg.norm(out_new[0]))):.6f}")

    # 嘗試用 NPU 內建編譯器（PREFER_PLUGIN, 預設）編譯新 IR
    print("\n[NPU] try compile_model on patched IR with default (PREFER_PLUGIN) compiler ...")
    try:
        t0 = time.perf_counter()
        cm_npu = core.compile_model(DST, "NPU")
        compile_ms = (time.perf_counter() - t0) * 1000
        print(f"  OK  compile = {compile_ms:.1f} ms")
        ir = cm_npu.create_infer_request()
        # warmup + 10s latency loop
        for _ in range(5):
            ir.infer(feed)
        times = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 10.0:
            s = time.perf_counter()
            ir.infer(feed)
            times.append((time.perf_counter() - s) * 1000)
        avg = sum(times) / len(times)
        med = sorted(times)[len(times) // 2]
        print(f"  iters={len(times)}  avg={avg:.3f} ms  median={med:.3f} ms  FPS={1000.0 / avg:.2f}")
    except Exception as e:
        print(f"  FAIL: {str(e).splitlines()[0][:300]}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
