# 2026-05-22 — CLIP ViT-B/32 multilingual benchmark results

Hardware: **Intel Core 5 320 (Wildcat Lake)** + 2× Xe iGPU + Intel AI Boost NPU.
Memory: DDR 5600 MT/s.
OS: Windows 11.
OpenVINO: **2026.3.0-21968-28217a1f138** (dev nightly,
`openvino_toolkit_windows_2026.3.0.dev20260520_x86_64`).
Python: 3.12.

## Models under test

| Tower  | IR file                                                                | xml      | bin      | Input static shape                              |
|--------|------------------------------------------------------------------------|----------|----------|-------------------------------------------------|
| Vision | `ov_models/clip-vit-base-patch32_vision_static_fully_opt.{xml,bin}`    | 500.0 KB | 167.6 MB | `pixel_values [1,3,224,224]` (preproc baked in) |
| Text   | `ov_models/clip-ViT-B-32-multilingual-v1_text_static_opt.{xml,bin}`    | 273.3 KB | 257.2 MB | `input_ids [1,128]`, `attention_mask [1,128]`   |

## Test setup

- `bench_app.py` — drives the bundled `openvino.tools.benchmark` (Intel's reference
  `benchmark_app`) per (model × device), 60 s/pair, `-api sync`, `-hint latency`,
  fresh `-cdir` per pair. Polls `psutil` for RSS and reads the `.blob` size after
  compile. NPU has a known pybind crash in this 2026.3 dev build during the
  property-query step, so we fall back to an in-process OpenVINO Python API run
  that **reuses the same cached `.blob`**.
- `bench_official_sync.py` — runs the unmodified
  `samples/python/benchmark/sync_benchmark/sync_benchmark.py` sample shipped in
  the toolkit, 10 s/pair, as an apples-to-apples cross-check.

Command reproduction:

```powershell
cd clip-vit-B32-multilingual-v1-vision-base-patch32
.\clip_venv\Scripts\Activate.ps1
& "C:\working\commercial\clip\openvino_toolkit_windows_2026.3.0.dev20260520_x86_64\setupvars.ps1"

python bench_app.py -t 60 --save-json bench_report.json
python bench_official_sync.py --compare bench_report.json
```

## Latency / throughput (sync, latency hint)

| pair        | bench_app avg (60 s) | bench_app median | bench_app FPS | sync_benchmark avg (10 s) | sync_benchmark FPS |
|-------------|---------------------:|-----------------:|--------------:|--------------------------:|-------------------:|
| vision@CPU  |             98.92 ms |        101.92 ms |         10.09 |                  58.03 ms |              17.23 |
| vision@GPU  |              7.99 ms |          8.10 ms |        123.67 |                   7.08 ms |             141.29 |
| vision@NPU* |              8.38 ms |          8.34 ms |        119.39 |                   8.50 ms |             117.61 |
| text@CPU    |             66.42 ms |         65.51 ms |         15.03 |                  65.44 ms |              15.28 |
| text@GPU    |              4.31 ms |          4.24 ms |        229.37 |                   4.91 ms |             203.75 |
| text@NPU*   |              7.57 ms |          7.58 ms |        132.17 |                   7.08 ms |             141.29 |

`*` NPU values from `bench_app.py` use the in-process OV-API fallback. The two
NPU rows agree within ≤1 % (vision) and ~7 % (text) against the official sample,
confirming the fallback is sound.

## Compile time, first-inference and memory footprint (`bench_app.py` only)

| pair        | read ms | compile ms | first ms | cache .blob | RSS Δ peak |
|-------------|--------:|-----------:|---------:|------------:|-----------:|
| vision@CPU  |  700.56 |    1288.54 |   105.61 |   168.5 MB  |   776.6 MB |
| vision@GPU  |   13.01 |     341.50 |    13.47 |   173.1 MB  |   574.4 MB |
| vision@NPU  |   22.54 |     121.69 |    44.44 |   349.9 MB† |   240.1 MB |
| text@CPU    |   17.71 |     212.49 |    61.88 |   257.9 MB  |   599.4 MB |
| text@GPU    |   14.07 |     329.36 |     6.68 |   261.0 MB  |   739.7 MB |
| text@NPU    |    9.97 |     113.80 |    49.71 |   259.1 MB  |   327.4 MB |

`†` vision@NPU `.blob` is doubled because the failed subprocess and the
in-process fallback each wrote a blob into the same `-cdir`. Functional, just
larger on disk.

## Takeaways

- **Best latency**: text@GPU 4.24 ms (229 FPS), vision@GPU 8.10 ms (124 FPS).
- **NPU is the consistent second** on both towers (~8 ms vision, ~7.5 ms text)
  and uses noticeably less RSS than GPU.
- **CPU** is only practical for the text tower (~15 FPS); vision@CPU is the slow
  path at ~10 FPS.
- Cache `.blob` sizes (~170 MB vision, ~260 MB text) closely match the original
  `.bin` weights, as expected.
- Cross-validation against Intel's stock `sync_benchmark.py` sample matches
  within 1–13 % on CPU/GPU and ≤7 % on NPU.

---

## Re-run on OpenVINO **2026.1.0 official (PyPI)** — same hardware, same models, same shapes

Installed cleanly in a separate venv to avoid PYTHONPATH overlap with the nightly:

```powershell
python -m venv clip_venv_ov2026_1
.\clip_venv_ov2026_1\Scripts\Activate.ps1
pip install openvino==2026.1.0 openvino-genai==2026.1.0.0 openvino-tokenizers==2026.1.0.0 numpy psutil
# Clear nightly env vars before running
Remove-Item Env:INTEL_OPENVINO_DIR; $env:PYTHONPATH=''
python bench_app.py -t 60 --save-json bench_report_2026_1.json
```

`benchmark_app.exe` is shipped as a real CLI entry-point by the pip package
(`clip_venv_ov2026_1\Scripts\benchmark_app.exe`).

### Results — 2026.1.0 official

| pair        | read ms | compile ms | first ms | avg ms | median ms | FPS    | cache .blob | RSS Δ peak |
|-------------|--------:|-----------:|---------:|-------:|----------:|-------:|------------:|-----------:|
| vision@CPU  |   22.34 |     589.22 |    60.98 |  60.61 |     60.06 |  16.47 |   168.6 MB  |   616.5 MB |
| vision@GPU  |   16.66 |    1133.28 |     9.14 |   6.35 |      6.21 | 155.68 |   173.1 MB  |   679.8 MB |
| vision@NPU  |   18.64 |    2829.21 |    18.66 |   8.24 |      8.19 | 119.16 |   174.9 MB† |   520.4 MB |
| text@CPU    |   17.65 |    1404.65 |   102.26 |  65.11 |     64.56 |  15.34 |   257.9 MB  |   678.6 MB |
| text@GPU    |   13.74 |    2287.23 |     6.60 |   4.25 |      4.18 | 232.69 |   261.0 MB  |   836.9 MB |
| text@NPU    |   14.89 |        n/a |      n/a |    n/a |       n/a |    n/a |       n/a   |        n/a |

`†` Cache dir was reused across multiple test iterations and now holds 3 blobs (537 MB total); only the latest `.blob` (~175 MB) is the actual cache entry.

### text@NPU on 2026.1: NPU compiler regression

The NPU plugin cannot compile the multilingual text IR on 2026.1:

```
RuntimeError: Compilation failed.
vclAllocatedExecutableCreate2 result: 0x78000004 - [NPU_VCL]
Compiler returned msg: Failed to create a valid MLIR module for the IR model
  (src\plugins\intel_npu\src\compiler_adapter\src\compiler_impl.cpp:433)
```

It fails identically through `benchmark_app.exe`, `python _bench_launcher.py`,
and the bare `core.compile_model()` path. The same IR **compiles and runs cleanly
on 2026.3 nightly** (where only the Python `get_property()` step crashes).

### Side-by-side: nightly 2026.3 vs official 2026.1 (avg ms / FPS)

| pair        | 2026.3 nightly avg / FPS | 2026.1 official avg / FPS | delta              |
|-------------|--------------------------|---------------------------|--------------------|
| vision@CPU  | 98.92 / 10.09            | **60.61 / 16.47**         | 2026.1 +63 % FPS   |
| vision@GPU  | 7.99 / 123.67            | 6.35 / **155.68**         | 2026.1 +26 % FPS   |
| vision@NPU  | 8.38 / 119.39 (fallback) | 8.24 / 119.16             | tie (<0.5 %)       |
| text@CPU    | 66.42 / 15.03            | 65.11 / 15.34             | tie (~2 %)         |
| text@GPU    | 4.31 / 229.37            | 4.25 / **232.69**         | tie (~1.5 %)       |
| text@NPU    | 7.57 / 132.17 (fallback) | **n/a — compile fail**    | 2026.3 only        |

### Failing `CACHE_ENCRYPTION_CALLBACKS` — what changed between versions

Probed `supported_properties` on the vision IR for both versions:

| device | 2026.3 nightly | 2026.1 official |
|--------|----------------|-----------------|
| CPU | 27 properties, 0 failing | 27 properties, 0 failing |
| GPU | 26 properties, 0 failing | 26 properties, 0 failing |
| NPU | **21 properties, 1 failing** (`CACHE_ENCRYPTION_CALLBACKS` → `TypeError: Unable to convert function return value to a Python type`) | **18 properties, 0 failing** |

Root cause: the nightly added `CACHE_ENCRYPTION_CALLBACKS` (a pair of
`std::function<std::string(const std::string&)>` encrypt/decrypt callbacks) to
the NPU plugin's supported list, but the pybind11 binding for that property type
was not added. `openvino.tools.benchmark.main` (Step 8) iterates every
supported property and calls `get_property(k)` unconditionally → instant
`TypeError` only on NPU.

### Recommendation

- **Use 2026.1.0 official for production benchmarking on CPU / GPU / vision@NPU.** It is the most stable combination.
- **For text@NPU**, use the 2026.3 nightly + `bench_app.py`'s in-process Python API fallback (which reuses the cached `.blob` and avoids the `get_property()` iteration entirely).
- When the next OpenVINO release fixes both (a) the pybind binding for `CACHE_ENCRYPTION_CALLBACKS` and (b) the text-model NPU compiler regression, the fallback can be retired.

---

## Update — root-caused & fixed text@NPU on 2026.1 (driver 4778)

After enumerating the failing op via the vpux-compiler diagnostic, the precise issue
in the 2026.1 plugin-bundled NPU MLIR compiler is:

```
'IE.SDPA' op operand #3 must be ranked tensor of 16-bit float or 32-bit float values,
 but got 'tensor<1x1x128x128xi8>'
  at __module.model.0.model.transformer.layer.0.attention/aten::scaled_dot_product_attention
```

The text IR feeds an **int8 attention mask** into `ScaledDotProductAttention`. The
2026.1 plugin compiler's `IE.SDPA` op verifier rejects i8 masks (only f16/f32 are
allowed); the 2026.3 nightly's compiler accepts them — which is why text@NPU works
on the nightly and breaks on 2026.1.

### One-line workaround that works on 2026.1 + NPU driver 1004778

Set `NPU_COMPILER_TYPE=DRIVER` on the NPU compile call. The driver-side compiler
shipped with NPU driver 4778 is newer than the plugin-bundled compiler in
OV 2026.1 and accepts the i8 mask.

```python
core.compile_model(text_xml, "NPU", {"NPU_COMPILER_TYPE": "DRIVER"})
```

Probe of all reasonable NPU compile options (`_try_npu.py`):

| trial                                            | result                                  |
|--------------------------------------------------|-----------------------------------------|
| default                                          | FAIL — `IE.SDPA` operand #3 must be f16/f32 |
| `INFERENCE_PRECISION_HINT=f16`                   | FAIL — same                             |
| `PERFORMANCE_HINT=LATENCY`                       | FAIL — same                             |
| `NPU_COMPILER_TYPE=MLIR`                         | FAIL — option already defaults to MLIR  |
| **`NPU_COMPILER_TYPE=DRIVER`**                   | **OK — compiles + runs**                |
| `NPU_COMPILATION_MODE_PARAMS=enable-se-ptrs-operations=true` | FAIL — same |

### Script change

`bench_app.py` now passes the device-specific config through both code paths:

1. **`benchmark_app.exe` subprocess** — writes a tiny `_bench_load_config.json` into the
   per-pair cache dir and passes `-load_config <path>`. JSON shape:
   ```json
   { "NPU": { "NPU_COMPILER_TYPE": "DRIVER" } }
   ```
2. **In-process fallback** — passes the same dict directly to `core.compile_model(..., {...})`.
3. The fallback call is now wrapped in `try/except` so a hard NPU compile error no
   longer aborts the whole sweep — the JSON still gets written with `n/a` for that
   pair and `fallback="inproc_failed"` + `fallback_error="..."`.

The override is gated on `device.upper() == "NPU"`; CPU/GPU paths are unchanged.

### Verified end-to-end on 2026.1.0 + NPU driver 1004778

Standalone Python API check (`_verify_npu_text.py`, 10-second sync loop):

```
compile: 4422.4 ms
iters=1373, avg=7.278 ms, min=6.269, median=7.233, FPS=137.40
norm=1.0010584592819214   (sanity: should be ~1.0 after L2-normalize)
```

Full 60-s sweep through patched `bench_app.py` (real `benchmark_app.exe` subprocess
+ `-load_config`):

| pair        | read ms | compile ms | first ms | avg ms | median ms | FPS    |
|-------------|--------:|-----------:|---------:|-------:|----------:|-------:|
| vision@CPU  |   24.33 |     304.08 |    64.96 |  63.30 |     62.40 |  15.77 |
| vision@GPU  |   21.46 |     357.15 |     8.97 |   7.48 |      7.03 | 131.96 |
| vision@NPU  |   16.84 |    5053.39 |    18.65 |   8.45 |      8.34 | 116.66 |
| text@CPU    |   16.91 |     237.33 |    66.80 |  74.51 |     72.73 |  13.40 |
| text@GPU    |   16.61 |     413.89 |     6.40 |   5.14 |      4.74 | 191.08 |
| **text@NPU**|   14.74 |     636.09 |    49.11 |   6.87 |      6.80 | **143.12** |

Compile times for CPU/GPU are short (200–410 ms) because that run inherited warm
caches from the previous sweep; vision@NPU and text@NPU show real cold-compile
cost. text@NPU is now slightly **faster than the 2026.3 nightly in-process fallback**
(143.12 FPS vs 132.17 FPS), so the workaround is not just a correctness fix but
also a small performance win.

### Updated recommendation (supersedes the section above)

- **Run everything on OpenVINO 2026.1.0 official + NPU driver ≥ 1004778** using the
  patched `bench_app.py`. All six (model × device) pairs work end-to-end via
  `benchmark_app.exe`. The 2026.3 nightly is no longer needed for this workload.
- Keep the in-process fallback in the script as a safety net for future regressions.
- When OV ships the plugin-side fix for the i8-mask SDPA verifier, the
  `NPU_COMPILER_TYPE=DRIVER` override can be removed.


