# applications.ai.openvino.samples.clip

This project is OpenVINO based CLIP for image search.

Please following below steps to get ov models and run inference.  

1. Please download python e.g. 3.12
2. Following below commands to create virtual envirenment and install required libraries. 
    ```sh
    python -m venv clip_venv
    clip_venv\Scripts\activate
    python -m pip install --upgrade pip wheel setuptools
    pip install -r requirements.txt
    ```
3. Get static text embedding model with offloaded post processing
    ```sh
    python get_static_text_model_clip_ViT_B_32_multilingual.py
    ```
    > __Notes__ 
    > In sentence-transformers/clip-ViT-B-32-multilingual-v1 model, the max token length is `128`, we set 128 as default static shape token length.
    > If you want to change the token length for text embedding model, please modify seq_length in file `get_static_text_model_clip_ViT_B_32_multilingual.py`.
    > 
    > * code:
    >   ```py
    >   seq_length = 128 
    >   ov_inputs = {
    >       "input_ids": [1, seq_length],
    >       "attention_mask": [1, seq_length],
    >   }
    >   ```
    > 
   
4. Get static vision embedding model with offloaded preprocessing and post processing
    ```sh
    python get_static_vision_model_openai_clip_ViT_patch32.py
    ```
5. Get openvino tokenizer
    ```sh
    python get_ov_tokenizer_clip_ViT_B_32_multilingual.py
    ```
6. Run inference for sentence-transformers/clip-ViT-B-32-multilingual-v1 (text embedding) + openai/clip-vit-base-patch32 (vision embedding)
    * Run sample for models include preprocessing and post processing
        ```py
        python infer_clip_ViT_B_32.py
        ```

    * Simple test Text embedding
        ```py
        python get_text_embedding_clip_ViT_B_32_multilingual.py
        ```
    * simple test Vision embedding
        ```py
        python get_vision_embedding_openai_clip_ViT_patch32.py
        ```

## Benchmark (latency / throughput / memory / cache)

This repo also ships two benchmark drivers for sweeping `{vision, text} × {CPU, GPU, NPU}`:

| Script                  | What it runs                                                                                                  |
|-------------------------|---------------------------------------------------------------------------------------------------------------|
| `bench_app.py`          | Wraps Intel's `openvino.tools.benchmark` (the `benchmark_app` tool). Reports read / compile / first / avg / median / FPS plus `.blob` cache size and process RSS delta. NPU uses an in-process OV-API fallback that reuses the cached `.blob` to work around a 2026.3 dev-build pybind issue. |
| `bench_official_sync.py`| Wraps the unmodified Intel sample `samples/python/benchmark/sync_benchmark/sync_benchmark.py` from the toolkit distribution and prints a side-by-side comparison vs `bench_app.py`. |
| `_bench_launcher.py`    | Tiny entry-point used by `bench_app.py` to invoke `openvino.tools.benchmark.main.main()` as a subprocess (the bundled `main.py` has no `__main__` guard). |

Usage (after activating `clip_venv` and sourcing the OpenVINO `setupvars.ps1`):

```powershell
# Full sweep with our wrapper, 60 s/pair
python bench_app.py -t 60 --save-json bench_report.json

# Cross-check against the official Intel sync_benchmark.py sample
python bench_official_sync.py --compare bench_report.json
```

Latest test report: see [`2026-05-22-benchmark-results.md`](./2026-05-22-benchmark-results.md).

---

## NPU benchmark_app result (OpenVINO 2026.1, latency / sync, 60 s, Intel Core 5 320 + Intel AI Boost NPU)

Direct cross-validation using the toolkit's `benchmark_app` (no custom wrappers).

| 模型 | 階段 | read_ms | compile_ms | 1st_ms | median_ms | avg_ms | min_ms | max_ms | FPS | iters |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text  (`*_text_static_opt_npu.xml`, `[1,128]` i64)         | 1st cold              |  38.11 | **2194.21** | 19.82 | 7.89 | 7.89 | 6.90 | 21.27 | **125.13** | 7508 |
| text                                                       | 2nd warm (cache hit)  |  15.99 |  **640.85** | 51.23 | **7.84** | 7.86 | 6.85 | 23.78 | **125.63** | 7538 |
| vision (`*_vision_static_fully_opt.xml`, `[1,3,224,224]` u8)| 1st cold              |  26.51 | **2787.33** | 21.66 | 8.13 | 8.23 | 7.84 | 46.52 | **119.83** | 7190 |
| vision                                                     | 2nd warm (cache hit)  |  15.22 |  **588.95** | 41.47 | **8.16** | 8.24 | 7.85 | 38.68 | **119.89** | 7194 |

Notes:
- Static shape, `LATENCY` hint, single inference request, `INFERENCE_PRECISION_HINT = f16`, `NPU_PLATFORM = 5020`.
- 2nd run shows `LOADED_FROM_CACHE: True` — NPU blob cache works; compile drops 3.4x (text) / 4.7x (vision).
- Text IR uses the pre-fixed `*_npu.xml` variant produced by `fix_text_ir_for_npu.py` (already shipped in `ov_models/`).
- Vision IR `pixel_values` is rewritten to `u8` by the embedded PrePostProcessor; benchmark_app picks this up automatically.

### Exact reproducible steps (PowerShell)

```powershell
# 0) one-time: clean environment so nothing else leaks into OV
Set-Location C:\working\commercial\clip\clip-vit-B32-multilingual-v1-vision-base-patch32
Remove-Item Env:INTEL_OPENVINO_DIR -EA SilentlyContinue
$env:PYTHONPATH = ''
$env:PYTHONUNBUFFERED = '1'

# 1) activate the OV 2026.1 venv (benchmark_app ships with the openvino wheel)
.\clip_venv_ov2026_1\Scripts\Activate.ps1

# 2) sanity check: NPU must be visible
python -c "import openvino as ov; c=ov.Core(); print(c.available_devices)"
#   Expected: ['CPU', 'GPU', 'GPU.1', 'NPU']

# 3) clean cache dirs so the 1st run is truly cold
Remove-Item -Recurse -Force .\bench_cache\ba_text_NPU,.\bench_cache\ba_vision_NPU -EA SilentlyContinue

# 4) TEXT @ NPU -- run twice (cold then warm)
benchmark_app -m .\ov_models\clip-ViT-B-32-multilingual-v1_text_static_opt_npu.xml `
              -d NPU -api sync -t 60 -cdir .\bench_cache\ba_text_NPU `
              2>&1 | Tee-Object .\bench_app_npu_text_1st.log
benchmark_app -m .\ov_models\clip-ViT-B-32-multilingual-v1_text_static_opt_npu.xml `
              -d NPU -api sync -t 60 -cdir .\bench_cache\ba_text_NPU `
              2>&1 | Tee-Object .\bench_app_npu_text_2nd.log

# 5) VISION @ NPU -- run twice (cold then warm)
benchmark_app -m .\ov_models\clip-vit-base-patch32_vision_static_fully_opt.xml `
              -d NPU -api sync -t 60 -cdir .\bench_cache\ba_vision_NPU `
              2>&1 | Tee-Object .\bench_app_npu_vision_1st.log
benchmark_app -m .\ov_models\clip-vit-base-patch32_vision_static_fully_opt.xml `
              -d NPU -api sync -t 60 -cdir .\bench_cache\ba_vision_NPU `
              2>&1 | Tee-Object .\bench_app_npu_vision_2nd.log

# 6) (optional) compare with GPU / CPU just by swapping -d
benchmark_app -m .\ov_models\clip-ViT-B-32-multilingual-v1_text_static_opt.xml -d GPU -api sync -t 60 -cdir .\bench_cache\ba_text_GPU
benchmark_app -m .\ov_models\clip-vit-base-patch32_vision_static_fully_opt.xml -d GPU -api sync -t 60 -cdir .\bench_cache\ba_vision_GPU
```

Flags cheat-sheet:
- `-d NPU` -- target device (use `NPU`, not `NPU.0`).
- `-api sync` -- single inference request (matches `bench_clean.py`'s latency measurement).
- `-t 60` -- 60-second sustained run.
- `-cdir <dir>` -- model cache directory. Same dir on both runs -> 2nd run benefits from cached compiled blob.
- Default `-hint latency` (1 stream) is implied; no need to specify.
- For text IR remember to use the **`_npu.xml`** variant (i64 inputs are still OK on NPU because `fix_text_ir_for_npu.py` already stripped the dynamic ops; if you ever need to regenerate it: `python fix_text_ir_for_npu.py`).

