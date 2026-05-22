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

