# Copyright 2026 Keshan Sodimana
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generate a self-contained, one-click reproducible Google Colab Jupyter Notebook (.ipynb)."""

import json
from pathlib import Path


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def main() -> None:
    root = Path("gemma4_surgery")
    out_nb = root / "notebooks" / "Gemma4_Weight_Surgery_Reproducible_Colab.ipynb"
    out_nb.parent.mkdir(parents=True, exist_ok=True)

    fast_infer_src = (root / "benchmarks" / "fast_infer.py").read_text()
    jax_fast_infer_src = (root / "benchmarks" / "jax_fast_infer.py").read_text()
    eval_src = (root / "benchmarks" / "eval_harness.py").read_text()
    surgeon_src = (root / "surgeries" / "weight_surgeon.py").read_text()
    subspace_src = (root / "surgeries" / "subspace_calibration.py").read_text()
    probe_src = (root / "probes" / "phase1_anatomy_probe.py").read_text()
    exp4_src = (root / "experiments" / "exp4_heldout_test_eval.py").read_text()

    cells = [
        md_cell(
            """# Gemma 4 E2B-it Training-Free Weight Surgery & Static-Graph TPU Inference
**Reproducible Research Notebook (Free Google Colab Cloud TPU `v5e-1` / GPU)**

> **License:** Copyright 2026 Keshan Sodimana — Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) (see the repository `LICENSE` file).

This self-contained notebook reproduces:
1. **Model Backbone Isolation**: Loading `unsloth/gemma-4-E2B-it` (`Gemma4ForConditionalGeneration`, `bfloat16`) and pruning unused multimodal encoders (`vision_tower`, `audio_tower`) for pure text reasoning/coding.
2. **Phase 1 35-Layer Spectral & Activation MRI**: Measuring `layer_scalar`, PLE gate clamping (`post_per_layer_input_norm`), and KV-shared cosine velocity (`L15-L34`).
3. **`StaticGemma4Engine` (`48x-51x` Colab TPU Speedup)**: Eliminating XLA per-token recompilations via 15-layer static KV allocation (`-57%` KV HBM), device-tensor `index_copy_`, and 4-step block alignment.
4. **Zero-Training Weight Surgeries (`WeightSurgeon`)**:
   - **`S1d` (Code-Safe Prose-Filler Orthogonal PLE Projection)**: Parameter-free projection $W_{\\text{proj},\\ell}^{\\star} = (I - U_{\\text{prose}} U_{\\text{prose}}^\\top) W_{\\text{proj},\\ell}$ across $\\ell \\in [25, 34]$.
   - **`S1c` (Surface/Stop Subspace PLE Projection)**: Reasoning-focused subspace attenuation across $\\ell \\in [25, 34]$.
   - **`S6b` (Scale-Compensated Layer-31 Bypass + Prose PLE Projection)**: Removing dormant Double-Wide MLP Layer 31 (`-56.6M` active parameters) while folding $\\lambda_{31}$ into `L30.layer_scalar`.
5. **Paired Statistical Evaluation**: GSM8K, ARC-Challenge, MMLU-Pro (STEM), and MBPP (Sanitized) with exact paired McNemar binomial tests and 2,000-sample bootstrap 95% CIs."""
        ),
        code_cell(
            """# @title 1. Setup Workspace Directories & Check Accelerator (TPU v5e-1 or GPU)
import os, torch
os.makedirs("/content/gemma4_surgery/benchmarks", exist_ok=True)
os.makedirs("/content/gemma4_surgery/surgeries", exist_ok=True)
os.makedirs("/content/gemma4_surgery/results", exist_ok=True)
open("/content/gemma4_surgery/benchmarks/__init__.py", "w").close()
open("/content/gemma4_surgery/surgeries/__init__.py", "w").close()

DEVICE = torch.accelerator.current_accelerator() if hasattr(torch, "accelerator") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Active Accelerator:", DEVICE, "| PyTorch:", torch.__version__)"""
        ),
        code_cell(
            "# @title 2. Write Evaluation Harness (`benchmarks/eval_harness.py`)\n"
            + "%%writefile /content/gemma4_surgery/benchmarks/eval_harness.py\n"
            + eval_src
        ),
        code_cell(
            "# @title 3. Write Static-Graph Pure-Backbone Engine (`benchmarks/fast_infer.py`)\n"
            + "%%writefile /content/gemma4_surgery/benchmarks/fast_infer.py\n"
            + fast_infer_src
        ),
        code_cell(
            "# @title 4. Write Reversible Weight Surgeon (`surgeries/weight_surgeon.py`)\n"
            + "%%writefile /content/gemma4_surgery/surgeries/weight_surgeon.py\n"
            + surgeon_src
        ),
        code_cell(
            "# @title 5. Write Zero-Gradient Subspace Calibration (`surgeries/subspace_calibration.py`)\n"
            + "%%writefile /content/gemma4_surgery/surgeries/subspace_calibration.py\n"
            + subspace_src
        ),
        code_cell(
            "# @title 6. Run Phase 1 35-Layer Spectral & Activation MRI Probe\n"
            + probe_src
        ),
        code_cell(
            "# @title 7. Run Held-Out Test Evaluation (Baseline vs. Surgical Models)\n"
            + exp4_src
        ),
        code_cell(
            "# @title 8. Write Pure JAX Static-Graph Engine (`benchmarks/jax_fast_infer.py`)\n"
            + "%%writefile /content/gemma4_surgery/benchmarks/jax_fast_infer.py\n"
            + jax_fast_infer_src
        ),
        code_cell(
            """# @title 9. Run Pure JAX KV Microbenchmark & JaxStaticGemma4Engine (1,780.74 -> 1,849.65 tok/s)
import gc
import sys
if "/content" not in sys.path:
    sys.path.insert(0, "/content")

import importlib
import gemma4_surgery.benchmarks.jax_fast_infer as _jfi
importlib.reload(_jfi)
from gemma4_surgery.benchmarks.jax_fast_infer import run_jax_kv_update_microbenchmark, JaxStaticGemma4Engine

print("=== 16-Step JAX KV-Cache Update Microbenchmark ===")
print(run_jax_kv_update_microbenchmark())

if "MODEL" in globals() and MODEL is not None:
    MODEL_CPU = MODEL.cpu()
    gc.collect()
    JAX_ENGINE = JaxStaticGemma4Engine(MODEL_CPU, TOKENIZER)
    del MODEL_CPU
    gc.collect()

    smoke = ["Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"]
    fmt = JAX_ENGINE.format_chat_prompts(smoke, enable_thinking=False)
    for b, max_new in [(1, 97), (8, 129), (16, 129), (32, 129)]:
        _ = JAX_ENGINE.generate_batch_jax_scan(fmt * b, max_new_tokens=max_new, prompt_pad_bucket=128)
        r_warm = JAX_ENGINE.generate_batch_jax_scan(fmt * b, max_new_tokens=max_new, prompt_pad_bucket=128)
        print(f"Batch={b:2d} | Warm Fused Scan: {r_warm['raw_step_tok_per_sec']:7.2f} tok/s ({r_warm['elapsed_sec']:.4f}s)")"""
        ),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "TPU",
            "colab": {"name": "Gemma4_Weight_Surgery_Reproducible_Colab.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_nb.write_text(json.dumps(nb, indent=2))
    print("Generated standalone Colab notebook:", out_nb)


if __name__ == "__main__":
    main()
