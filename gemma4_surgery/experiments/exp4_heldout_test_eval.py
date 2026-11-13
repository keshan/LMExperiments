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

"""
Exp 4: Modular Checkpointed Held-Out `test`-Split Evaluation (`N = 384`: 96 per benchmark).

Supports incremental execution via `os.environ.get("EXP4_STAGE", "all")`:
  - `EXP4_STAGE=baseline` -> loads model/test suites, runs `00_baseline`, caches `EXP4_STATE` in kernel & JSON
  - `EXP4_STAGE=S1d`      -> runs `S1d_prose_orth_proj_25_34` against cached baseline & updates JSON
  - `EXP4_STAGE=S1c`      -> runs `S1c_ple_stop_subspace_clean` against cached baseline & updates JSON
  - `EXP4_STAGE=S6b`      -> runs `S6b_bypass_L31_comp_plus_prose_orth` against cached baseline & updates JSON
  - `EXP4_STAGE=all`      -> runs baseline + all three locked finalists sequentially
"""

import json
import os
import subprocess
import sys
import tarfile
from typing import Any, Callable, Dict, List

try:
    import datasets  # noqa: F401
    from transformers import Gemma4ForConditionalGeneration  # noqa: F401
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "-U", "transformers", "datasets", "accelerate", "safetensors"]
    )

import torch

if os.path.exists("/content/gemma4_pkg.tar.gz"):
    with tarfile.open("/content/gemma4_pkg.tar.gz", "r:gz") as tf:
        tf.extractall("/content", filter="data")

for _mod in list(sys.modules):
    if _mod == "benchmarks" or _mod.startswith("benchmarks.") or _mod == "surgeries" or _mod.startswith("surgeries."):
        del sys.modules[_mod]

sys.path.insert(0, "/content/gemma4_surgery")
from benchmarks.eval_harness import (
    grade_item,
    load_arc_challenge,
    load_gsm8k,
    load_mbpp_sanitized,
    load_mmlu_pro_stem,
    paired_significance,
)
from benchmarks.fast_infer import StaticGemma4Engine
from surgeries.subspace_calibration import (
    compute_prose_filler_subspace,
    compute_surface_stop_subspace,
)
from surgeries.weight_surgeon import WeightSurgeon

OUT_PATH = "/content/gemma4_surgery/results/exp4_heldout_test_results.json"
os.makedirs("/content/gemma4_surgery/results", exist_ok=True)

STAGE = os.environ.get("EXP4_STAGE", "all").strip()

if "MODEL" not in globals():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    DEVICE = torch.accelerator.current_accelerator() if hasattr(torch, "accelerator") else torch.device("cpu")
    REPO_ID = "unsloth/gemma-4-E2B-it"
    print(f"[Loading {REPO_ID} onto {DEVICE}...]", flush=True)
    TOKENIZER = AutoTokenizer.from_pretrained(REPO_ID)
    MODEL = Gemma4ForConditionalGeneration.from_pretrained(REPO_ID, dtype=torch.bfloat16)
    MODEL.model.vision_tower = None
    MODEL.model.audio_tower = None
    MODEL.model.embed_vision = None
    MODEL.model.embed_audio = None
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

if "EXP4_ENGINE" not in globals():
    EXP4_ENGINE = StaticGemma4Engine(MODEL, TOKENIZER, DEVICE)
if "EXP4_SURGEON" not in globals():
    EXP4_SURGEON = WeightSurgeon(EXP4_ENGINE.text_model)

BATCH_SIZE = 32
PAD_BUCKET = 512
MAX_NEW_TOKENS = 257

if "EXP4_TEST_SUITES" not in globals():
    EXP4_TEST_SUITES = {
        "gsm8k": load_gsm8k(split_mode="test", n=96, seed=2026),
        "arc_challenge": load_arc_challenge(split_mode="test", n=96, seed=2026),
        "mmlu_pro": load_mmlu_pro_stem(split_mode="test", n=96, seed=2026),
        "mbpp": load_mbpp_sanitized(split_mode="test", n=96, seed=2026),
    }
    print("Loaded HELD-OUT TEST suites:", {k: len(v) for k, v in EXP4_TEST_SUITES.items()}, flush=True)

PROSE_BASIS = compute_prose_filler_subspace(EXP4_ENGINE.lm_head.weight, TOKENIZER, rank=8)
STOP_BASIS = compute_surface_stop_subspace(EXP4_ENGINE.lm_head.weight, TOKENIZER, rank=8)

LOCKED_RECIPES: Dict[str, tuple[str, Callable[[WeightSurgeon], None]]] = {
    "S1d": (
        "S1d_prose_orth_proj_25_34",
        lambda s: [
            s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", PROSE_BASIS, -1.0)
            for l in range(25, 35)
        ],
    ),
    "S1c": (
        "S1c_ple_stop_subspace_clean",
        lambda s: [
            s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", STOP_BASIS, -0.80)
            for l in range(25, 35)
        ],
    ),
    "S6b": (
        "S6b_bypass_L31_comp_plus_prose_orth",
        lambda s: (
            [
                s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", PROSE_BASIS, -1.0)
                for l in range(25, 35)
            ],
            s.bypass_kv_shared_layers([31], compensate_scalar=True),
        ),
    ),
}


def run_test_eval(
    tag: str,
    baseline_flags: Dict[str, List[bool]] | None = None,
    baseline_texts: Dict[str, List[str]] | None = None,
) -> tuple[Dict[str, Any], Dict[str, List[str]]]:
    suite_results: Dict[str, Any] = {}
    raw_texts: Dict[str, List[str]] = {}
    qualitative_wins: List[Dict[str, Any]] = []
    qualitative_losses: List[Dict[str, Any]] = []

    total_correct = 0
    total_items = 0
    total_tokens = 0
    total_sec = 0.0

    for bname, items in EXP4_TEST_SUITES.items():
        prompts = [it["prompt"] for it in items]
        gen = EXP4_ENGINE.generate_dataset(
            prompts,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            prompt_pad_bucket=PAD_BUCKET,
            enable_thinking=False,
            sync_every=4,
        )
        raw_texts[bname] = gen["texts"]
        correct_flags: List[bool] = []
        for idx, (it, out_txt) in enumerate(zip(items, gen["texts"])):
            ok, _ = grade_item(it, out_txt)
            correct_flags.append(bool(ok))

            if baseline_flags is not None and baseline_texts is not None:
                b_ok = baseline_flags[bname][idx]
                if (not b_ok) and ok and len(qualitative_wins) < 20:
                    qualitative_wins.append({
                        "id": it["id"],
                        "benchmark": bname,
                        "prompt": it["prompt"][:500],
                        "target": str(it["target"])[:200],
                        "baseline_output": baseline_texts[bname][idx][:600],
                        "surgery_output": out_txt[:600],
                    })
                elif b_ok and (not ok) and len(qualitative_losses) < 10:
                    qualitative_losses.append({
                        "id": it["id"],
                        "benchmark": bname,
                        "prompt": it["prompt"][:500],
                        "target": str(it["target"])[:200],
                        "baseline_output": baseline_texts[bname][idx][:600],
                        "surgery_output": out_txt[:600],
                    })

        acc = sum(correct_flags) / len(correct_flags)
        entry: Dict[str, Any] = {
            "n": len(items),
            "acc": round(acc, 4),
            "num_correct": sum(correct_flags),
            "total_tokens": gen["total_tokens"],
            "elapsed_sec": gen["total_elapsed_sec"],
            "tok_per_sec": gen["tok_per_sec"],
            "correct_flags": correct_flags,
        }
        if baseline_flags is not None:
            entry["vs_baseline"] = paired_significance(baseline_flags[bname], correct_flags)
        suite_results[bname] = entry

        total_correct += sum(correct_flags)
        total_items += len(items)
        total_tokens += gen["total_tokens"]
        total_sec += gen["total_elapsed_sec"]

    reasoning_keys = ["gsm8k", "arc_challenge", "mmlu_pro"]
    reas_corr = sum(suite_results[k]["num_correct"] for k in reasoning_keys)
    reas_n = sum(suite_results[k]["n"] for k in reasoning_keys)

    res: Dict[str, Any] = {
        "tag": tag,
        "active_surgeries": list(EXP4_SURGEON.active_log),
        "overall_acc": round(total_correct / total_items, 4),
        "total_correct": total_correct,
        "total_items": total_items,
        "reasoning_3bench_acc": round(reas_corr / reas_n, 4),
        "reasoning_3bench_correct": reas_corr,
        "reasoning_3bench_n": reas_n,
        "overall_tok_per_sec": round(total_tokens / max(total_sec, 1e-6), 2),
        "total_elapsed_sec": round(total_sec, 2),
        "benchmarks": suite_results,
    }
    if baseline_flags is not None:
        flat_base = [f for b in EXP4_TEST_SUITES for f in baseline_flags[b]]
        flat_surg = [f for b in EXP4_TEST_SUITES for f in suite_results[b]["correct_flags"]]
        res["overall_vs_baseline"] = paired_significance(flat_base, flat_surg)

        reas_base = [f for b in reasoning_keys for f in baseline_flags[b]]
        reas_surg = [f for b in reasoning_keys for f in suite_results[b]["correct_flags"]]
        res["reasoning_3bench_vs_baseline"] = paired_significance(reas_base, reas_surg)
        res["qualitative_wins"] = qualitative_wins
        res["qualitative_losses"] = qualitative_losses

    print(
        f"[TEST | {tag}] overall={res['overall_acc']*100:.2f}% ({total_correct}/{total_items}) | "
        f"reasoning={res['reasoning_3bench_acc']*100:.2f}% ({reas_corr}/{reas_n}) | "
        f"speed={res['overall_tok_per_sec']} tok/s ({res['total_elapsed_sec']}s) | "
        + " | ".join(f"{b}={suite_results[b]['acc']*100:.2f}%" for b in EXP4_TEST_SUITES),
        flush=True,
    )
    return res, raw_texts


if "EXP4_RESULTS" not in globals():
    if os.path.exists(OUT_PATH):
        EXP4_RESULTS = json.load(open(OUT_PATH))
    else:
        EXP4_RESULTS = {"models": {}}

if STAGE in ("baseline", "all"):
    EXP4_SURGEON.restore_all()
    base_res, EXP4_BASE_TEXTS = run_test_eval("00_baseline")
    EXP4_RESULTS["baseline"] = base_res
    EXP4_BASE_FLAGS = {b: base_res["benchmarks"][b]["correct_flags"] for b in EXP4_TEST_SUITES}
    with open(OUT_PATH, "w") as fh:
        json.dump(EXP4_RESULTS, fh, indent=2)

stages_to_run = ["S1d", "S1c", "S6b"] if STAGE == "all" else ([STAGE] if STAGE in LOCKED_RECIPES else [])
for skey in stages_to_run:
    full_name, recipe_fn = LOCKED_RECIPES[skey]
    EXP4_SURGEON.restore_all()
    recipe_fn(EXP4_SURGEON)
    m_res, _ = run_test_eval(full_name, baseline_flags=EXP4_BASE_FLAGS, baseline_texts=EXP4_BASE_TEXTS)
    EXP4_RESULTS["models"][full_name] = m_res
    EXP4_SURGEON.restore_all()
    with open(OUT_PATH, "w") as fh:
        json.dump(EXP4_RESULTS, fh, indent=2)

print(f"Finished EXP4_STAGE={STAGE}! Saved to {OUT_PATH}", flush=True)
