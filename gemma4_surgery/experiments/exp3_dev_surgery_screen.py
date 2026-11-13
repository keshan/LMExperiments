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
Exp 3: Strict Dev-Split Screening & Ablation of Gemma 4 Weight Surgeries (Zero Test Leakage).

Screens Unmodified Baseline + 10 Individual Surgeries/Controls across 4 Mechanistic Families +
2 Principled Combinations on strictly disjoint `dev` splits (`4 x 32 = 128` problems per candidate):
  - GSM8K (`train` split slice, n=32)
  - ARC-Challenge (`validation` split, n=32)
  - MMLU-Pro STEM (`validation` split, n=32)
  - MBPP Sanitized (`validation` + `train` splits, n=32)

All batches use uniform `(B=32, P=512, G=257)` shapes so TPU/XLA (`torch_tpu`) compiles ONCE!
Saves incremental checkpoints after every candidate to `/content/gemma4_surgery/results/exp3_dev_screen.json`.
"""

import json
import os
import sys
import tarfile
import time
from typing import Any, Callable, Dict, List

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
    compute_domain_centered_correctness_subspace,
    compute_latent_think_subspace,
    compute_surface_stop_subspace,
)
from surgeries.weight_surgeon import WeightSurgeon

OUT_PATH = "/content/gemma4_surgery/results/exp3_dev_screen.json"
os.makedirs("/content/gemma4_surgery/results", exist_ok=True)

if "MODEL" not in globals():
    from transformers import AutoTokenizer, Gemma4ForConditionalGeneration

    DEVICE = torch.accelerator.current_accelerator() if hasattr(torch, "accelerator") else torch.device("cpu")
    REPO_ID = "unsloth/gemma-4-E2B-it"
    TOKENIZER = AutoTokenizer.from_pretrained(REPO_ID)
    MODEL = Gemma4ForConditionalGeneration.from_pretrained(REPO_ID, dtype=torch.bfloat16)
    MODEL.model.vision_tower = None
    MODEL.model.audio_tower = None
    MODEL.model.embed_vision = None
    MODEL.model.embed_audio = None
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()

ENGINE = StaticGemma4Engine(MODEL, TOKENIZER, DEVICE)
SURGEON = WeightSurgeon(ENGINE.text_model)

BATCH_SIZE = 32
PAD_BUCKET = 512
MAX_NEW_TOKENS = 257  # 1 + 64 * 4 (exact 4-step XLA block alignment)

DEV_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "gsm8k": load_gsm8k(split_mode="dev", n=32, seed=42),
    "arc_challenge": load_arc_challenge(split_mode="dev", n=32, seed=42),
    "mmlu_pro": load_mmlu_pro_stem(split_mode="dev", n=32, seed=42),
    "mbpp": load_mbpp_sanitized(split_mode="dev", n=32, seed=42),
}

print("Loaded DEV suites:", {k: len(v) for k, v in DEV_SUITES.items()}, flush=True)


def evaluate_current_weights(tag: str, baseline_records: Dict[str, List[bool]] | None = None) -> Dict[str, Any]:
    suite_results: Dict[str, Any] = {}
    total_correct = 0
    total_items = 0
    total_tokens = 0
    total_sec = 0.0

    for bname, items in DEV_SUITES.items():
        prompts = [it["prompt"] for it in items]
        gen = ENGINE.generate_dataset(
            prompts,
            batch_size=BATCH_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            prompt_pad_bucket=PAD_BUCKET,
            enable_thinking=False,
            sync_every=4,
        )
        correct_flags: List[bool] = []
        for it, out_txt in zip(items, gen["texts"]):
            ok, _ = grade_item(it, out_txt)
            correct_flags.append(bool(ok))

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
        if baseline_records is not None and bname in baseline_records:
            entry["vs_baseline"] = paired_significance(baseline_records[bname], correct_flags)
        suite_results[bname] = entry

        total_correct += sum(correct_flags)
        total_items += len(items)
        total_tokens += gen["total_tokens"]
        total_sec += gen["total_elapsed_sec"]

    overall_acc = total_correct / max(1, total_items)
    res: Dict[str, Any] = {
        "tag": tag,
        "active_surgeries": list(SURGEON.active_log),
        "overall_acc": round(overall_acc, 4),
        "total_correct": total_correct,
        "total_items": total_items,
        "overall_tok_per_sec": round(total_tokens / max(total_sec, 1e-6), 2),
        "total_elapsed_sec": round(total_sec, 2),
        "benchmarks": suite_results,
    }
    if baseline_records is not None:
        flat_base = [f for b in DEV_SUITES for f in baseline_records[b]]
        flat_surg = [f for b in DEV_SUITES for f in suite_results[b]["correct_flags"]]
        res["overall_vs_baseline"] = paired_significance(flat_base, flat_surg)

    print(
        f"[{tag}] overall={res['overall_acc']*100:.2f}% ({total_correct}/{total_items}) | "
        f"speed={res['overall_tok_per_sec']} tok/s ({res['total_elapsed_sec']}s) | "
        + " | ".join(f"{b}={suite_results[b]['acc']*100:.1f}%" for b in DEV_SUITES),
        flush=True,
    )
    return res


all_results: Dict[str, Any] = {"candidates": {}}

# 1. Run Baseline FIRST (also warms up the [32, 512] XLA graph!)
SURGEON.restore_all()
base_res = evaluate_current_weights("00_baseline", baseline_records=None)
all_results["baseline"] = base_res
base_flags = {b: base_res["benchmarks"][b]["correct_flags"] for b in DEV_SUITES}

with open(OUT_PATH, "w") as fh:
    json.dump(all_results, fh, indent=2)

# 2. Extract analytical + dev-only subspaces (reuses warm [32, 512] prefill graph in ~2s!)
t_cal = time.perf_counter()
STOP_BASIS = compute_surface_stop_subspace(ENGINE.lm_head.weight, TOKENIZER, rank=8)
cal_prompts_32 = (
    [x["prompt"] for x in DEV_SUITES["gsm8k"][:8]]
    + [x["prompt"] for x in DEV_SUITES["arc_challenge"][:8]]
    + [x["prompt"] for x in DEV_SUITES["mmlu_pro"][:8]]
    + [x["prompt"] for x in DEV_SUITES["mbpp"][:8]]
)
THINK_BASIS = compute_latent_think_subspace(
    ENGINE.text_model,
    ENGINE.lm_head.weight,
    TOKENIZER,
    dev_prompts=cal_prompts_32,
    target_layers=(33, 34),
    rank=4,
    batch_size=BATCH_SIZE,
    pad_len=PAD_BUCKET,
    device=DEVICE,
)
FISHER_BASIS = compute_domain_centered_correctness_subspace(
    ENGINE.text_model,
    ENGINE.lm_head.weight,
    TOKENIZER,
    dev_suites_prompts={b: [x["prompt"] for x in DEV_SUITES[b]] for b in DEV_SUITES},
    dev_baseline_correct=base_flags,
    target_layers=(33, 34),
    rank=2,
    batch_size=BATCH_SIZE,
    pad_len=PAD_BUCKET,
    device=DEVICE,
)
# Save dev-calibrated basis tensors to disk so Exp 4 held-out test uses frozen dev bases!
torch.save(
    {"stop_basis": STOP_BASIS, "think_basis": THINK_BASIS, "fisher_basis": FISHER_BASIS},
    "/content/gemma4_surgery/results/dev_calibrated_bases.pt",
)
print(f"Calibrated STOP, THINK, and FISHER subspaces on dev in {time.perf_counter() - t_cal:.2f}s", flush=True)

# 3. Define 10 mechanistic surgical candidates + controls
CANDIDATES: List[tuple[str, Callable[[WeightSurgeon], None]]] = [
    (
        "S1a_ple_readout_damp",
        lambda s: (
            s.scale_ple_post_norm(range(25, 33), 0.25),
            s.scale_ple_post_norm([33, 34], 0.50),
        ),
    ),
    (
        "S1b_ple_entry_and_readout_damp",
        lambda s: (
            s.scale_ple_post_norm([15, 16, 17, 18], 0.65),
            s.scale_ple_post_norm(range(25, 33), 0.25),
            s.scale_ple_post_norm([33, 34], 0.50),
        ),
    ),
    (
        "S1c_ple_stop_subspace_clean",
        lambda s: [
            s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", STOP_BASIS, -0.80)
            for l in range(25, 35)
        ],
    ),
    (
        "S2a_global_kv_temp_and_gain",
        lambda s: (
            s.scale_q_norm_temperature([19, 24, 29, 34], 1.18),
            s.scale_post_attn_norm([19, 24, 29, 34], 1.15),
        ),
    ),
    (
        "S2b_kv_shared_temp_taper",
        lambda s: s.scale_q_norm_temperature(
            range(15, 35),
            {l: 1.0 + 0.20 * ((l - 14) / 20.0) for l in range(15, 35)},
        ),
    ),
    (
        "S3a_bypass_30_32_compensated",
        lambda s: s.bypass_kv_shared_layers([30, 31, 32], compensate_scalar=True),
    ),
    (
        "S3b_bypass_31_32_compensated",
        lambda s: s.bypass_kv_shared_layers([31, 32], compensate_scalar=True),
    ),
    (
        "S3_ctrl_naive_drop_30_32_no_comp",
        lambda s: s.bypass_kv_shared_layers([30, 31, 32], compensate_scalar=False),
    ),
    (
        "S4a_latent_think_subspace_graft",
        lambda s: (
            s.graft_residual_subspace_gain("layers.33.mlp.down_proj.weight", THINK_BASIS[33], 0.35),
            s.graft_residual_subspace_gain("layers.34.mlp.down_proj.weight", THINK_BASIS[34], 0.35),
        ),
    ),
    (
        "S4b_fisher_correctness_graft",
        lambda s: (
            s.graft_residual_subspace_gain("layers.33.mlp.down_proj.weight", FISHER_BASIS[33], 0.35),
            s.graft_residual_subspace_gain("layers.34.mlp.down_proj.weight", FISHER_BASIS[34], 0.35),
        ),
    ),
]

for name, apply_fn in CANDIDATES:
    SURGEON.restore_all()
    apply_fn(SURGEON)
    res = evaluate_current_weights(name, baseline_records=base_flags)
    all_results["candidates"][name] = res
    SURGEON.restore_all()
    with open(OUT_PATH, "w") as fh:
        json.dump(all_results, fh, indent=2)

# 4. Compose winning orthogonal mechanisms automatically from dev evidence!
best_s1 = max(
    ["S1a_ple_readout_damp", "S1b_ple_entry_and_readout_damp", "S1c_ple_stop_subspace_clean"],
    key=lambda k: all_results["candidates"][k]["overall_acc"],
)
best_s2 = max(
    ["S2a_global_kv_temp_and_gain", "S2b_kv_shared_temp_taper"],
    key=lambda k: all_results["candidates"][k]["overall_acc"],
)
best_s4 = max(
    ["S4a_latent_think_subspace_graft", "S4b_fisher_correctness_graft"],
    key=lambda k: all_results["candidates"][k]["overall_acc"],
)
fn_map = dict(CANDIDATES)
base_acc = base_res["overall_acc"]

winning_parts = [k for k in [best_s1, best_s2, best_s4] if all_results["candidates"][k]["overall_acc"] > base_acc]
print("Winning orthogonal dev families:", winning_parts, flush=True)

if len(winning_parts) >= 2:
    SURGEON.restore_all()
    for wk in winning_parts:
        fn_map[wk](SURGEON)
    combo_res = evaluate_current_weights("S5_combo_accuracy", baseline_records=base_flags)
    all_results["candidates"]["S5_combo_accuracy"] = combo_res
    SURGEON.restore_all()

best_bypass = max(
    ["S3a_bypass_30_32_compensated", "S3b_bypass_31_32_compensated"],
    key=lambda k: all_results["candidates"][k]["overall_acc"],
)
SURGEON.restore_all()
for wk in winning_parts:
    fn_map[wk](SURGEON)
fn_map[best_bypass](SURGEON)
pareto_res = evaluate_current_weights("S6_combo_pareto_speed_acc", baseline_records=base_flags)
all_results["candidates"]["S6_combo_pareto_speed_acc"] = pareto_res
SURGEON.restore_all()

with open(OUT_PATH, "w") as fh:
    json.dump(all_results, fh, indent=2)
print("Completed Exp 3 dev screening! Saved to", OUT_PATH, flush=True)
