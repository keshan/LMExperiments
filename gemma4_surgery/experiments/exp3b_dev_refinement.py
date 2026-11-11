"""
Exp 3b: Mechanistic Refinement on Dev Splits (Testing Code-Safe Prose Orthogonal Projection
and Single-Layer L31 Dormant Bypass before locking weights for Held-Out Test Evaluation).
"""

import json
import os
import sys
import tarfile
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
    compute_prose_filler_subspace,
    compute_surface_stop_subspace,
)
from surgeries.weight_surgeon import WeightSurgeon

EXP3_PATH = "/content/gemma4_surgery/results/exp3_dev_screen.json"
OUT_PATH = "/content/gemma4_surgery/results/exp3b_dev_refinement.json"

ENGINE = StaticGemma4Engine(MODEL, TOKENIZER, DEVICE)
SURGEON = WeightSurgeon(ENGINE.text_model)

BATCH_SIZE = 32
PAD_BUCKET = 512
MAX_NEW_TOKENS = 257

DEV_SUITES: Dict[str, List[Dict[str, Any]]] = {
    "gsm8k": load_gsm8k(split_mode="dev", n=32, seed=42),
    "arc_challenge": load_arc_challenge(split_mode="dev", n=32, seed=42),
    "mmlu_pro": load_mmlu_pro_stem(split_mode="dev", n=32, seed=42),
    "mbpp": load_mbpp_sanitized(split_mode="dev", n=32, seed=42),
}

prev_exp3 = json.load(open(EXP3_PATH))
base_flags = {b: prev_exp3["baseline"]["benchmarks"][b]["correct_flags"] for b in DEV_SUITES}
saved_bases = torch.load("/content/gemma4_surgery/results/dev_calibrated_bases.pt", weights_only=True)
THINK_BASIS = saved_bases["think_basis"]

PROSE_BASIS = compute_prose_filler_subspace(ENGINE.lm_head.weight, TOKENIZER, rank=8)
STOP_BASIS = compute_surface_stop_subspace(ENGINE.lm_head.weight, TOKENIZER, rank=8)


def evaluate_current_weights(tag: str) -> Dict[str, Any]:
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
        correct_flags = [bool(grade_item(it, txt)[0]) for it, txt in zip(items, gen["texts"])]
        acc = sum(correct_flags) / len(correct_flags)
        suite_results[bname] = {
            "n": len(items),
            "acc": round(acc, 4),
            "num_correct": sum(correct_flags),
            "total_tokens": gen["total_tokens"],
            "elapsed_sec": gen["total_elapsed_sec"],
            "tok_per_sec": gen["tok_per_sec"],
            "correct_flags": correct_flags,
            "vs_baseline": paired_significance(base_flags[bname], correct_flags),
        }
        total_correct += sum(correct_flags)
        total_items += len(items)
        total_tokens += gen["total_tokens"]
        total_sec += gen["total_elapsed_sec"]

    flat_base = [f for b in DEV_SUITES for f in base_flags[b]]
    flat_surg = [f for b in DEV_SUITES for f in suite_results[b]["correct_flags"]]
    res = {
        "tag": tag,
        "active_surgeries": list(SURGEON.active_log),
        "overall_acc": round(total_correct / total_items, 4),
        "total_correct": total_correct,
        "total_items": total_items,
        "overall_tok_per_sec": round(total_tokens / max(total_sec, 1e-6), 2),
        "total_elapsed_sec": round(total_sec, 2),
        "overall_vs_baseline": paired_significance(flat_base, flat_surg),
        "benchmarks": suite_results,
    }
    print(
        f"[{tag}] overall={res['overall_acc']*100:.2f}% ({total_correct}/{total_items}) | "
        f"speed={res['overall_tok_per_sec']} tok/s ({res['total_elapsed_sec']}s) | "
        + " | ".join(f"{b}={suite_results[b]['acc']*100:.1f}%" for b in DEV_SUITES),
        flush=True,
    )
    return res


REFINED_CANDIDATES: List[tuple[str, Callable[[WeightSurgeon], None]]] = [
    # 1. Exact parameter-free orthogonal projection (-1.0) of CODE-SAFE prose filler subspace on L25-34
    (
        "S1d_prose_orth_proj_25_34",
        lambda s: [
            s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", PROSE_BASIS, -1.0)
            for l in range(25, 35)
        ],
    ),
    # 2. Exact parameter-free orthogonal projection (-1.0) of surface stop subspace on L25-34
    (
        "S1e_stop_orth_proj_25_34",
        lambda s: [
            s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", STOP_BASIS, -1.0)
            for l in range(25, 35)
        ],
    ),
    # 3. Code-safe prose orthogonal projection + gentle KV-shared q_norm taper + latent think graft
    (
        "S5b_prose_orth_plus_q_and_think",
        lambda s: (
            [
                s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", PROSE_BASIS, -1.0)
                for l in range(25, 35)
            ],
            s.scale_q_norm_temperature(
                range(15, 35),
                {l: 1.0 + 0.10 * ((l - 14) / 20.0) for l in range(15, 35)},
            ),
            s.graft_residual_subspace_gain("layers.33.mlp.down_proj.weight", THINK_BASIS[33], 0.25),
            s.graft_residual_subspace_gain("layers.34.mlp.down_proj.weight", THINK_BASIS[34], 0.25),
        ),
    ),
    # 4. Single most-dormant layer L31 compensated bypass + prose orthogonal PLE projection
    (
        "S6b_bypass_L31_comp_plus_prose_orth",
        lambda s: (
            [
                s.graft_residual_subspace_gain(f"layers.{l}.per_layer_projection.weight", PROSE_BASIS, -1.0)
                for l in range(25, 35)
            ],
            s.bypass_kv_shared_layers([31], compensate_scalar=True),
        ),
    ),
]

results = {"candidates": {}}
for name, fn in REFINED_CANDIDATES:
    SURGEON.restore_all()
    fn(SURGEON)
    results["candidates"][name] = evaluate_current_weights(name)
    SURGEON.restore_all()
    with open(OUT_PATH, "w") as fh:
        json.dump(results, fh, indent=2)

print("Completed Exp 3b dev refinement! Saved to", OUT_PATH, flush=True)
