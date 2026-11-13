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
Rigorous Paired Evaluation Harness for Gemma 4 Weight Surgery Experiments.

Enforces strict Calibration/Dev vs. Held-Out Test separation (zero test leakage),
fixed bucket padding + StaticCache support for fast TPU v5e-1 / GPU batched inference,
and paired statistical testing (McNemar exact binomial test + paired bootstrap 95% CI).
"""

import ast
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataset Loaders (Strict Dev vs. Test Splits)
# ---------------------------------------------------------------------------

def load_gsm8k(split_mode: str = "dev", n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """
    split_mode='dev'  -> samples from GSM8K 'train' split (calibration/ablation only)
    split_mode='test' -> samples from GSM8K 'test' split (held-out final evaluation)
    """
    from datasets import load_dataset

    hf_split = "train" if split_mode == "dev" else "test"
    ds = load_dataset("openai/gsm8k", "main", split=hf_split)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    chosen = idxs[: min(n, len(idxs))]

    items = []
    for i in chosen:
        row = ds[i]
        ans_raw = row["answer"]
        target = ans_raw.split("####")[-1].strip().replace(",", "")
        items.append({
            "id": f"gsm8k_{hf_split}_{i}",
            "benchmark": "gsm8k",
            "prompt": (
                "Solve the following math word problem concisely in 3 to 5 short lines of arithmetic "
                "(no markdown headers). End your response with exactly 'FINAL ANSWER: <number>'.\n\n"
                f"Problem: {row['question']}"
            ),
            "question_raw": row["question"],
            "target": target,
        })
    return items


def load_arc_challenge(split_mode: str = "dev", n: int = 80, seed: int = 42) -> List[Dict[str, Any]]:
    """
    split_mode='dev'  -> 'validation' split
    split_mode='test' -> 'test' split
    """
    from datasets import load_dataset

    hf_split = "validation" if split_mode == "dev" else "test"
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=hf_split)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    chosen = idxs[: min(n, len(idxs))]

    label_map = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
    items = []
    for i in chosen:
        row = ds[i]
        texts = row["choices"]["text"]
        labels = [label_map.get(l, l) for l in row["choices"]["label"]]
        ans_key = label_map.get(row["answerKey"], row["answerKey"]).strip().upper()
        if len(texts) != 4 or ans_key not in ("A", "B", "C", "D"):
            continue
        opts_str = "\n".join(f"({labels[j]}) {texts[j]}" for j in range(4))
        items.append({
            "id": f"arc_{hf_split}_{i}",
            "benchmark": "arc_challenge",
            "prompt": (
                "Answer the following science question concisely in 2-3 short sentences, then end "
                "with 'FINAL ANSWER: <letter>' where <letter> is A, B, C, or D.\n\n"
                f"Question: {row['question']}\n\n{opts_str}"
            ),
            "choices": dict(zip(labels, texts)),
            "target": ans_key,
        })
    return items


def load_mmlu_pro_stem(split_mode: str = "dev", n: int = 60, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Challenging 10-option graduate/undergraduate STEM reasoning from TIGER-Lab/MMLU-Pro
    (featured directly in the Gemma 4 Technical Report benchmark suite).
    split_mode='dev'  -> 'validation' (70 items) + disjoint slice if needed
    split_mode='test' -> 'test' split filtered to STEM categories (physics, math, chemistry, cs, biology)
    """
    from datasets import load_dataset

    stem_cats = {"math", "physics", "chemistry", "computer science", "biology", "engineering"}
    if split_mode == "dev":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
        pool = [row for row in ds if row.get("category", "").lower() in stem_cats]
        if len(pool) < n:
            # Supplement from tail of test split reserved strictly for dev? No — never touch test!
            # Use full validation split (all categories, 70 items) so test remains 100% untouched.
            pool = list(ds)
    else:
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        pool = [row for row in ds if row.get("category", "").lower() in stem_cats]

    rng = random.Random(seed)
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = idxs[: min(n, len(idxs))]

    letters = "ABCDEFGHIJ"
    items = []
    for k in chosen:
        row = pool[k]
        opts = row["options"]
        valid_letters = letters[: len(opts)]
        opts_str = "\n".join(f"({valid_letters[j]}) {opts[j]}" for j in range(len(opts)))
        ans_letter = str(row["answer"]).strip().upper()
        items.append({
            "id": f"mmlu_pro_{split_mode}_{row.get('question_id', k)}",
            "benchmark": "mmlu_pro",
            "category": row.get("category", "unknown"),
            "prompt": (
                "Solve the following multiple-choice problem concisely in 3 to 6 short lines (no verbose headers). "
                f"End your response with 'FINAL ANSWER: <letter>' (one of {', '.join(valid_letters)}).\n\n"
                f"Question: {row['question']}\n\n{opts_str}"
            ),
            "valid_letters": valid_letters,
            "target": ans_letter,
        })
    return items


def load_mbpp_sanitized(split_mode: str = "dev", n: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """
    split_mode='dev'  -> MBPP sanitized 'validation' + 'train' splits (never 'test')
    split_mode='test' -> MBPP sanitized 'test' split (257 items)
    """
    from datasets import load_dataset

    if split_mode == "dev":
        ds_val = load_dataset("google-research-datasets/mbpp", "sanitized", split="validation")
        ds_tr = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")
        pool = list(ds_val) + list(ds_tr)
    else:
        pool = list(load_dataset("google-research-datasets/mbpp", "sanitized", split="test"))

    rng = random.Random(seed)
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = idxs[: min(n, len(idxs))]

    items = []
    for i in chosen:
        row = pool[i]
        desc = row.get("prompt") or row.get("text") or ""
        tests = row.get("test_list", [])
        if isinstance(tests, str):
            tests = ast.literal_eval(tests)
        items.append({
            "id": f"mbpp_{split_mode}_{row.get('task_id', i)}",
            "benchmark": "mbpp",
            "prompt": (
                "Write ONLY a self-contained Python 3 solution inside a ```python ... ``` block "
                "(no prose explanation) that satisfies the problem and passes the assertions below.\n\n"
                f"Problem: {desc}\n\nAssertions:\n" + "\n".join(tests[:3])
            ),
            "tests": tests,
            "target": row.get("code", ""),
        })
    return items


# ---------------------------------------------------------------------------
# Deterministic Graders
# ---------------------------------------------------------------------------

def strip_thinking_block(text: str) -> str:
    """Extract final user-facing answer after Gemma 4 thought block / turn delimiters."""
    cleaned = text
    # Strip trailing Gemma 4 turn end markers if decoded with skip_special_tokens=False
    for end_tok in ["<turn|>", "<end_of_turn>", "<eos>"]:
        if end_tok in cleaned:
            cleaned = cleaned.split(end_tok)[0]

    # Native Gemma 4 thinking channel: <|channel>thought\n...\n<channel|>
    if "<channel|>" in cleaned:
        after = cleaned.split("<channel|>")[-1].strip()
        if after:
            return after

    # Legacy / generic thought tags fallback
    for sep in ["<|thought|>", "<thought>"]:
        if sep in cleaned:
            for end_sep in ["<|end_of_thought|>", "</thought>"]:
                if end_sep in cleaned and cleaned.rfind(end_sep) > cleaned.find(sep):
                    after = cleaned.split(end_sep)[-1].strip()
                    if after:
                        return after
    return cleaned.strip()


def grade_gsm8k(output: str, target: str) -> Tuple[bool, Optional[str]]:
    clean = strip_thinking_block(output)
    target_norm = target.strip().replace(",", "")
    try:
        target_val = float(target_norm)
    except ValueError:
        target_val = None

    patterns = [
        r"FINAL\s*ANSWER\s*[:\-]?\s*\$?\s*([\-]?\d[\d,]*(?:\.\d+)?)",
        r"\\boxed\{\s*([\-]?\d[\d,]*(?:\.\d+)?)\s*\}",
        r"(?:answer\s+is|equals|=)\s*\$?\s*([\-]?\d[\d,]*(?:\.\d+)?)\s*[\.\n]*$",
    ]
    pred_str = None
    for pat in patterns:
        matches = re.findall(pat, clean, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            pred_str = matches[-1].replace(",", "")
            break
    if pred_str is None:
        nums = re.findall(r"[\-]?\d[\d,]*(?:\.\d+)?", clean)
        if nums:
            pred_str = nums[-1].replace(",", "")

    if pred_str is None:
        return False, None
    try:
        if target_val is not None and abs(float(pred_str) - target_val) < 1e-4:
            return True, pred_str
    except ValueError:
        pass
    return (pred_str == target_norm), pred_str


def grade_mcq(output: str, target: str, valid_letters: str = "ABCD") -> Tuple[bool, Optional[str]]:
    clean = strip_thinking_block(output)
    target = target.strip().upper()
    cls_pat = f"[{valid_letters}]"
    patterns = [
        rf"FINAL\s*ANSWER\s*[:\-]?\s*[\(\[]?({cls_pat})[\)\]]?(?![a-zA-Z])",
        rf"\\boxed\{{\s*[\(\[]?({cls_pat})[\)\]]?\s*\}}",
        rf"(?:correct\s+answer|answer)\s*(?:is|:)\s*\**[\(\[]?({cls_pat})[\)\]]?\**(?![a-zA-Z])",
        rf"^\s*[\(\[]?({cls_pat})[\)\]]?\s*$",
    ]
    for pat in patterns:
        matches = re.findall(pat, clean, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            pred = matches[-1].upper()
            return (pred == target), pred
    # Fallback in last 120 chars
    tail = clean[-120:]
    matches = re.findall(rf"[\(\[]({cls_pat})[\)\]]", tail)
    if matches:
        pred = matches[-1].upper()
        return (pred == target), pred
    return False, None


def grade_mbpp(output: str, tests: List[str], timeout: float = 4.0) -> Tuple[bool, str]:
    clean = strip_thinking_block(output)
    fences = re.findall(r"```(?:python)?\s*(.*?)```", clean, flags=re.DOTALL)
    if fences:
        code = fences[-1].strip()
    else:
        m = re.search(r"((?:import\s+\w+.*\n|from\s+\w+.*\n)*def\s+\w+.*)", clean, flags=re.DOTALL)
        code = m.group(1).strip() if m else clean.strip()
    if "def " not in code:
        return False, "no_def"
    script = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        tmp_path = fh.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (proc.returncode == 0), ("ok" if proc.returncode == 0 else proc.stderr[-160:])
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def grade_item(item: Dict[str, Any], output: str) -> Tuple[bool, Any]:
    b = item["benchmark"]
    if b == "gsm8k":
        return grade_gsm8k(output, item["target"])
    elif b in ("arc_challenge", "mmlu_pro"):
        return grade_mcq(output, item["target"], item.get("valid_letters", "ABCD"))
    elif b == "mbpp":
        return grade_mbpp(output, item["tests"])
    raise ValueError(f"Unknown benchmark {b}")


# ---------------------------------------------------------------------------
# Paired Statistical Tests (McNemar Exact + Paired Bootstrap CI)
# ---------------------------------------------------------------------------

def paired_significance(base_correct: List[bool], surg_correct: List[bool], seed: int = 42) -> Dict[str, Any]:
    assert len(base_correct) == len(surg_correct)
    n = len(base_correct)
    b01 = sum((not b) and s for b, s in zip(base_correct, surg_correct))  # surgery fixed baseline error
    b10 = sum(b and (not s) for b, s in zip(base_correct, surg_correct))  # surgery broke baseline success
    b11 = sum(b and s for b, s in zip(base_correct, surg_correct))
    b00 = sum((not b) and (not s) for b, s in zip(base_correct, surg_correct))

    # Exact two-sided McNemar binomial p-value on discordant pairs
    disc = b01 + b10
    if disc == 0:
        p_val = 1.0
    else:
        k = min(b01, b10)
        tail = sum(math.comb(disc, i) for i in range(k + 1)) / (2 ** disc)
        p_val = min(1.0, 2.0 * tail)

    # Paired bootstrap 95% CI on accuracy delta
    rng = random.Random(seed)
    deltas = [int(s) - int(b) for b, s in zip(base_correct, surg_correct)]
    boot = []
    for _ in range(2000):
        samp = [deltas[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(samp) / n)
    boot.sort()
    return {
        "n": n,
        "base_acc": round(sum(base_correct) / max(1, n), 4),
        "surg_acc": round(sum(surg_correct) / max(1, n), 4),
        "delta_acc": round((sum(surg_correct) - sum(base_correct)) / max(1, n), 4),
        "gains_b01": b01,
        "losses_b10": b10,
        "both_correct_b11": b11,
        "both_wrong_b00": b00,
        "mcnemar_exact_p": round(p_val, 5),
        "ci95_delta": [round(boot[int(0.025 * len(boot))], 4), round(boot[int(0.975 * len(boot))], 4)],
    }
