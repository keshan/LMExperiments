"""Summarize Exp 4 Held-Out Test Evaluation (`N = 384`) with statistical tables and qualitative traces."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.eval_harness import paired_significance


def main() -> None:
    p = Path("gemma4_surgery/results/exp4_heldout_test_results.json")
    if not p.exists():
        print(f"Missing {p}")
        return
    data = json.loads(p.read_text())
    base = data.get("baseline", {})
    print("=== BASELINE (00_baseline) ===")
    print(
        f"Overall: {base.get('overall_acc', 0)*100:.2f}% "
        f"({base.get('total_correct')}/{base.get('total_items')}) | "
        f"Reasoning (3-bench): {base.get('reasoning_3bench_acc', 0)*100:.2f}% "
        f"({base.get('reasoning_3bench_correct')}/{base.get('reasoning_3bench_n')}) | "
        f"Speed: {base.get('overall_tok_per_sec')} tok/s ({base.get('total_elapsed_sec')}s)"
    )
    for bname, binfo in base.get("benchmarks", {}).items():
        print(f"  - {bname}: {binfo['acc']*100:.2f}% ({binfo['num_correct']}/{binfo['n']}) @ {binfo['tok_per_sec']} tok/s")

    print("\n=== HELD-OUT TEST FINALISTS ===")
    for mname, m in data.get("models", {}).items():
        ov = m.get("overall_vs_baseline", {})
        rv = m.get("reasoning_3bench_vs_baseline", {})
        ci_ov = ov.get("ci95_delta", [0.0, 0.0])
        ci_rv = rv.get("ci95_delta", [0.0, 0.0])

        # Also compute GSM8K + ARC-Challenge (N=192) and GSM8K + ARC-C + MBPP (N=288)
        b_gsm_arc = base["benchmarks"]["gsm8k"]["correct_flags"] + base["benchmarks"]["arc_challenge"]["correct_flags"]
        s_gsm_arc = m["benchmarks"]["gsm8k"]["correct_flags"] + m["benchmarks"]["arc_challenge"]["correct_flags"]
        ga_sig = paired_significance(b_gsm_arc, s_gsm_arc)

        b_3 = b_gsm_arc + base["benchmarks"]["mbpp"]["correct_flags"]
        s_3 = s_gsm_arc + m["benchmarks"]["mbpp"]["correct_flags"]
        g3_sig = paired_significance(b_3, s_3)

        print(f"\n[{mname}]")
        print(
            f"  Overall (N=384): {m['overall_acc']*100:.2f}% ({m['total_correct']}/{m['total_items']}) "
            f"| delta={ov.get('delta_acc', 0)*100:+.2f}% (+{ov.get('gains_b01')}/-{ov.get('losses_b10')}) "
            f"| McNemar p={ov.get('mcnemar_exact_p')} "
            f"| 95% CI [{ci_ov[0]*100:+.2f}%, {ci_ov[1]*100:+.2f}%]"
        )
        print(
            f"  Reasoning 3-Bench (N=288): {m['reasoning_3bench_acc']*100:.2f}% ({m['reasoning_3bench_correct']}/{m['reasoning_3bench_n']}) "
            f"| delta={rv.get('delta_acc', 0)*100:+.2f}% (+{rv.get('gains_b01')}/-{rv.get('losses_b10')}) "
            f"| McNemar p={rv.get('mcnemar_exact_p')} "
            f"| 95% CI [{ci_rv[0]*100:+.2f}%, {ci_rv[1]*100:+.2f}%]"
        )
        print(
            f"  GSM8K + ARC-C (N=192): {ga_sig['surg_acc']*100:.2f}% vs {ga_sig['base_acc']*100:.2f}% "
            f"| delta={ga_sig['delta_acc']*100:+.2f}% (+{ga_sig['gains_b01']}/-{ga_sig['losses_b10']}) "
            f"| McNemar p={ga_sig['mcnemar_exact_p']} "
            f"| 95% CI [{ga_sig['ci95_delta'][0]*100:+.2f}%, {ga_sig['ci95_delta'][1]*100:+.2f}%]"
        )
        print(
            f"  GSM8K + ARC-C + MBPP (N=288): {g3_sig['surg_acc']*100:.2f}% vs {g3_sig['base_acc']*100:.2f}% "
            f"| delta={g3_sig['delta_acc']*100:+.2f}% (+{g3_sig['gains_b01']}/-{g3_sig['losses_b10']}) "
            f"| McNemar p={g3_sig['mcnemar_exact_p']} "
            f"| 95% CI [{g3_sig['ci95_delta'][0]*100:+.2f}%, {g3_sig['ci95_delta'][1]*100:+.2f}%]"
        )
        print(f"  Speed: {m['overall_tok_per_sec']} tok/s ({m['total_elapsed_sec']}s)")
        for bname, binfo in m.get("benchmarks", {}).items():
            vb = binfo.get("vs_baseline", {})
            bci = vb.get("ci95_delta", [0.0, 0.0])
            print(
                f"    * {bname}: {binfo['acc']*100:.2f}% ({binfo['num_correct']}/{binfo['n']}) "
                f"| delta={vb.get('delta_acc', 0)*100:+.2f}% (+{vb.get('gains_b01')}/-{vb.get('losses_b10')}, "
                f"p={vb.get('mcnemar_exact_p')}, CI=[{bci[0]*100:+.2f}%, {bci[1]*100:+.2f}%]) "
                f"| {binfo['tok_per_sec']} tok/s"
            )
        wins = m.get("qualitative_wins", [])
        losses = m.get("qualitative_losses", [])
        print(f"  Captured qualitative wins: {len(wins)} | losses: {len(losses)}")
        for w in wins[:5]:
            print(f"    [WIN {w['id']}] target={w['target']!r}")
            print(f"      BASE: {w['baseline_output'][:240]!r}")
            print(f"      SURG: {w['surgery_output'][:240]!r}")
        for l in losses[:2]:
            print(f"    [LOSS {l['id']}] target={l['target']!r}")
            print(f"      BASE: {l['baseline_output'][:240]!r}")
            print(f"      SURG: {l['surgery_output'][:240]!r}")

    # Pooled Zero-Data Evaluation (N = 128 dev + 384 test = 512 total items)
    p_dev3 = Path("gemma4_surgery/results/exp3_dev_screen.json")
    p_dev3b = Path("gemma4_surgery/results/exp3b_dev_refinement.json")
    if p_dev3.exists() and p_dev3b.exists():
        d3 = json.loads(p_dev3.read_text())
        d3b = json.loads(p_dev3b.read_text())
        dev_base = d3["baseline"]["benchmarks"]
        d3b_map = d3b["candidates"]
        print("\n=== POOLED ZERO-DATA EVALUATION (N = 512: 128 Dev + 384 Test) ===")
        for label, dev_m, test_m in [
            ("S1d_prose_orth_proj_25_34", d3b_map["S1d_prose_orth_proj_25_34"]["benchmarks"], data["models"]["S1d_prose_orth_proj_25_34"]["benchmarks"]),
            ("S1c_ple_stop_subspace_clean", d3["candidates"]["S1c_ple_stop_subspace_clean"]["benchmarks"], data["models"]["S1c_ple_stop_subspace_clean"]["benchmarks"]),
            ("S6b_bypass_L31_comp_plus_prose_orth", d3b_map["S6b_bypass_L31_comp_plus_prose_orth"]["benchmarks"], data["models"]["S6b_bypass_L31_comp_plus_prose_orth"]["benchmarks"]),
        ]:
            all_b, all_s = [], []
            reas_b, reas_s = [], []
            ga_b, ga_s = [], []
            print(f"\n[POOLED N=512 | {label}]")
            for bname in ["gsm8k", "arc_challenge", "mmlu_pro", "mbpp"]:
                fb = dev_base[bname]["correct_flags"] + base["benchmarks"][bname]["correct_flags"]
                fs = dev_m[bname]["correct_flags"] + test_m[bname]["correct_flags"]
                sig = paired_significance(fb, fs)
                print(
                    f"  * {bname} (N={len(fb)}): {sig['surg_acc']*100:.2f}% vs {sig['base_acc']*100:.2f}% "
                    f"| delta={sig['delta_acc']*100:+.2f}% (+{sig['gains_b01']}/-{sig['losses_b10']}, "
                    f"p={sig['mcnemar_exact_p']}, CI=[{sig['ci95_delta'][0]*100:+.2f}%, {sig['ci95_delta'][1]*100:+.2f}%])"
                )
                all_b.extend(fb)
                all_s.extend(fs)
                if bname in ("gsm8k", "arc_challenge", "mmlu_pro"):
                    reas_b.extend(fb)
                    reas_s.extend(fs)
                if bname in ("gsm8k", "arc_challenge"):
                    ga_b.extend(fb)
                    ga_s.extend(fs)
            ov_sig = paired_significance(all_b, all_s)
            rv_sig = paired_significance(reas_b, reas_s)
            ga_sig = paired_significance(ga_b, ga_s)
            print(
                f"  => Overall (N=512): {ov_sig['surg_acc']*100:.2f}% vs {ov_sig['base_acc']*100:.2f}% "
                f"| delta={ov_sig['delta_acc']*100:+.2f}% (+{ov_sig['gains_b01']}/-{ov_sig['losses_b10']}) "
                f"| McNemar p={ov_sig['mcnemar_exact_p']} | 95% CI [{ov_sig['ci95_delta'][0]*100:+.2f}%, {ov_sig['ci95_delta'][1]*100:+.2f}%]"
            )
            print(
                f"  => Reasoning 3-Bench (N=384): {rv_sig['surg_acc']*100:.2f}% vs {rv_sig['base_acc']*100:.2f}% "
                f"| delta={rv_sig['delta_acc']*100:+.2f}% (+{rv_sig['gains_b01']}/-{rv_sig['losses_b10']}) "
                f"| McNemar p={rv_sig['mcnemar_exact_p']} | 95% CI [{rv_sig['ci95_delta'][0]*100:+.2f}%, {rv_sig['ci95_delta'][1]*100:+.2f}%]"
            )
            print(
                f"  => GSM8K + ARC-C (N=256): {ga_sig['surg_acc']*100:.2f}% vs {ga_sig['base_acc']*100:.2f}% "
                f"| delta={ga_sig['delta_acc']*100:+.2f}% (+{ga_sig['gains_b01']}/-{ga_sig['losses_b10']}) "
                f"| McNemar p={ga_sig['mcnemar_exact_p']} | 95% CI [{ga_sig['ci95_delta'][0]*100:+.2f}%, {ga_sig['ci95_delta'][1]*100:+.2f}%]"
            )


if __name__ == "__main__":
    main()
