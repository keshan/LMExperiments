"""Print detailed statistical breakdown of Exp 3 dev-split weight surgery screening."""

import json
import sys
from pathlib import Path


def main() -> None:
    p = Path(sys.argv[1] if len(sys.argv) > 1 else "gemma4_surgery/results/exp3_dev_screen.json")
    d = json.loads(p.read_text())
    base = d["baseline"]
    print(
        f"BASELINE: overall={base['overall_acc']*100:.2f}% ({base['total_correct']}/{base['total_items']}) | "
        f"toks={sum(b['total_tokens'] for b in base['benchmarks'].values())} | speed={base['overall_tok_per_sec']} tok/s"
    )
    for bname, bm in base["benchmarks"].items():
        print(f"  - {bname:14s}: {bm['acc']*100:5.1f}% ({bm['num_correct']}/{bm['n']}) | toks={bm['total_tokens']}")
    print("=" * 110)

    for cname, cr in d["candidates"].items():
        ov = cr["overall_vs_baseline"]
        tot_toks = sum(b["total_tokens"] for b in cr["benchmarks"].values())
        print(
            f"{cname:34s} | acc={cr['overall_acc']*100:5.2f}% (delta={ov['delta_acc']*100:+5.2f}%, "
            f"+{ov['gains_b01']}/-{ov['losses_b10']}, p={ov['mcnemar_exact_p']:.4f}, CI={ov['ci95_delta']}) | "
            f"toks={tot_toks:5d} | {cr['overall_tok_per_sec']:5.1f} tok/s"
        )
        sub_parts = []
        for bname, bm in cr["benchmarks"].items():
            vb = bm["vs_baseline"]
            sub_parts.append(
                f"{bname}:{bm['acc']*100:4.1f}%({vb['delta_acc']*100:+4.1f}%,+{vb['gains_b01']}/-{vb['losses_b10']},tok={bm['total_tokens']})"
            )
        print("   " + " | ".join(sub_parts))


if __name__ == "__main__":
    main()
