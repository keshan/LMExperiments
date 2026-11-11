"""Summarize 35-layer Phase 1 MRI weight and activation telemetry."""

import json
import sys
from pathlib import Path


def main() -> None:
    report_path = Path(
        sys.argv[1] if len(sys.argv) > 1 else "gemma4_surgery/results/phase1_anatomy_report.json"
    )
    d = json.loads(report_path.read_text())
    w_map = {x["layer"]: x for x in d["layer_weight_anatomy"]}
    a_map = {x["layer"]: x for x in d["activation_mri"]}

    header = (
        "L  | Typ | KVSh | scalar  | q_rms | k_rms | ple_w_rms | ple_sr | "
        "h_in  | attn_u | mlp_u  | ple_post | pre_cv->post_cv | cos"
    )
    print(header)
    print("-" * len(header))
    for l in range(35):
        w = w_map[l]
        a = a_map[l]
        lt = "Sld" if w["attn_is_sliding"] else "Glb"
        sh = "Yes" if w["attn_is_kv_shared_layer"] else "No "
        sc = w["layer_scalar"]
        qr = w["attn_q_norm_rms"]
        kr = w.get("attn_k_norm_rms") or 0.0
        pw = w["post_per_layer_input_norm_rms"]
        psr = w["ple_proj_svd"]["stable_rank"]
        hin = a["h_in_norm"]
        au = a["attn_update_norm"]
        mu = a["mlp_update_norm"]
        pp = a["ple_post_norm_mean"]
        cv_pre = a["ple_pre_norm_cv"]
        cv_post = a["ple_post_norm_cv"]
        cos = a["cos_in_out"]
        print(
            f"{l:2d} | {lt} | {sh}  | {sc:7.4f} | {qr:5.3f} | {kr:5.3f} | "
            f"{pw:9.4f} | {psr:6.2f} | {hin:5.1f} | {au:6.1f} | {mu:6.1f} | "
            f"{pp:8.2f} | {cv_pre:5.3f}->{cv_post:5.3f}  | {cos:5.3f}"
        )


if __name__ == "__main__":
    main()
