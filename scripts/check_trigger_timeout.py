"""Read-only: perturbation trigger timing and 280-step horizon censoring from saved eval reports."""
import json, glob, csv, numpy as np
from pathlib import Path
E = Path("results/residual_robustness/evals"); O = Path("results/residual_robustness/checks"); O.mkdir(exist_ok=True)
CN = {"0cm": "0cm", "py2": "+y 2cm", "my2": "-y 2cm", "px2": "+x 2cm", "mx2": "-x 2cm",
      "py4": "+y 4cm", "my4": "-y 4cm", "px4": "+x 4cm", "mx4": "-x 4cm"}
POL = ["base", "s42_val", "s43_val", "s44_val", "s42_final", "s43_final", "s44_final"]
trig, hor = [], []
for p in POL:
    for c, cname in CN.items():
        f = E / f"{p}_{c}.json"
        if not f.exists(): continue
        eps = json.load(open(f))["episodes"]
        ev = [e["perturb_events"][0] for e in eps if e["perturb_events"]]
        st = np.array([v["step"] for v in ev]); dist = np.array([v["eef_distance"] for v in ev])
        trig.append({"policy": p, "condition": cname, "n": len(ev), "step_min": st.min(), "step_p50": np.median(st),
                     "step_max": st.max(), "step_sd": round(st.std(), 2),
                     "chunk_phase_hist(step%10)": np.bincount(st % 10, minlength=10).tolist(),
                     "eef_dist_min": round(dist.min(), 4), "eef_dist_max": round(dist.max(), 4),
                     "frac_dist_ge_0.17": round(float((dist >= 0.17).mean()), 3)})
        L = np.array([e["env_steps"] for e in eps]); S = np.array([bool(e["success"]) for e in eps])
        fail = ~S
        row = {"policy": p, "condition": cname, "success_280": S.mean(), "failures": int(fail.sum()),
               "failures_at_limit": int((fail & (L >= 280)).sum()),
               "success_len_p50": float(np.median(L[S])) if S.any() else None,
               "success_len_p90": float(np.percentile(L[S], 90)) if S.any() else None,
               "success_len_max": int(L[S].max()) if S.any() else None}
        for T in (160, 200, 240, 260):
            row[f"success_by_{T}"] = float((S & (L <= T)).mean())
        row["successes_in_241_280"] = int((S & (L > 240)).sum())
        hor.append(row)
for name, rows in (("trigger_timing.csv", trig), ("horizon_censoring.csv", hor)):
    with open(O / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
# base EEF-object distance at step 10 is not logged; report the per-episode trigger step spread instead
print("trigger"); [print(r) for r in trig if r["policy"] in ("base", "s42_val") and r["condition"] in ("+y 2cm", "+y 4cm")]
print("horizon")
for r in hor:
    if r["policy"] in ("base", "s42_val", "s43_val", "s44_val"):
        print(r["policy"], r["condition"], {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("policy", "condition")})
