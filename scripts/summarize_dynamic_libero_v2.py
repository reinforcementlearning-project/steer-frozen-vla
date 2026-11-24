"""Summarize Dynamic LIBERO v2 validation rollouts (scripts/run_dynamic_libero_v2.py outputs).

Per run: trigger status/reason counts (applied, rejected, not triggered), trigger step,
EEF distance and chunk phase at the trigger, whether the actual displacement matches the
sampled direction and size, VLA re-planning cadence, spatial shift statistics, success
rate (Wilson 95% CI). With a no-perturbation run that logged per-step distances
(--log-distance) it also estimates, offline, how often each timing window / chunk phase
is reachable on the frozen policy's trajectories. With --legacy-ref it checks that a
v2 legacy run reproduces an earlier v1 evaluation episode by episode.

  python scripts/summarize_dynamic_libero_v2.py results/dynamic_libero_v2/validation \
      --legacy-ref legacy_primary_py2=<v1 eval json> --legacy-ref legacy_4cm_mx4=<v1 eval json>
"""
import argparse
import csv
import json
from collections import Counter
from math import sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {"legacy": (0.05, 0.18), "early": (0.15, 0.18), "middle": (0.10, 0.15), "late": (0.05, 0.10)}
DIRS = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def summarize_run(path):
    rep = json.loads(path.read_text())
    eps = rep["episodes"]
    n = len(eps)
    st = [e["dynamic_libero"] for e in eps]
    status = Counter(s.get("status") for s in st)
    reasons = Counter(f'{s.get("status")}:{s.get("reason")}' for s in st if s.get("status") != "applied")
    ev = [e["perturb_events"][0] for e in eps if e["perturb_events"]]
    applied = [x for x in ev if x["applied"]]
    disp_ok = 0
    for x in applied:
        axis, sign = DIRS[x["direction"]]
        exp = np.zeros(3)
        exp[axis] = sign * x["distance_m"]
        disp_ok += bool(np.allclose(x["actual_displacement_vector_m"], exp, atol=1e-6))
    steps = np.array([x["step"] for x in ev]) if ev else np.array([])
    dist = np.array([x["eef_distance"] for x in ev]) if ev else np.array([])
    succ = sum(bool(e["success"]) for e in eps)
    cfg = rep["config"]
    pert = cfg.get("perturbation") or {}
    row = {"run": path.stem, "config": cfg["name"], "split": cfg["split"],
           "timing": pert.get("timing"), "window_m": pert.get("window"),
           "chunk_phase": pert.get("chunk_phase"), "spatial": cfg["spatial"]["mode"], "max_steps": cfg["max_steps"],
           "episodes": n, "applied": status.get("applied", 0), "rejected": status.get("rejected", 0),
           "not_triggered": status.get("not_triggered", 0), "disabled": status.get("disabled", 0),
           "applied_rate": round(status.get("applied", 0) / n, 4) if n else None,
           "non_applied_reasons": dict(reasons),
           "displacement_matches_config": f"{disp_ok}/{len(applied)}",
           "trigger_step_min": int(steps.min()) if len(steps) else None,
           "trigger_step_median": float(np.median(steps)) if len(steps) else None,
           "trigger_step_max": int(steps.max()) if len(steps) else None,
           "eef_distance_min": round(float(dist.min()), 4) if len(dist) else None,
           "eef_distance_max": round(float(dist.max()), 4) if len(dist) else None,
           "chunk_phase_hist": np.bincount([x["chunk_phase"] for x in ev], minlength=10).tolist() if ev else None,
           "directions": dict(Counter(s["choice"]["direction"] for s in st if s.get("choice"))),
           "replan_cadence_ok": f'{sum(e["replan_cadence_ok"] for e in eps)}/{n}',
           "successes": succ, "success_rate": round(succ / n, 4) if n else None,
           "success_wilson95": wilson(succ, n)}
    sp = [s["spatial"] for s in st if s["spatial"].get("applied")]
    if sp:
        off = np.array([x["offset_m"] for x in sp])
        rej = Counter()
        for x in sp:
            rej.update(x["rejections"])
        row.update(spatial_applied=f"{len(sp)}/{n}", spatial_offset_abs_mean_cm=np.round(np.abs(off).mean(0) * 100, 2).tolist(),
                   spatial_offset_range_cm=[np.round(off.min(0) * 100, 2).tolist(), np.round(off.max(0) * 100, 2).tolist()],
                   spatial_attempts_mean=round(float(np.mean([x["attempts"] for x in sp])), 2),
                   spatial_rejections=dict(rej),
                   spatial_unique_offsets=len({tuple(np.round(o, 6)) for o in off}))
    return row, rep


def feasibility(trace_path, max_lift=0.015, earliest=10, n=10):
    t = np.load(trace_path)
    rows = []
    eps = np.unique(t["episode"])
    for name, (lo, hi) in WINDOWS.items():
        for phase in (None, 0, 1, 5):
            reach, first, dists = 0, [], []
            for e in eps:
                m = t["episode"] == e
                step, d, c, dz = t["step"][m], t["eef_distance"][m], t["robot_contact"][m], t["object_dz"][m]
                ok = (step >= earliest) & (d >= lo) & (d <= hi) & (~c.astype(bool)) & (np.abs(dz) < max_lift)
                if phase is not None:
                    ok &= (step % n) == phase
                if ok.any():
                    reach += 1
                    i = int(np.argmax(ok))
                    first.append(int(step[i]))
                    dists.append(float(d[i]))
            rows.append({"window": name, "window_m": [lo, hi], "chunk_phase": phase, "episodes": len(eps),
                         "reachable_episodes": reach, "reachable_rate": round(reach / len(eps), 4),
                         "first_step_median": float(np.median(first)) if first else None,
                         "first_step_range": [min(first), max(first)] if first else None,
                         "distance_at_first_median": round(float(np.median(dists)), 4) if dists else None})
    per_ep = []
    for e in eps:
        m = t["episode"] == e
        d, c, dz, step = t["eef_distance"][m], t["robot_contact"][m].astype(bool), t["object_dz"][m], t["step"][m]
        first_contact = int(step[np.argmax(c)]) if c.any() else None
        lift = np.abs(dz) >= max_lift
        per_ep.append({"min_distance": float(d.min()), "first_contact_step": first_contact,
                       "first_lift_step": int(step[np.argmax(lift)]) if lift.any() else None,
                       "distance_at_first_contact": float(d[np.argmax(c)]) if c.any() else None})
    pe = {"min_distance_median": round(float(np.median([x["min_distance"] for x in per_ep])), 4),
          "first_contact_step_median": float(np.median([x["first_contact_step"] for x in per_ep if x["first_contact_step"]])),
          "distance_at_first_contact_median": round(float(np.median([x["distance_at_first_contact"] for x in per_ep
                                                                      if x["distance_at_first_contact"] is not None])), 4),
          "episodes_with_contact": sum(x["first_contact_step"] is not None for x in per_ep)}
    return rows, pe


def legacy_check(v2_rep, ref_path):
    ref = json.loads(Path(ref_path).read_text())
    a, b = ref["episodes"], v2_rep["episodes"]
    same = Counter()
    for x, y in zip(a, b):
        same["initial_state_hash"] += x["initial_state_hash"] == y["initial_state_hash"]
        same["success"] += x["success"] == y["success"]
        same["env_steps"] += x["env_steps"] == y["env_steps"]
        ex, ey = x["perturb_events"][0], y["perturb_events"][0]
        for k in ("step", "before", "after", "applied"):
            same[f"event_{k}"] += ex[k] == ey[k]
        same["event_eef_distance"] += abs(ex["eef_distance"] - ey["eef_distance"]) < 1e-12
    return {"reference": str(ref_path), "episodes": min(len(a), len(b)), **dict(same)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--legacy-ref", action="append", default=[], help="run_stem=path/to/v1_eval.json")
    args = ap.parse_args()
    d = ROOT / args.dir
    refs = dict(r.split("=", 1) for r in args.legacy_ref)
    rows, legacy, out = [], [], {}
    for path in sorted(d.glob("*.json")):
        rep = json.loads(path.read_text())
        if "wall_seconds" not in rep:
            print(f"WARNING incomplete {path.name}")
            continue
        row, rep = summarize_run(path)
        rows.append(row)
        if path.stem in refs:
            legacy.append({"run": path.stem, **legacy_check(rep, refs[path.stem])})
        trace = Path(str(path) + ".trace.npz")
        if trace.exists():
            feas, pe = feasibility(trace)
            out["feasibility"] = feas
            out["frozen_trajectory"] = pe
    out["runs"] = rows
    out["legacy_reproduction"] = legacy
    (d / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    with open(d / "summary_runs.csv", "w", newline="") as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    if "feasibility" in out:
        with open(d / "summary_feasibility.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out["feasibility"][0]))
            w.writeheader()
            w.writerows(out["feasibility"])
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
