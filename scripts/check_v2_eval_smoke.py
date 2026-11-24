"""Smoke checks for Residual-SAC evaluation through the Dynamic LIBERO v2 adapter.

For every report in a directory: residual dimension and executed scale, VLA chunk length,
v2 config, per-episode info["dynamic_libero"] status, 6-D residual output (7th dim 0).
Runs named *_legacy_* are compared episode by episode (outcome, length, perturbation
event, per-step executed actions) with the earlier v1 evaluation of the same policy.

  python scripts/check_v2_eval_smoke.py results/residual_v2_generalization/smoke \
      --v1-dir ../steer-frozen-vla/results/residual_robustness/evals
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V1_NAME = {"legacy_py2": "py2", "legacy_my2": "my2", "legacy_px2": "px2", "legacy_mx2": "mx2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--v1-dir", required=True)
    args = ap.parse_args()
    d, v1 = ROOT / args.dir, Path(args.v1_dir)
    rows = []
    for f in sorted(d.glob("*.json")):
        rep = json.loads(f.read_text())
        if "wall_seconds" not in rep:
            print(f"WARNING incomplete {f.name}")
            continue
        cfg = rep["dynamic_libero_v2_config"]
        eps = rep["episodes"]
        st = [e.get("dynamic_libero") or {} for e in eps]
        steps = np.load(str(f) + ".steps.npz")
        u = steps["u"]
        row = {"run": f.stem, "config": cfg["name"], "split": cfg["split"],
               "window_m": cfg["perturbation"]["window"], "max_steps": cfg["max_steps"],
               "residual_dims": rep["residual_dims"], "executed_scale": rep["executed_residual_scale"],
               "n_action_steps": rep["n_action_steps"], "checkpoint_sha256": rep["checkpoint_sha256"],
               "episodes": len(eps), "status": [s.get("status") for s in st],
               "trigger_step": [s.get("step") for s in st],
               "eef_distance": [round(s["eef_distance"], 4) if s.get("eef_distance") is not None else None for s in st],
               "dynamic_libero_keys_ok": all({"config", "split", "status", "choice", "episode_seed"} <= set(s) for s in st),
               "u_dim7_all_zero": bool(np.all(u[:, 6] == 0)),
               "u_dims_1_6_nonzero_frac": float(np.mean(np.any(u[:, :6] != 0, axis=1))),
               "u_abs_max": float(np.abs(u).max()), "success": [bool(e["success"]) for e in eps]}
        key = next((k for k in V1_NAME if f.stem.endswith(k)), None)
        if key:
            pol = f.stem[: -len(key) - 1]
            ref_f = v1 / f"{pol}_{V1_NAME[key]}.json"
            ref = json.loads(ref_f.read_text())
            same = {"success": 0, "env_steps": 0, "initial_state_hash": 0, "event": 0}
            for a, b in zip(ref["episodes"], eps):
                same["success"] += a["success"] == b["success"]
                same["env_steps"] += a["env_steps"] == b["env_steps"]
                same["initial_state_hash"] += a["initial_state_hash"] == b["initial_state_hash"]
                ea, eb = a["perturb_events"][0], b["perturb_events"][0]
                same["event"] += all(ea[k] == eb[k] for k in ("step", "before", "after", "applied", "eef_distance"))
            rs = np.load(str(ref_f) + ".steps.npz")
            m = rs["episode"] < len(eps)
            n = int(m.sum())
            exec_diff = float(np.abs(rs["exec_action"][m] - steps["exec_action"][:n]).max()) if n == len(steps["episode"]) else float("inf")
            row["v1_reproduction"] = {"reference": str(ref_f), **{k: f"{v}/{len(eps)}" for k, v in same.items()},
                                      "steps_compared": n, "max_abs_diff_exec_action": exec_diff,
                                      "max_abs_diff_residual": float(np.abs(rs["u"][m] - u[:n]).max()) if n == len(u) else float("inf")}
        rows.append(row)
        print(json.dumps(row))
    (d / "smoke_checks.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
