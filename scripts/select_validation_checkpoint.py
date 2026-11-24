"""Apply the pre-registered validation rule (<run>/validation_rule.json) and write
<run>/validation/selection.json plus a per-checkpoint CSV.

Usage: select_validation_checkpoint.py <run_dir>
"""
import csv
import json
import sys
from pathlib import Path


def main():
    run = Path(sys.argv[1])
    rule = json.loads((run / "validation_rule.json").read_text())
    n_expected = rule["episodes_per_condition"]
    rows = {}
    for f in sorted((run / "validation").glob("val_*_*.json")):
        d = json.loads(f.read_text())
        step, cond = int(f.stem.split("_")[1]), f.stem.split("_")[2]
        eps = d["episodes"]
        if "wall_seconds" not in d or len(eps) != n_expected or d["args"]["seed"] != rule["eval_seed"]:
            raise SystemExit(f"incomplete or mismatched validation report: {f}")
        r = rows.setdefault(step, {"env_step": step, "checkpoint": d["checkpoint"]})
        r[f"succ_{cond}"] = sum(bool(e["success"]) for e in eps)
        r[f"applied_{cond}"] = sum(any(x["applied"] for x in e.get("perturb_events", [])) for e in eps)
    conds = [f"{'pm'[c['direction'] < 0]}{'xy'[c['axis']]}{c['distance_cm']}" for c in rule["validation_conditions"]]
    for r in rows.values():
        missing = [c for c in conds if f"succ_{c}" not in r]
        if missing:
            raise SystemExit(f"checkpoint {r['env_step']} missing {missing}")
        r["score"] = sum(r[f"succ_{c}"] for c in conds)
        r["episodes"] = n_expected * len(conds)
    table = [rows[k] for k in sorted(rows)]
    best = max(table, key=lambda r: (r["score"], r["env_step"]))  # ties -> later checkpoint
    with open(run / "validation" / "validation_scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    sel = {"rule": rule["selection"], "selected_env_step": best["env_step"], "selected_checkpoint": best["checkpoint"],
           "selected_score": best["score"], "episodes": best["episodes"],
           "final_score": rows[max(rows)]["score"], "n_candidates": len(table)}
    (run / "validation" / "selection.json").write_text(json.dumps(sel, indent=2))
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
