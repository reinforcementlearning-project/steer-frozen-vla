"""Compare residual eval reports with frozen-base reports, paired per episode.

Usage: compare_residual_eval.py --base results/eval50_{d}cm.json --residual <eval.json> [...pairs]
Pairs are matched by (noise_seed, initial_state_hash); any mismatch is an error.
"""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.summarize import wilson


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(base_path, res_path):
    base, res = (json.loads(Path(p).read_text()) for p in (base_path, res_path))
    key = lambda e: (e["noise_seed"], e["initial_state_hash"])
    b_eps = {key(e): e for e in base["episodes"]}
    r_eps = {key(e): e for e in res["episodes"]}
    if set(b_eps) != set(r_eps):
        raise ValueError(f"Unpaired episodes between {base_path} and {res_path}")
    n = len(r_eps)
    bs = sum(bool(e["success"]) for e in b_eps.values())
    rs = sum(bool(e["success"]) for e in r_eps.values())
    gained = sum(bool(r_eps[k]["success"]) and not b_eps[k]["success"] for k in r_eps)
    lost = sum(bool(b_eps[k]["success"]) and not r_eps[k]["success"] for k in r_eps)
    applied = lambda e: any(x["applied"] for x in e.get("perturb_events", []))
    both = [k for k in r_eps if applied(r_eps[k]) and applied(b_eps[k])]
    g2 = sum(bool(r_eps[k]["success"]) and not b_eps[k]["success"] for k in both)
    l2 = sum(bool(b_eps[k]["success"]) and not r_eps[k]["success"] for k in both)

    def perturb_stats(eps):
        ev = [e.get("perturb_events", []) for e in eps.values()]
        steps = [x[0]["step"] for x in ev if x and x[0]["applied"]]
        disp = [x[0].get("actual_displacement_m", 0.) for x in ev if x and x[0]["applied"]]
        rej = {}
        for x in ev:
            if x and not x[0]["applied"]:
                rej[x[0]["rejection"]] = rej.get(x[0]["rejection"], 0) + 1
        return {"applied": len(steps), "rejected": rej, "never_triggered": sum(not x for x in ev),
                "trigger_step_mean": sum(steps) / len(steps) if steps else None,
                "displacement_m_mean": sum(disp) / len(disp) if disp else None}

    a = res["args"]
    return {"distance_cm": a["distance_cm"], "axis": a.get("axis", 1), "direction": a.get("direction", 1),
            "checkpoint": res.get("checkpoint", "frozen_base_zero_residual"),
            "episodes": n, "complete": "wall_seconds" in res,
            "perturbation": {"base": perturb_stats(b_eps), "residual": perturb_stats(r_eps)},
            "base": {"successes": bs, "rate": bs / n, "wilson95": wilson(bs, n)},
            "residual": {"successes": rs, "rate": rs / n, "wilson95": wilson(rs, n)},
            "delta_pp": 100 * (rs - bs) / n, "gained": gained, "lost": lost,
            "mcnemar_exact_p": mcnemar_exact(gained, lost),
            "both_applied": {"episodes": len(both), "gained": g2, "lost": l2,
                             "mcnemar_exact_p": mcnemar_exact(g2, l2)},
            "residual_norm_mean": sum(e.get("residual_norm_mean", 0.) for e in r_eps.values()) / n,
            "mean_env_steps": {"base": sum(e["env_steps"] for e in b_eps.values()) / n,
                               "residual": sum(e["env_steps"] for e in r_eps.values()) / n}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pair", nargs=2, action="append", metavar=("BASE", "RESIDUAL"), required=True)
    p.add_argument("--out")
    args = p.parse_args()
    rows = [compare(b, r) for b, r in args.pair]
    text = json.dumps(rows, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
