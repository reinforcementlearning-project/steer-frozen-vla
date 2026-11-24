"""Diagnostic: does the 280-step horizon censor +-y 4cm results?

Compares the official 280-step evaluations (results/residual_robustness/evals) with
re-runs of the same policies, initial states and noise seeds at a 400-step horizon
(results/residual_robustness/evals_h400). Read-only with respect to both.

1. Prefix check: the 400-step run must reproduce the 280-step run up to step 280
   (same initial state, perturbation step, success/length for episodes that ended
   by 280, no success by 280 for episodes that failed, identical per-step actions).
2. Cumulative success at T = 240/280/320/360/400 (Wilson 95% CI) and the
   distribution of success times.
3. Residual - Base paired differences at each T (exact McNemar, paired bootstrap CI).

The 400-step numbers are diagnostic only; the 280-step results stay the official metric.
"""
import argparse
import csv
import json
from math import comb, sqrt
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E280 = ROOT / "results/residual_robustness/evals"
E400 = ROOT / "results/residual_robustness/evals_h400"
OUT = ROOT / "results/residual_robustness/horizon400"
POLICIES = ["base", "s42_val", "s43_val", "s44_val"]
CONDS = {"py4": "+y 4cm", "my4": "-y 4cm"}
TS = [240, 280, 320, 360, 400]
BINS = [0, 160, 200, 240, 280, 320, 360, 400]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load(d, p, c):
    f = d / f"{p}_{c}.json"
    if not f.exists():
        return None, None
    rep = json.loads(f.read_text())
    if "wall_seconds" not in rep:
        print(f"WARNING incomplete {f}")
        return None, None
    npz = Path(str(f) + ".steps.npz")
    return rep, (np.load(npz) if npz.exists() else None)


def success_time(rep):
    """Episode length if successful, else inf."""
    return np.array([e["env_steps"] if e["success"] else np.inf for e in rep["episodes"]])


def prefix_check(p, c, r280, s280, r400, s400):
    e280, e400 = r280["episodes"], r400["episodes"]
    row = {"policy": p, "condition": CONDS[c], "episodes": len(e400),
           "max_steps_280": r280["args"]["max_steps"], "max_steps_400": r400["args"]["max_steps"],
           "same_checkpoint_sha256": r280["checkpoint_sha256"] == r400["checkpoint_sha256"]}
    same_init = same_trig = same_outcome = 0
    for a, b in zip(e280, e400):
        same_init += a["initial_state_hash"] == b["initial_state_hash"] and a["noise_seed"] == b["noise_seed"]
        ta = a["perturb_events"][0]["step"] if a["perturb_events"] else -1
        tb = b["perturb_events"][0]["step"] if b["perturb_events"] else -1
        same_trig += ta == tb
        if a["success"]:
            same_outcome += bool(b["success"]) and b["env_steps"] == a["env_steps"]
        else:  # failed within 280 -> must not succeed by 280 in the longer run
            same_outcome += b["env_steps"] > 280 and not (b["success"] and b["env_steps"] <= 280)
    row.update(same_initial_state_and_seed=same_init, same_trigger_step=same_trig,
               same_outcome_up_to_280=same_outcome)
    if s280 is not None and s400 is not None:
        dmax_exec = dmax_u = 0.0
        compared = 0
        for ep in range(len(e280)):
            m280 = s280["episode"] == ep
            m400 = (s400["episode"] == ep) & (s400["step"] < 280)
            n = int(min(m280.sum(), m400.sum()))
            if int(m280.sum()) != int(m400.sum()):
                dmax_exec = np.inf
            compared += n
            dmax_exec = max(dmax_exec, float(np.abs(s280["exec_action"][m280][:n] - s400["exec_action"][m400][:n]).max()))
            dmax_u = max(dmax_u, float(np.abs(s280["u"][m280][:n] - s400["u"][m400][:n]).max()))
        row.update(steps_compared=compared, max_abs_diff_exec_action=dmax_exec, max_abs_diff_residual=dmax_u)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    checks, cum, dist, diffs, times = [], [], [], [], {}
    for c in CONDS:
        for p in POLICIES:
            r280, s280 = load(E280, p, c)
            r400, s400 = load(E400, p, c)
            if r280 is None or r400 is None:
                continue
            checks.append(prefix_check(p, c, r280, s280, r400, s400))
            t = success_time(r400)
            times[(p, c)] = t
            n = len(t)
            for T in TS:
                k = int((t <= T).sum())
                lo, hi = wilson(k, n)
                cum.append({"policy": p, "condition": CONDS[c], "T": T, "episodes": n, "successes": k,
                            "success_rate": round(k / n, 4), "wilson95_lo": lo, "wilson95_hi": hi})
            s = t[np.isfinite(t)]
            hist = np.histogram(s, bins=BINS)[0].tolist()
            dist.append({"policy": p, "condition": CONDS[c], "successes_by_400": len(s),
                         "success_step_p25": float(np.percentile(s, 25)) if len(s) else None,
                         "success_step_median": float(np.median(s)) if len(s) else None,
                         "success_step_p75": float(np.percentile(s, 75)) if len(s) else None,
                         "success_step_max": float(s.max()) if len(s) else None,
                         **{f"succ_{a + 1}-{b}": h for a, b, h in zip(BINS[:-1], BINS[1:], hist)},
                         "successes_281_400": int(((s > 280) & (s <= 400)).sum())})
    # paired residual - base, per seed and condition, plus both conditions pooled per seed
    groups = [(p, [c]) for p in POLICIES[1:] for c in CONDS] + [(p, list(CONDS)) for p in POLICIES[1:]]
    for p, cs in groups:
        if not all((p, c) in times and ("base", c) in times for c in cs):
            continue
        for T in TS:
            res = np.concatenate([times[(p, c)] <= T for c in cs]).astype(float)
            bas = np.concatenate([times[("base", c)] <= T for c in cs]).astype(float)
            d = res - bas
            b, cc = int((d > 0).sum()), int((d < 0).sum())
            # bootstrap over episode indices (pooled conditions resample each condition's episodes jointly)
            n_ep = len(times[("base", cs[0])])
            dm = d.reshape(len(cs), n_ep).mean(0)
            boots = dm[rng.integers(0, n_ep, size=(args.boot, n_ep))].mean(1)
            diffs.append({"policy": p, "conditions": "/".join(CONDS[c] for c in cs), "T": T,
                          "base_rate": round(bas.mean(), 4), "policy_rate": round(res.mean(), 4),
                          "diff": round(d.mean(), 4), "boot95_lo": round(float(np.percentile(boots, 2.5)), 4),
                          "boot95_hi": round(float(np.percentile(boots, 97.5)), 4),
                          "gained": b, "lost": cc, "mcnemar_exact_p": round(mcnemar_exact(b, cc), 5)})
    # seed mean of the pooled difference
    for T in TS:
        v = [r["diff"] for r in diffs if r["T"] == T and r["conditions"] == "+y 4cm/-y 4cm"]
        if len(v) == 3:
            diffs.append({"policy": "seed_mean(42/43/44)", "conditions": "+y 4cm/-y 4cm", "T": T,
                          "diff": round(float(np.mean(v)), 4), "boot95_lo": "", "boot95_hi": "",
                          "base_rate": "", "policy_rate": "", "gained": "", "lost": "",
                          "mcnemar_exact_p": f"seed range {min(v):+.3f}..{max(v):+.3f}"})
    for name, rows in (("H_prefix_check.csv", checks), ("H_cumulative_success.csv", cum),
                       ("H_success_time_distribution.csv", dist), ("H_residual_minus_base.csv", diffs)):
        if rows:
            with open(OUT / name, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    plot(times)
    print(f"horizon400: {len(checks)} runs checked")
    for r in checks:
        print(r)


def plot(times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    colors = {"base": "black", "s42_val": "tab:blue", "s43_val": "tab:orange", "s44_val": "tab:green"}
    grid = np.arange(1, 401)
    for ax, c in zip(axes, CONDS):
        for p in POLICIES:
            if (p, c) not in times:
                continue
            t = times[(p, c)]
            ax.step(grid, [(t <= T).mean() for T in grid], where="post", color=colors[p],
                    lw=2.2 if p == "base" else 1.4, label="Frozen Base" if p == "base" else p)
        ax.axvline(280, color="red", ls="--", lw=1)
        ax.text(283, 0.02, "official horizon 280", color="red", fontsize=8)
        ax.set_title(f"{CONDS[c]}: cumulative success (50 episodes, diagnostic 400-step run)")
        ax.set_xlabel("env step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("success rate by step")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    fig.savefig(OUT / "figures/H_cumulative_success.png", dpi=130)


if __name__ == "__main__":
    main()
