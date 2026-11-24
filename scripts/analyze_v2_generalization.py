"""Residual-SAC generalization to Dynamic LIBERO v2 timing windows (Early/Middle/Late).

Inputs (read-only):
  Early  = legacy 18 cm rule, the official v1 evaluations of the same checkpoints
           (default ../steer-frozen-vla/results/residual_robustness/evals/<policy>_<cond>.json)
  Middle / Late (and 4 cm variants) = v2 evaluations in results/residual_v2_generalization/evals
           named <policy>_<timing>_<cond>.json, ablations <policy>_<timing>_<cond>_postoff.json
Every rate is over all 50 episodes; perturbation applied / rejected / not-triggered rates are
reported next to it (never success among applied episodes only).

Outputs: results/residual_v2_generalization/analysis/*.csv, *.json, figures/*.png
"""
import argparse
import csv
import json
from math import comb, sqrt
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "results/residual_v2_generalization/evals"
OUT = ROOT / "results/residual_v2_generalization/analysis"
POLICIES = ["base", "s42_val", "s43_val", "s44_val"]
RESIDUAL = POLICIES[1:]
TIMINGS = ["early", "middle", "late"]
TIMING_LABEL = {"early": "Early (legacy 18 cm rule)", "middle": "Middle 10-15 cm", "late": "Late 5-10 cm"}
DIRS = {"py": "+y", "my": "-y", "px": "+x", "mx": "-x"}


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
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    order = np.argsort(ps)
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def load(policy, timing, cond, v1_dir, suffix=""):
    if timing == "early":  # official v1 evaluations (robustness branch); ablation files are <policy>_postoff_<cond>
        f = v1_dir / (f"{policy}_postoff_{cond}.json" if suffix else f"{policy}_{cond}.json")
    else:
        f = V2 / f"{policy}_{timing}_{cond}{suffix}.json"
    if not f.exists():
        return None
    rep = json.loads(f.read_text())
    if "wall_seconds" not in rep:
        print(f"WARNING incomplete {f}")
        return None
    return rep


def episode_table(rep):
    rows = []
    for e in rep["episodes"]:
        dl = e.get("dynamic_libero")
        ev = e["perturb_events"][0] if e["perturb_events"] else None
        if dl is not None:
            status, reason = dl["status"], dl.get("reason")
        elif ev is None:
            status, reason = "not_triggered", "unknown_v1"
        else:
            status, reason = ("applied", None) if ev["applied"] else ("rejected", ev["rejection"])
        rows.append({"success": bool(e["success"]), "len": int(e["env_steps"]), "status": status, "reason": reason,
                     "step": ev["step"] if ev else None, "dist": ev["eef_distance"] if ev else None,
                     "phase": (ev.get("chunk_phase", ev["step"] % 10) if ev else None),
                     "init": e["initial_state_hash"]})
    return rows


def boot_ci(d, rng, n_boot):
    """d: (k_conditions, n_episodes) paired differences; resample episode indices jointly."""
    n = d.shape[1]
    idx = rng.integers(0, n, size=(n_boot, n))
    m = d.mean(0)[idx].mean(1)
    return round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)


def signflip_p(dm, rng, n=200000):
    obs = abs(dm.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, dm.size))
    return float((np.abs((signs * dm).mean(1)) >= obs - 1e-12).mean())


def branch_response(policy, timing, conds, v1_dir, K=20, scale=0.1):
    """Four direction runs of one policy/timing share every pre-trigger state (verified here), so
    they are branches from one state. Direction-specific residual = mean_d ||u_d(k) - mean_d u(k)||,
    common = ||mean_d u(k)|| (executed units, x scale) at k steps after the trigger."""
    reps, npz = [], []
    for c in conds:
        f = (v1_dir / f"{policy}_{c}.json") if timing == "early" else (V2 / f"{policy}_{timing}_{c}.json")
        if not f.exists() or not Path(str(f) + ".steps.npz").exists():
            return None
        reps.append(json.loads(f.read_text()))
        npz.append(np.load(str(f) + ".steps.npz"))
    n_ep = min(len(r["episodes"]) for r in reps)
    same_trig, max_pre = 0, 0.0
    spec = np.zeros(K)
    comm = np.zeros(K)
    used = 0
    for ep in range(n_ep):
        trig = [r["episodes"][ep]["perturb_events"][0]["step"] if r["episodes"][ep]["perturb_events"] else None for r in reps]
        if None in trig or len(set(trig)) != 1:
            continue
        same_trig += 1
        t0 = trig[0]
        us, obs = [], []
        for z in npz:
            m = z["episode"] == ep
            us.append(z["u"][m][:, :6])
            obs.append(z["obs"][m])
        L = min(len(u) for u in us)
        if L < t0 + K:
            continue
        pre = max(float(np.abs(o[:t0] - obs[0][:t0]).max()) for o in obs)
        max_pre = max(max_pre, pre)
        # step index t0 is the first action chosen after the perturbation (obs refreshed)
        U = np.stack([u[t0:t0 + K] for u in us]) * scale  # (4, K, 6)
        mu = U.mean(0)
        spec += np.linalg.norm(U - mu, axis=2).mean(0)
        comm += np.linalg.norm(mu, axis=1)
        used += 1
    if used == 0:
        return None
    spec /= used
    comm /= used
    return {"policy": policy, "timing": timing, "episodes_same_trigger": same_trig, "episodes_used": used,
            "max_abs_obs_diff_before_trigger": max_pre,
            "specific_k1": round(float(spec[0]), 5), "specific_k5": round(float(spec[4]), 5),
            "specific_k10": round(float(spec[9]), 5), "specific_k20": round(float(spec[19]), 5),
            "common_k1_20_mean": round(float(comm.mean()), 5),
            "ratio_specific_over_common_k1_20": round(float(spec.mean() / comm.mean()), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-dir", default=str(ROOT.parent / "steer-frozen-vla/results/residual_robustness/evals"))
    ap.add_argument("--distance", type=int, default=2, choices=[2, 4])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--perm", type=int, default=200000)
    args = ap.parse_args()
    v1 = Path(args.v1_dir)
    D = args.distance
    conds = [f"{k}{D}" for k in DIRS]
    out = OUT / f"{D}cm"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tab, missing = {}, []
    for t in TIMINGS:
        for c in conds:
            for p in POLICIES:
                rep = load(p, t, c, v1)
                if rep is None:
                    missing.append(f"{p}/{t}/{c}")
                    continue
                tab[(p, t, c)] = episode_table(rep)
                if p != "base":
                    rep_off = load(p, t, c, v1, suffix="_postoff")
                    if rep_off is not None:
                        tab[(p + "_postoff", t, c)] = episode_table(rep_off)
    # ---------------- per condition
    cond_rows = []
    for (p, t, c), eps in sorted(tab.items()):
        n = len(eps)
        s = sum(e["success"] for e in eps)
        st = [e["status"] for e in eps]
        steps = [e["step"] for e in eps if e["step"] is not None]
        dists = [e["dist"] for e in eps if e["dist"] is not None]
        lens = [e["len"] for e in eps if e["success"]]
        reasons = {}
        for e in eps:
            if e["status"] != "applied":
                reasons[f'{e["status"]}:{e["reason"]}'] = reasons.get(f'{e["status"]}:{e["reason"]}', 0) + 1
        cond_rows.append({"policy": p, "timing": t, "condition": f"{DIRS[c[:2]]} {D}cm", "episodes": n,
                          "successes": s, "success_rate": round(s / n, 4), "wilson95": wilson(s, n),
                          "applied_rate": round(st.count("applied") / n, 4), "rejected_rate": round(st.count("rejected") / n, 4),
                          "not_triggered_rate": round(st.count("not_triggered") / n, 4), "non_applied_reasons": reasons,
                          "success_among_applied_NOT_PRIMARY": round(sum(e["success"] for e in eps if e["status"] == "applied")
                                                                     / max(1, st.count("applied")), 4),
                          "trigger_step_mean": round(float(np.mean(steps)), 2) if steps else None,
                          "trigger_step_min": min(steps) if steps else None, "trigger_step_max": max(steps) if steps else None,
                          "eef_distance_mean": round(float(np.mean(dists)), 4) if dists else None,
                          "eef_distance_min": round(min(dists), 4) if dists else None,
                          "eef_distance_max": round(max(dists), 4) if dists else None,
                          "success_len_median": float(np.median(lens)) if lens else None,
                          "success_len_mean": round(float(np.mean(lens)), 1) if lens else None})
    # ---------------- paired residual - base (per condition, and pooled over directions per timing)
    pair_rows = []
    for p in RESIDUAL + [r + "_postoff" for r in RESIDUAL]:
        ref = "base" if not p.endswith("_postoff") else p[: -len("_postoff")]
        for t in TIMINGS:
            avail = [c for c in conds if (p, t, c) in tab and (ref, t, c) in tab]
            if not avail:
                continue
            for group in [[c] for c in avail] + ([avail] if len(avail) == 4 else []):
                a = np.array([[e["success"] for e in tab[(p, t, c)]] for c in group], float)
                b = np.array([[e["success"] for e in tab[(ref, t, c)]] for c in group], float)
                assert all(x["init"] == y["init"] for c in group for x, y in zip(tab[(p, t, c)], tab[(ref, t, c)]))
                d = a - b
                gained, lost = int((d > 0).sum()), int((d < 0).sum())
                lo, hi = boot_ci(d, rng, args.boot)
                row = {"policy": p, "reference": ref, "timing": t,
                       "conditions": "all 4 directions" if len(group) == 4 else f"{DIRS[group[0][:2]]} {D}cm",
                       "policy_rate": round(a.mean(), 4), "reference_rate": round(b.mean(), 4),
                       "diff": round(d.mean(), 4), "boot95_lo": lo, "boot95_hi": hi,
                       "gained": gained, "lost": lost, "mcnemar_exact_p": round(mcnemar_exact(gained, lost), 5)}
                if len(group) == 4:
                    row["signflip_p"] = round(signflip_p(d.mean(0), rng, args.perm), 5)
                pair_rows.append(row)
    # Holm over the pooled Middle/Late tests (3 seeds x 2 timings): the family reported as main result
    fam = [r for r in pair_rows if r["conditions"] == "all 4 directions" and r["timing"] in ("middle", "late")
           and r["reference"] == "base"]
    for r, hp in zip(fam, holm([r["signflip_p"] for r in fam])):
        r["holm_p_middle_late_family"] = round(hp, 5)
    # ---------------- timing change (paired by episode within policy): middle/late - early
    tchange = []
    for p in POLICIES:
        for t in ("middle", "late"):
            avail = [c for c in conds if (p, t, c) in tab and (p, "early", c) in tab]
            if len(avail) != 4:
                continue
            a = np.array([[e["success"] for e in tab[(p, t, c)]] for c in avail], float)
            b = np.array([[e["success"] for e in tab[(p, "early", c)]] for c in avail], float)
            d = a - b
            lo, hi = boot_ci(d, rng, args.boot)
            tchange.append({"policy": p, "timing": t, "early_rate": round(b.mean(), 4), "timing_rate": round(a.mean(), 4),
                            "change": round(d.mean(), 4), "boot95_lo": lo, "boot95_hi": hi,
                            "per_direction": {DIRS[c[:2]]: round(float(d[i].mean()), 3) for i, c in enumerate(avail)}})
    # ---------------- seed consistency
    seed_rows = []
    for t in TIMINGS:
        v = [r["diff"] for r in pair_rows if r["timing"] == t and r["conditions"] == "all 4 directions"
             and r["policy"] in RESIDUAL]
        if len(v) == 3:
            per_dir = {}
            for c in conds:
                lab = f"{DIRS[c[:2]]} {D}cm"
                per_dir[lab] = [r["diff"] for r in pair_rows if r["timing"] == t and r["conditions"] == lab and r["policy"] in RESIDUAL]
            seed_rows.append({"timing": t, "diff_mean": round(float(np.mean(v)), 4), "diff_sd": round(float(np.std(v, ddof=1)), 4),
                              "diff_min": min(v), "diff_max": max(v), "seeds_positive": sum(x > 0 for x in v),
                              "per_direction_diffs_s42_s43_s44": per_dir})
    # ---------------- branch response (C2 analogue) per policy and timing
    branch_rows = [r for r in (branch_response(p, t, conds, v1) for p in RESIDUAL for t in TIMINGS) if r]
    # ---------------- write
    def write(name, rows):
        if not rows:
            return
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    write("G_condition_summary.csv", cond_rows)
    write("G_paired_vs_base.csv", [r for r in pair_rows if r["reference"] == "base"])
    write("G_postoff_ablation.csv", [r for r in pair_rows if r["reference"] != "base"])
    write("G_timing_change_vs_early.csv", tchange)
    write("G_seed_consistency.csv", seed_rows)
    write("G_branch_response.csv", branch_rows)
    (out / "G_summary.json").write_text(json.dumps({"missing": missing, "conditions": cond_rows, "paired": pair_rows,
                                                    "timing_change": tchange, "seed_consistency": seed_rows,
                                                    "branch_response": branch_rows}, indent=1, default=str))
    plot(cond_rows, pair_rows, out, D)
    print(f"{len(cond_rows)} condition rows, {len(pair_rows)} paired rows, missing {len(missing)}: {missing[:12]}")


def plot(cond_rows, pair_rows, out, D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"base": "#444444", "s42_val": "tab:blue", "s43_val": "tab:orange", "s44_val": "tab:green"}
    labs = [f"{d} {D}cm" for d in ("+y", "-y", "+x", "-x")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), sharey=True)
    for ax, t in zip(axes, TIMINGS):
        for j, p in enumerate(POLICIES):
            ys, los, his = [], [], []
            for lab in labs:
                r = next((r for r in cond_rows if r["policy"] == p and r["timing"] == t and r["condition"] == lab), None)
                ys.append(r["success_rate"] if r else np.nan)
                los.append(r["success_rate"] - r["wilson95"][0] if r else 0)
                his.append(r["wilson95"][1] - r["success_rate"] if r else 0)
            x = np.arange(4) + (j - 1.5) * 0.2
            ax.bar(x, ys, 0.2, color=colors[p], label="Frozen Base" if p == "base" else p,
                   yerr=[los, his], capsize=2, error_kw={"lw": 0.8})
        ax.set_xticks(range(4))
        ax.set_xticklabels(labs)
        ax.set_title(TIMING_LABEL[t])
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("success rate (50 episodes, Wilson 95% CI)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "figures/G_success_by_timing_direction.png", dpi=130)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, p in enumerate(RESIDUAL):
        xs, ys, lo, hi = [], [], [], []
        for i, t in enumerate(TIMINGS):
            r = next((r for r in pair_rows if r["policy"] == p and r["timing"] == t and r["conditions"] == "all 4 directions"), None)
            if r:
                xs.append(i + (j - 1) * 0.15)
                ys.append(r["diff"])
                lo.append(r["diff"] - r["boot95_lo"])
                hi.append(r["boot95_hi"] - r["diff"])
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", color=colors[p], label=p, capsize=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([TIMING_LABEL[t] for t in TIMINGS])
    ax.set_ylabel(f"Residual - Base success ({D}cm, 4-direction mean)")
    ax.set_title("Paired difference vs Frozen Base (95% bootstrap CI over episodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures/G_paired_diff_by_timing.png", dpi=130)


if __name__ == "__main__":
    main()
