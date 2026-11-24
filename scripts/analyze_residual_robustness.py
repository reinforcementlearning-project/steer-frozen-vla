"""Robustness (RQ1) and reactive-adaptation analysis of the fixed Residual-SAC candidate.

Follows configs/robustness_protocol.json. Reads results/residual_robustness/evals/
(<policy>_<condition>.json + .steps.npz) and the three training runs; writes CSVs and
figures into results/residual_robustness/. Read-only for every input.

Usage: analyze_residual_robustness.py [--perm 200000] [--boot 10000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_residual_eval import mcnemar_exact  # noqa: E402
from evaluation.summarize import wilson  # noqa: E402
from analyze_residual_diagnostics import (CONDS, DIMS, RESIDUAL_SCALE, lda_cv, perturb_steps, rnd,  # noqa: E402
                                          training_stability, write_csv)

OUT = ROOT / "results/residual_robustness"
EV = OUT / "evals"
SEEDS = [42, 43, 44]
RUN = {s: ROOT / f"results/residual_6d_rand_fixa001_2cm_s{s}_20k" for s in SEEDS}
C2 = ["py2", "my2", "px2", "mx2"]
C4 = ["py4", "my4", "px4", "mx4"]
METRICS = {"M1": ["0cm"], "M2": C2, "M3": C4}


def load(pol, cond, steps=False):
    f = EV / f"{pol}_{cond}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    if "wall_seconds" not in d or len(d["episodes"]) != 50:
        print(f"WARNING skipping incomplete report {f.name}")
        return None
    if steps:
        s = np.load(str(f) + ".steps.npz")
        return d, {k: s[k] for k in s.files}
    return d


def succ(d):
    return np.array([bool(e["success"]) for e in d["episodes"]], dtype=float)


def check_pairing(a, b):
    ka = [(e["noise_seed"], e["initial_state_hash"]) for e in a["episodes"]]
    kb = [(e["noise_seed"], e["initial_state_hash"]) for e in b["episodes"]]
    if ka != kb:
        raise SystemExit("episodes are not paired")


def signflip_p(d, n_perm, rng):
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = np.abs((signs * d).mean(1))
    return float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))


def boot_ci(d, n_boot, rng):
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


# ============================================================== robustness
def robustness(args):
    rows, cond_rows, perturb_rows, metric_rows, m4_rows, sanity_rows = [], [], [], [], [], []
    base = {c: load("base", c) for c in CONDS}
    policies = ["base"] + [f"s{s}_{k}" for s in SEEDS for k in ("val", "final")]
    for pol in policies + [f"s{s}_val_postoff" for s in SEEDS] + ["s42_final_scale0"]:
        for c, (axis, direc, dist, label) in CONDS.items():
            d = load(pol, c)
            if d is None:
                continue
            check_pairing(d, base[c])
            y = succ(d)
            lo, hi = wilson(int(y.sum()), len(y))
            ev = [e.get("perturb_events", []) for e in d["episodes"]]
            applied = [x[0] for x in ev if x and x[0]["applied"]]
            rej = [x[0]["rejection"] for x in ev if x and not x[0]["applied"]]
            rows.append({"policy": pol, "condition": label, "episodes": len(y), "successes": int(y.sum()),
                         "success_rate": rnd(y.mean()), "wilson95_lo": rnd(lo), "wilson95_hi": rnd(hi),
                         "checkpoint": d.get("checkpoint")})
            perturb_rows.append({"policy": pol, "condition": label, "applied": len(applied),
                                 "rejected": len(rej), "rejection_reasons": ";".join(sorted(set(rej))),
                                 "never_triggered": sum(not x for x in ev),
                                 "displacement_m_mean": rnd(np.mean([a.get("actual_displacement_m", 0) for a in applied])) if applied else None,
                                 "displacement_m_min": rnd(np.min([a.get("actual_displacement_m", 0) for a in applied])) if applied else None,
                                 "trigger_step_mean": rnd(np.mean([a["step"] for a in applied]), 2) if applied else None,
                                 "trigger_step_sd": rnd(np.std([a["step"] for a in applied]), 2) if applied else None,
                                 "trigger_eef_dist_m": rnd(np.mean([a["eef_distance"] for a in applied])) if applied else None})
            if pol != "base":
                yb = succ(base[c])
                g, l = int(((y == 1) & (yb == 0)).sum()), int(((y == 0) & (yb == 1)).sum())
                cond_rows.append({"policy": pol, "condition": label, "base_rate": rnd(yb.mean()),
                                  "policy_rate": rnd(y.mean()), "delta_pp": rnd(100 * (y.mean() - yb.mean()), 1),
                                  "gained": g, "lost": l, "mcnemar_exact_p": rnd(mcnemar_exact(g, l), 5),
                                  "role": "exploratory"})
            if pol == "s42_final_scale0":
                yb = succ(base[c])
                same = sum(x["success"] == z["success"] and x["env_steps"] == z["env_steps"]
                           for x, z in zip(d["episodes"], base[c]["episodes"]))
                sanity_rows.append({"check": "alpha=0 wrapper, actor running (seed 42 final)", "condition": label,
                                    "episodes": 50, "identical_success_and_length_vs_base": same,
                                    "policy_successes": int(y.sum()), "base_successes": int(yb.sum())})

    # primary + secondary metrics
    rng = np.random.default_rng(0)
    for s in SEEDS:
        for kind in ("val", "final"):
            pol = f"s{s}_{kind}"
            if not all((EV / f"{pol}_{c}.json").exists() for c in CONDS):
                continue
            for m, cs in METRICS.items():
                yr = np.mean([succ(load(pol, c)) for c in cs], 0)
                yb = np.mean([succ(base[c]) for c in cs], 0)
                d = yr - yb
                lo, hi = boot_ci(d, args.boot, rng)
                metric_rows.append({"seed": s, "policy": pol, "checkpoint_kind": kind, "metric": m,
                                    "conditions": "/".join(CONDS[c][3] for c in cs),
                                    "base_rate": rnd(yb.mean()), "policy_rate": rnd(yr.mean()),
                                    "diff": rnd(d.mean()), "boot95_lo": rnd(lo), "boot95_hi": rnd(hi),
                                    "signflip_p": signflip_p(d, args.perm, rng),
                                    "role": "primary" if kind == "val" else "exploratory"})
            for dist, cs in ((2, C2), (4, C4)):
                rates = {CONDS[c][3]: succ(load(pol, c)).mean() for c in cs}
                brates = {CONDS[c][3]: succ(base[c]).mean() for c in cs}
                m4_rows.append({"seed": s, "policy": pol, "distance_cm": dist,
                                "min_rate": rnd(min(rates.values())), "min_condition": min(rates, key=rates.get),
                                "max_rate": rnd(max(rates.values())), "max_condition": max(rates, key=rates.get),
                                "range": rnd(max(rates.values()) - min(rates.values())),
                                "base_min_rate": rnd(min(brates.values())), "base_max_rate": rnd(max(brates.values())),
                                "base_range": rnd(max(brates.values()) - min(brates.values())),
                                "n_directions_below_base": sum(rates[k] < brates[k] for k in rates),
                                "worst_direction_change_vs_base": rnd(min(rates[k] - brates[k] for k in rates))})
    prim = [r for r in metric_rows if r["role"] == "primary"]
    if prim:
        adj = holm(np.array([r["signflip_p"] for r in prim]))
        for r, a in zip(prim, adj):
            r["holm_p"] = rnd(a, 5)
    for r in metric_rows:
        r["signflip_p"] = rnd(r["signflip_p"], 5)
    # seed variability
    var_rows = []
    for kind in ("val", "final"):
        for m in METRICS:
            v = [r for r in metric_rows if r["checkpoint_kind"] == kind and r["metric"] == m]
            if len(v) < 2:
                continue
            diffs = np.array([r["diff"] for r in v])
            rates = np.array([r["policy_rate"] for r in v])
            var_rows.append({"checkpoint_kind": kind, "metric": m, "n_seeds": len(v),
                             "seeds": "/".join(str(r["seed"]) for r in v),
                             "policy_rate_mean": rnd(rates.mean()), "policy_rate_sd": rnd(rates.std(ddof=1)),
                             "diff_mean": rnd(diffs.mean()), "diff_sd": rnd(diffs.std(ddof=1)),
                             "diff_min": rnd(diffs.min()), "diff_max": rnd(diffs.max()),
                             "seeds_with_positive_diff": int((diffs > 0).sum()),
                             "base_rate": v[0]["base_rate"]})
    write_csv(OUT / "R_success_by_condition.csv", rows)
    write_csv(OUT / "R_mcnemar_by_condition.csv", cond_rows)
    write_csv(OUT / "R_perturbation_check.csv", perturb_rows)
    write_csv(OUT / "R_primary_metrics.csv", metric_rows)
    write_csv(OUT / "R_direction_generalization_M4.csv", m4_rows)
    write_csv(OUT / "R_seed_variability.csv", var_rows)
    write_csv(OUT / "R_sanity_alpha0.csv", sanity_rows)
    plot_success(rows)
    plot_metrics(metric_rows)
    print(f"robustness: {len(rows)} runs, {len(metric_rows)} metric rows")
    return rows


def plot_success(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [v[3] for v in CONDS.values()]
    for kind in ("val", "final"):
        pols = ["base"] + [f"s{s}_{kind}" for s in SEEDS]
        cols = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#2ca02c"]
        fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
        w = .8 / len(pols)
        for i, (pol, col) in enumerate(zip(pols, cols)):
            xs, ys, lo, hi = [], [], [], []
            for j, lab in enumerate(labels):
                r = next((r for r in rows if r["policy"] == pol and r["condition"] == lab), None)
                if r:
                    xs.append(j + (i - (len(pols) - 1) / 2) * w)
                    ys.append(r["success_rate"])
                    lo.append(r["success_rate"] - r["wilson95_lo"])
                    hi.append(r["wilson95_hi"] - r["success_rate"])
            ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, color=col,
                   label="Frozen base" if pol == "base" else f"seed {pol[1:3]} ({kind})")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 1.08)
        ax.axvspan(.5, 4.5, color="#1f77b4", alpha=.04)
        ax.text(2.5, 1.03, "training range (2cm)", ha="center", fontsize=8)
        ax.text(6.5, 1.03, "out of training range (4cm)", ha="center", fontsize=8)
        ax.set_ylabel("success (50 ep, Wilson 95% CI)")
        ax.set_title(f"Residual-SAC (6D, random dir, entropy coef 0.01) — {'validation-selected' if kind == 'val' else 'final 20k'} checkpoints")
        ax.legend(ncol=4, fontsize=8, loc="lower left")
        ax.grid(axis="y", alpha=.25)
        fig.savefig(OUT / "figures" / f"R_success_{kind}.png", dpi=140)
        plt.close(fig)


def plot_metrics(metric_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True, sharey=True)
    for ax, m in zip(axes, METRICS):
        for j, kind in enumerate(("val", "final")):
            v = [r for r in metric_rows if r["metric"] == m and r["checkpoint_kind"] == kind]
            xs = np.arange(len(v)) + (j - .5) * .3
            ax.errorbar(xs, [r["diff"] for r in v], yerr=[[r["diff"] - r["boot95_lo"] for r in v],
                        [r["boot95_hi"] - r["diff"] for r in v]], fmt="o" if kind == "val" else "s", capsize=3,
                        label="validation-selected (primary)" if kind == "val" else "final 20k")
            ax.set_xticks(range(len(v)), [f"seed {r['seed']}" for r in v])
        ax.axhline(0, color="k", lw=.6)
        ax.set_title({"M1": "M1: 0cm", "M2": "M2: 2cm, 4-direction mean", "M3": "M3: 4cm, 4-direction mean"}[m])
        ax.grid(alpha=.25)
    axes[0].set_ylabel("success difference vs frozen base\n(paired bootstrap 95% CI)")
    axes[0].legend(fontsize=8)
    fig.savefig(OUT / "figures" / "R_primary_metrics.png", dpi=140)
    plt.close(fig)


# ============================================================== training
def training():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), constrained_layout=True)
    A = axes.ravel()
    for s in SEEDS:
        r = RUN[s]
        if not (r / "summary.json").exists():
            continue
        training_stability(r, OUT, OUT / "figures", f"s{s}_")
        tr, ep = pd.read_csv(r / "train.csv"), pd.read_csv(r / "episodes.csv")
        buf = np.load(r / "checkpoints/replay_buffer.npz")
        n = int(buf["meta"][1])
        rew = buf["rewards"][:n].ravel()
        late = tr[tr.env_step > 10000]
        sel = json.loads((r / "validation/selection.json").read_text()) if (r / "validation/selection.json").exists() else {}
        rows.append({"seed": s, "env_steps": int(tr.env_step.max()), "grad_updates": int(tr.grad_updates.max()),
                     "episodes": len(ep), "train_successes": int(ep.success.sum()),
                     "success_transitions": int((rew > 0).sum()), "success_transition_frac": rnd((rew > 0).mean(), 5),
                     "entropy_coef": rnd(tr.alpha.dropna().iloc[-1], 5),
                     "policy_entropy_min_after_2k": rnd(tr[tr.env_step > 2000].entropy.min()),
                     "policy_entropy_last": rnd(tr.entropy.dropna().iloc[-1]),
                     "entropy_collapse_below_0": bool((tr[tr.env_step > 2000].entropy < 0).any()),
                     "q1_mean_last": rnd(tr.q1_mean.dropna().iloc[-1]), "q1_mean_max": rnd(tr.q1_mean.max()),
                     "critic_loss_late_mean": rnd(late.critic_loss.mean(), 5),
                     "actor_loss_last": rnd(tr.actor_loss.dropna().iloc[-1]),
                     "episode_residual_norm_last5k_mean": rnd(ep[ep.env_step > 15000].residual_norm_mean.mean()),
                     "episode_residual_sat_last5k_mean": rnd(ep[ep.env_step > 15000].residual_sat_frac.mean(), 5),
                     "train_success_last5k": rnd(ep[ep.env_step > 15000].success.mean()),
                     "selected_checkpoint_step": sel.get("selected_env_step"),
                     "selected_val_score": sel.get("selected_score"), "final_val_score": sel.get("final_score")})
        A[0].plot(ep.env_step, ep.success.rolling(20, min_periods=5).mean(), label=f"seed {s}")
        A[1].plot(tr.env_step, tr.entropy, label=f"seed {s}")
        A[2].plot(ep.env_step, ep.residual_norm_mean, ".", ms=3, label=f"seed {s}")
        A[3].plot(tr.env_step, tr.q1_mean, label=f"seed {s}")
        A[4].plot(tr.env_step, tr.actor_loss, label=f"seed {s}")
        A[5].plot(tr.env_step, tr.critic_loss, label=f"seed {s}")
    for ax, t in zip(A, ("training success (rolling 20 ep, stochastic policy)", "policy entropy (-log pi)",
                         "per-episode mean executed residual ||0.1u||", "Q1 mean", "actor loss", "critic loss")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    A[5].set_yscale("log")
    fig.suptitle("Training stability across seeds (entropy coefficient fixed at 0.01)")
    fig.savefig(OUT / "figures" / "T_training_seeds.png", dpi=130)
    plt.close(fig)
    write_csv(OUT / "T_training_summary.csv", rows)
    print(f"training: {len(rows)} seeds")


# ============================================================== adaptation
def rel_steps(d, s):
    ps = perturb_steps(d)[s["episode"].astype(int)]
    return s["step"] - ps  # 0 = first action chosen after the perturbation was applied


def adaptation():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c1_rows, tc_rows, branch_rows, branch_tc, dec_rows, ablation_rows = [], [], [], [], [], []
    pols = [f"s{s}_{k}" for s in SEEDS for k in ("val", "final")]
    windows = {"pre(-10..-1)": (-10, 0), "post 0-4": (0, 5), "post 5-9": (5, 10), "post 10-19": (10, 20),
               "post 20-49": (20, 50)}
    for pol in pols:
        for c in CONDS:
            got = load(pol, c, steps=True)
            if got is None:
                continue
            d, s = got
            rel = rel_steps(d, s)
            u = s["u"][:, :6]
            dn = RESIDUAL_SCALE * np.linalg.norm(u, axis=1)
            eb = np.abs(s["exec_action"][:, :6] - s["base_action"][:, :6]).sum(1)
            for wname, (a, b) in windows.items():
                m = (rel >= a) & (rel < b)
                if m.sum() == 0:
                    continue
                c1_rows.append({"policy": pol, "condition": CONDS[c][3], "window": wname, "n_steps": int(m.sum()),
                                "delta_l2_mean": rnd(dn[m].mean()), "exec_minus_base_l1_mean": rnd(eb[m].mean()),
                                **{f"u_{n}_mean": rnd(u[m][:, i].mean()) for i, n in enumerate(DIMS[:6])}})
            for k in range(-10, 51):
                m = rel == k
                if m.sum() >= 10:
                    tc_rows.append({"policy": pol, "condition": CONDS[c][3], "rel_step": k, "n": int(m.sum()),
                                    "delta_l2_mean": rnd(dn[m].mean()),
                                    **{f"u_{n}_mean": rnd(u[m][:, i].mean()) for i, n in enumerate(DIMS[:3])}})

    # C2: the four direction runs of one policy share the pre-perturbation trajectory (branch point)
    agents = {}
    for pol in ["base"] + pols:
        for dist, cs in ((2, C2), (4, C4)):
            runs = [load(pol, c, steps=True) for c in cs]
            if any(r is None for r in runs):
                continue
            pre_max_diff, trig_equal, n_eps = 0., 0, 50
            per_k = {k: {"resp": [], "mean": [], "obs": []} for k in range(0, 31)}
            Xu, Xo, Y, G = [], [], [], []
            for e in range(n_eps):
                trig = [perturb_steps(d)[e] for d, _ in runs]
                if not all(np.isfinite(trig)):
                    continue
                trig_equal += len(set(trig)) == 1
                t0 = int(trig[0])
                seqs = []
                for d, s in runs:
                    m = s["episode"] == e
                    seqs.append((s["step"][m], s["obs"][m], s["u"][m][:, :6]))
                for st, ob, _ in seqs[1:]:
                    k0 = min(t0, len(st), len(seqs[0][0]))
                    pre_max_diff = max(pre_max_diff, float(np.abs(ob[:k0] - seqs[0][1][:k0]).max()) if k0 else 0.)
                for k in range(0, 31):
                    idx = t0 + k
                    if all(idx < len(st) for st, _, _ in seqs):
                        U = np.stack([uu[idx] for _, _, uu in seqs]) * RESIDUAL_SCALE
                        O = np.stack([ob[idx] for _, ob, _ in seqs])
                        per_k[k]["resp"].append(np.linalg.norm(U - U.mean(0), axis=1).mean())
                        per_k[k]["mean"].append(np.linalg.norm(U.mean(0)))
                        per_k[k]["obs"].append(np.linalg.norm(O - O.mean(0), axis=1).mean())
                        if 1 <= k <= 20:
                            for di in range(4):
                                Xu.append(U[di] / RESIDUAL_SCALE)
                                Xo.append(O[di])
                                Y.append(di)
                                G.append(e)
            for k, v in per_k.items():
                if len(v["resp"]) >= 10:
                    branch_tc.append({"policy": pol, "distance_cm": dist, "rel_step": k, "n_episodes": len(v["resp"]),
                                      "direction_specific_resid_norm": rnd(np.mean(v["resp"]), 5),
                                      "direction_common_resid_norm": rnd(np.mean(v["mean"]), 5),
                                      "obs_divergence": rnd(np.mean(v["obs"]), 5)})
            Xu, Xo, Y, G = map(np.asarray, (Xu, Xo, Y, G))
            branch_rows.append({"policy": pol, "distance_cm": dist, "episodes_all_triggered": int(trig_equal),
                                "max_abs_obs_diff_before_trigger": pre_max_diff,
                                "trigger_step_identical_episodes": int(trig_equal),
                                "resp_ratio_k1_20": rnd(np.mean([np.mean(per_k[k]["resp"]) for k in range(1, 21) if per_k[k]["resp"]]) /
                                                        max(np.mean([np.mean(per_k[k]["mean"]) for k in range(1, 21) if per_k[k]["mean"]]), 1e-9))
                                if pol != "base" else None})
            if len(Y):
                dec_rows.append({"policy": pol, "distance_cm": dist, "window": "post 1-20 (branch-aligned)",
                                 "n": len(Y), "chance": .25,
                                 "lda_acc_from_obs15": rnd(lda_cv(Xo, Y, G)),
                                 "lda_acc_from_residual6": rnd(lda_cv(Xu, Y, G)) if pol != "base" else None})

    # post-perturbation residual ablation (primary policies)
    for s in SEEDS:
        pol, off = f"s{s}_val", f"s{s}_val_postoff"
        for dist, cs in ((2, C2), (4, C4)):
            rf, ro = [load(pol, c) for c in cs], [load(off, c) for c in cs]
            if any(r is None for r in rf + ro):
                continue
            yf = np.concatenate([succ(r) for r in rf])
            yo = np.concatenate([succ(r) for r in ro])
            yb = np.concatenate([succ(load("base", c)) for c in cs])
            g, l = int(((yf == 1) & (yo == 0)).sum()), int(((yf == 0) & (yo == 1)).sum())
            ablation_rows.append({"seed": s, "distance_cm": dist, "episodes": len(yf),
                                  "full_rate": rnd(yf.mean()), "postoff_rate": rnd(yo.mean()), "base_rate": rnd(yb.mean()),
                                  "full_minus_postoff": rnd(yf.mean() - yo.mean()),
                                  "full_only_success": g, "postoff_only_success": l,
                                  "mcnemar_exact_p_pooled_dirs": rnd(mcnemar_exact(g, l), 5),
                                  "per_direction": "; ".join(f"{CONDS[c][3]} {succ(a).mean():.2f}/{succ(b).mean():.2f}"
                                                             for c, a, b in zip(cs, rf, ro))})
    write_csv(OUT / "C1_residual_pre_post_by_direction.csv", c1_rows)
    write_csv(OUT / "C1_residual_timecourse.csv", tc_rows)
    write_csv(OUT / "C2_branch_check_and_response.csv", branch_rows)
    write_csv(OUT / "C2_branch_response_timecourse.csv", branch_tc)
    write_csv(OUT / "C2_direction_decodability.csv", dec_rows)
    write_csv(OUT / "C_postperturb_ablation.csv", ablation_rows)

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), constrained_layout=True, sharey=True)
    for ax, s in zip(axes, SEEDS):
        for c in ["0cm"] + C4:
            v = [r for r in tc_rows if r["policy"] == f"s{s}_val" and r["condition"] == CONDS[c][3]]
            ax.plot([r["rel_step"] for r in v], [r["delta_l2_mean"] for r in v], label=CONDS[c][3])
        ax.axvline(0, color="k", ls="--", lw=.7)
        ax.set_title(f"seed {s} (validation-selected): executed residual norm", fontsize=10)
        ax.set_xlabel("steps relative to perturbation")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "figures" / "C1_residual_norm_timecourse.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(16, 6.5), constrained_layout=True, sharey="row")
    for j, s in enumerate(SEEDS):
        pol = f"s{s}_val"
        for row, dist, cs in ((0, 2, C2), (1, 4, C4)):
            ax = axes[row, j]
            for i, c in enumerate(cs):
                v = [r for r in c1_rows if r["policy"] == pol and r["condition"] == CONDS[c][3] and r["window"].startswith("post")]
                v = sorted(v, key=lambda r: list(windows).index(r["window"]))
                ax.plot(range(len(v)), [r["u_dx_mean"] for r in v], "-o", ms=3, color=f"C{i}", label=f"{CONDS[c][3]} dx")
                ax.plot(range(len(v)), [r["u_dy_mean"] for r in v], "--s", ms=3, color=f"C{i}", label=f"{CONDS[c][3]} dy")
            ax.set_xticks(range(4), ["0-4", "5-9", "10-19", "20-49"])
            ax.axhline(0, color="k", lw=.5)
            ax.set_title(f"seed {s}, {dist}cm: mean residual u (dx solid, dy dashed)", fontsize=9)
            ax.grid(alpha=.25)
            if j == 0:
                ax.legend(fontsize=6, ncol=2)
    fig.supxlabel("steps after perturbation (window)")
    fig.savefig(OUT / "figures" / "C1_direction_response.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
    for ax, dist in zip(axes, (2, 4)):
        for s in SEEDS:
            v = [r for r in branch_tc if r["policy"] == f"s{s}_val" and r["distance_cm"] == dist]
            ax.plot([r["rel_step"] for r in v], [r["direction_specific_resid_norm"] for r in v], label=f"seed {s}: direction-specific")
            ax.plot([r["rel_step"] for r in v], [r["direction_common_resid_norm"] for r in v], ":", label=f"seed {s}: common")
        ax.set_title(f"{dist}cm branches: residual split into direction-specific vs common part", fontsize=10)
        ax.set_xlabel("steps after perturbation")
        ax.set_ylabel("||0.1 u|| component")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    fig.savefig(OUT / "figures" / "C2_branch_response.png", dpi=140)
    plt.close(fig)
    print(f"adaptation: {len(c1_rows)} C1 rows, {len(branch_rows)} branch rows, {len(ablation_rows)} ablation rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perm", type=int, default=200000)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--parts", nargs="+", default=["robustness", "training", "adaptation"])
    args = p.parse_args()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    if "robustness" in args.parts:
        robustness(args)
    if "training" in args.parts:
        training()
    if "adaptation" in args.parts:
        adaptation()


if __name__ == "__main__":
    main()
