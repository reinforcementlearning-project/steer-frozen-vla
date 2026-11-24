"""Seed-42 Residual-SAC diagnostics: direction generalization (A), residual action
distribution (B) and SAC training stability (C).

Read-only with respect to the training run, the shared SAC code (rl/) and earlier
results. Writes only into --out (default results/residual_s42_diagnostics).

Usage: analyze_residual_diagnostics.py [--parts A B C]
Terminology: "entropy coefficient" = SAC temperature (log_alpha, auto-tuned);
"residual scale" = ResidualEnv.alpha = 0.1 (fixed; executed delta = 0.1 * u).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluation.summarize import wilson  # noqa: E402
from compare_residual_eval import compare  # noqa: E402
from rl import ActionSpec, SACAgent, SACConfig  # noqa: E402

RUN = ROOT / "results/residual_2cm_s42_20261124_20k"
RUNS = {"s42": RUN,  # baseline: 7D residual, fixed +y 2cm
        "v6d": ROOT / "results/residual_6d_2cm_s42_20k",  # follow-up 1: 6D residual (target entropy -6)
        "v6drand": ROOT / "results/residual_6d_rand_2cm_s42_20k",  # follow-up 2: 6D + random direction
        "v6drandfix": ROOT / "results/residual_6d_rand_fixa001_2cm_s42_20k"}  # follow-up 3: + fixed entropy coef 0.01
CKPTS = {"ck7k": RUN / "checkpoints/ckpt_00007000.pt", "final": RUN / "checkpoints/ckpt_00020000.pt",
         "v6d_final": RUNS["v6d"] / "checkpoints/ckpt_00020000.pt",
         "v6drand_final": RUNS["v6drand"] / "checkpoints/ckpt_00020000.pt",
         "fixa_final": RUNS["v6drandfix"] / "checkpoints/ckpt_00020000.pt"}
_sel = RUNS["v6drandfix"] / "validation/selection.json"
if _sel.exists():  # pre-registered validation-selected checkpoint
    CKPTS["fixa_val"] = ROOT / json.loads(_sel.read_text())["selected_checkpoint"]
CKPTS = {k: v for k, v in CKPTS.items() if v.exists()}
POLICIES = ["base"] + list(CKPTS)
PNAME = {"base": "Frozen base", "ck7k": "Residual 7k", "final": "Residual 20k",
         "v6d_final": "6D 20k", "v6drand_final": "6D+rand 20k", "fixa_final": "6D+rand fixed-coef 20k",
         "fixa_val": "6D+rand fixed-coef (val-selected)"}
PCOLOR = {"base": "#7f7f7f", "ck7k": "#1f77b4", "final": "#d62728", "v6d_final": "#2ca02c",
          "v6drand_final": "#9467bd", "fixa_final": "#ff7f0e", "fixa_val": "#8c564b"}
# name: (axis, direction, distance_cm, label)
CONDS = {"0cm": (1, 1, 0, "0cm"), "py2": (1, 1, 2, "+y 2cm"), "my2": (1, -1, 2, "-y 2cm"),
         "px2": (0, 1, 2, "+x 2cm"), "mx2": (0, -1, 2, "-x 2cm"), "py4": (1, 1, 4, "+y 4cm"),
         "my4": (1, -1, 4, "-y 4cm"), "px4": (0, 1, 4, "+x 4cm"), "mx4": (0, -1, 4, "-x 4cm")}
DIMS = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
RESIDUAL_SCALE = 0.1
SAT = 0.99


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def load_agent(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["agent"]
    agent = SACAgent(15, ActionSpec(sd["action_spec"]["dim"], -1, 1, "residual"), SACConfig(device="cpu"))
    agent.load_state_dict(sd)
    agent.actor.eval()
    return agent


@torch.no_grad()
def actor_out(agent, obs):
    """Deterministic residual, pre-tanh mean and std; 6D policies get a zero gripper column."""
    mu, log_std = agent.actor(torch.as_tensor(obs, dtype=torch.float32))
    pad = lambda x: np.pad(x, ((0, 0), (0, 7 - x.shape[1])))
    return pad(torch.tanh(mu).numpy()), pad(mu.numpy()), pad(log_std.exp().numpy())


def evals_dir(out):
    return out / "evals"


def load_eval(out, pol, cond, steps=False):
    f = evals_dir(out) / f"{pol}_{cond}.json"
    d = json.loads(f.read_text())
    if steps:
        s = np.load(str(f) + ".steps.npz")
        return d, {k: s[k] for k in s.files}
    return d


def perturb_steps(d):
    """1-based env step at which the perturbation was applied (np.inf if not applied)."""
    res = []
    for e in d["episodes"]:
        ev = e.get("perturb_events", [])
        res.append(ev[0]["step"] if ev and ev[0]["applied"] else np.inf)
    return np.asarray(res, dtype=float)


def phase_masks(d, s):
    ps = perturb_steps(d)[s["episode"].astype(int)]
    post = s["step"] >= ps
    return {"all": np.ones_like(post), "pre": ~post, "post": post}


# ============================================================== A
def part_a(out, figs):
    rows, cmp_rows, repro = [], [], []
    for pol in POLICIES:
        for c, (axis, direc, dist, label) in CONDS.items():
            f = evals_dir(out) / f"{pol}_{c}.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text())
            eps = d["episodes"]
            n, k = len(eps), sum(bool(e["success"]) for e in eps)
            ev = [e.get("perturb_events", []) for e in eps]
            applied = [x[0] for x in ev if x and x[0]["applied"]]
            rej = [x[0]["rejection"] for x in ev if x and not x[0]["applied"]]
            lo, hi = wilson(k, n)
            rows.append({"policy": pol, "condition": label, "axis": "xy"[axis], "direction": direc,
                         "distance_cm": dist, "episodes": n, "complete": "wall_seconds" in d,
                         "successes": k, "success_rate": rnd(k / n), "wilson95_lo": rnd(lo), "wilson95_hi": rnd(hi),
                         "perturb_applied": len(applied), "perturb_rejected": len(rej),
                         "rejection_reasons": ";".join(sorted(set(rej))),
                         "perturb_never_triggered": sum(not x for x in ev),
                         "trigger_step_mean": rnd(np.mean([a["step"] for a in applied]) if applied else None, 2),
                         "trigger_eef_dist_mean_m": rnd(np.mean([a["eef_distance"] for a in applied]) if applied else None),
                         "displacement_m_mean": rnd(np.mean([a.get("actual_displacement_m", 0) for a in applied]) if applied else None),
                         "unique_initial_states": len({e["initial_state_hash"] for e in eps})})
            bf = evals_dir(out) / f"base_{c}.json"
            if pol != "base" and "wall_seconds" in d and bf.exists() and "wall_seconds" in json.loads(bf.read_text()):
                r = compare(evals_dir(out) / f"base_{c}.json", f)
                cmp_rows.append({"policy": pol, "condition": label, "episodes": r["episodes"],
                                 "base_rate": r["base"]["rate"], "residual_rate": r["residual"]["rate"],
                                 "delta_pp": r["delta_pp"], "gained": r["gained"], "lost": r["lost"],
                                 "mcnemar_exact_p": rnd(r["mcnemar_exact_p"], 6),
                                 "both_applied_episodes": r["both_applied"]["episodes"],
                                 "both_applied_gained": r["both_applied"]["gained"],
                                 "both_applied_lost": r["both_applied"]["lost"],
                                 "both_applied_mcnemar_p": rnd(r["both_applied"]["mcnemar_exact_p"], 6)})
    # reproducibility vs earlier reports (same seeds -> identical per-episode outcomes expected)
    earlier = {("base", "0cm"): "results/eval50_0cm.json", ("base", "py2"): "results/eval50_2cm.json",
               ("base", "py4"): "results/eval50_4cm.json"}
    for d_cm, c in ((0, "0cm"), (2, "py2"), (4, "py4")):
        for pol, tag in (("ck7k", "ck7k"), ("final", "final")):
            earlier[(pol, c)] = f"results/residual_s42_eval/eval50_{tag}_{d_cm}cm.json"
    for (pol, c), old in earlier.items():
        f = evals_dir(out) / f"{pol}_{c}.json"
        if not f.exists() or not (ROOT / old).exists():
            continue
        a = json.loads(f.read_text())["episodes"]
        b = json.loads((ROOT / old).read_text())["episodes"]
        same = sum(x["success"] == y["success"] and x["initial_state_hash"] == y["initial_state_hash"]
                   and x["env_steps"] == y["env_steps"] for x, y in zip(a, b))
        repro.append({"policy": pol, "condition": CONDS[c][3], "earlier_report": old, "episodes": min(len(a), len(b)),
                      "identical_success_hash_and_length": same,
                      "earlier_successes": sum(bool(y["success"]) for y in b),
                      "new_successes": sum(bool(x["success"]) for x in a)})
    write_csv(out / "A_direction_success.csv", rows)
    write_csv(out / "A_direction_vs_base_mcnemar.csv", cmp_rows)
    write_csv(out / "A_reproducibility_check.csv", repro)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [v[3] for v in CONDS.values()]
    plot_success(rows, POLICIES, labels, figs / "A_direction_success.png",
                 "Direction generalization (training: +y 2cm or random direction, seed 42)")
    focus = [p for p in ("base", "v6drand_final", "fixa_final", "fixa_val") if p in POLICIES]
    if "fixa_final" in focus:
        plot_success(rows, focus, labels, figs / "A_fixed_entropy_coef.png",
                     "Fixed entropy coefficient 0.01 vs auto-tuned (6D + random direction, seed 42)")
    print(f"A: {len(rows)} runs, {len(cmp_rows)} comparisons, repro checks {len(repro)}")


def plot_success(rows, policies, labels, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    POLICIES = policies
    w = .8 / len(POLICIES)
    for i, pol in enumerate(POLICIES):
        xs, ys, lo, hi = [], [], [], []
        for j, lab in enumerate(labels):
            r = next((r for r in rows if r["policy"] == pol and r["condition"] == lab), None)
            if r:
                xs.append(j + (i - (len(POLICIES) - 1) / 2) * w)
                ys.append(r["success_rate"])
                lo.append(r["success_rate"] - r["wilson95_lo"])
                hi.append(r["wilson95_hi"] - r["success_rate"])
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=2, color=PCOLOR[pol], label=PNAME[pol])
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate (50 ep, Wilson 95% CI)")
    ax.set_title(title)
    ax.legend(ncol=min(len(POLICIES), 4), fontsize=8)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ============================================================== B
def residual_stats(u, base, execd):
    """u: residual in [-1,1]; executed delta = scale*u with gripper zeroed."""
    delta = RESIDUAL_SCALE * u.copy()
    delta[:, 6] = 0
    m6 = u[:, :6]
    norm2 = (m6 ** 2).sum(1)
    return {"n_steps": len(u),
            "delta_l2_mean": rnd(np.linalg.norm(delta, axis=1).mean()),
            "delta_l2_std": rnd(np.linalg.norm(delta, axis=1).std()),
            "sat_frac_any_dim0_5": rnd((np.abs(m6) > SAT).any(1).mean()),
            # share of the residual's power explained by one constant vector (1 = state-independent)
            "constancy_ratio": rnd((m6.mean(0) ** 2).sum() / max(norm2.mean(), 1e-12)),
            "exec_minus_base_l2_mean": rnd(np.linalg.norm(execd - base, axis=1).mean()) if execd is not None else None,
            "clip_frac_dim0_5": rnd((np.abs(base[:, :6] + delta[:, :6]) > 1).mean()) if base is not None else None}


def dim_rows(tag, u, base=None, execd=None):
    rows = []
    for i, name in enumerate(DIMS):
        r = {**tag, "dim": name, "u_mean": rnd(u[:, i].mean()), "u_std": rnd(u[:, i].std()),
             "sat_frac": rnd((np.abs(u[:, i]) > SAT).mean()),
             "delta_mean": rnd(RESIDUAL_SCALE * u[:, i].mean() if i < 6 else 0.)}
        if base is not None:
            r["base_mean"] = rnd(base[:, i].mean())
            r["exec_minus_base_mean"] = rnd((execd[:, i] - base[:, i]).mean())
            r["exec_minus_base_abs_mean"] = rnd(np.abs(execd[:, i] - base[:, i]).mean())
        rows.append(r)
    return rows


def lda_cv(X, y, groups, folds=5, seed=0):
    """Linear discriminant analysis, episode-grouped CV. Returns step accuracy."""
    ug = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(ug)
    correct = total = 0
    classes = np.unique(y)
    for f in range(folds):
        test_g = ug[f::folds]
        te = np.isin(groups, test_g)
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        mu_x, sd_x = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt, Xe = (X[tr] - mu_x) / sd_x, (X[te] - mu_x) / sd_x
        means = np.stack([Xt[y[tr] == c].mean(0) for c in classes])
        cent = Xt - means[np.searchsorted(classes, y[tr])]
        cov = cent.T @ cent / max(len(Xt) - len(classes), 1)
        cov += 1e-2 * np.eye(cov.shape[0])
        inv = np.linalg.inv(cov)
        w = means @ inv
        b = -0.5 * (w * means).sum(1)
        pred = classes[np.argmax(Xe @ w.T + b, axis=1)]
        correct += (pred == y[te]).sum()
        total += te.sum()
    return correct / total if total else None


def part_b(out, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agents = {k: load_agent(p) for k, p in CKPTS.items()}
    stat_rows, summary_rows, cond_rows, cf_rows, cf_summary, sens_rows, dec_rows = [], [], [], [], [], [], []

    # ---- on-policy residuals (each policy's own rollouts)
    onpol = {}
    for pol in CKPTS:
        for c, (_, _, _, label) in CONDS.items():
            if not (evals_dir(out) / f"{pol}_{c}.json.steps.npz").exists():
                continue
            d, s = load_eval(out, pol, c, steps=True)
            onpol[(pol, c)] = (d, s)
            for ph, m in phase_masks(d, s).items():
                if m.sum() == 0:
                    continue
                tag = {"source": "on_policy", "policy": pol, "condition": label, "phase": ph}
                stat_rows += dim_rows(tag, s["u"][m], s["base_action"][m], s["exec_action"][m])
                summary_rows.append({**tag, **residual_stats(s["u"][m], s["base_action"][m], s["exec_action"][m])})
            post = phase_masks(d, s)["post"]
            if post.sum():
                u = s["u"][post]
                cond_rows.append({"source": "on_policy", "policy": pol, "condition": label,
                                  "n_post_steps": int(post.sum()),
                                  **{f"u_{n}_mean": rnd(u[:, i].mean()) for i, n in enumerate(DIMS[:6])},
                                  **{f"base_{n}_mean": rnd(s["base_action"][post][:, i].mean()) for i, n in enumerate(DIMS[:3])}})

    # ---- counterfactual: both actors on the SAME frozen-base observations
    base_obs = {}
    for c, (_, _, _, label) in CONDS.items():
        if not (evals_dir(out) / f"base_{c}.json.steps.npz").exists():
            continue
        d, s = load_eval(out, "base", c, steps=True)
        base_obs[c] = (d, s)
        masks = phase_masks(d, s)
        outs = {}
        for pol, ag in agents.items():
            u, mu, std = actor_out(ag, s["obs"])
            outs[pol] = u
            for ph, m in masks.items():
                if m.sum() == 0:
                    continue
                tag = {"source": "same_base_obs", "policy": pol, "condition": label, "phase": ph}
                stat_rows += dim_rows(tag, u[m])
                summary_rows.append({**tag, **residual_stats(u[m], None, None),
                                     "pre_tanh_abs_mu_mean": rnd(np.abs(mu[m][:, :6]).mean()),
                                     "policy_std_mean": rnd(std[m][:, :6].mean())})
            if masks["post"].sum():
                up = u[masks["post"]]
                cond_rows.append({"source": "same_base_obs", "policy": pol, "condition": label,
                                  "n_post_steps": int(masks["post"].sum()),
                                  **{f"u_{n}_mean": rnd(up[:, i].mean()) for i, n in enumerate(DIMS[:6])}})
        a, b = outs["ck7k"][:, :6], outs["final"][:, :6]
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
        cf_summary.append({"condition": label, "n_steps": len(a), "cosine_ck7k_vs_final_mean": rnd(cos.mean()),
                           "ck7k_l2_mean": rnd(np.linalg.norm(a, axis=1).mean()),
                           "final_l2_mean": rnd(np.linalg.norm(b, axis=1).mean())})

    # ---- state sensitivity: replace one part of the observation by its pooled mean
    if base_obs:
        obs_all = np.concatenate([s["obs"] for _, s in base_obs.values()])
        mean_obs = obs_all.mean(0)
        for pol, ag in agents.items():
            u0 = actor_out(ag, obs_all)[0][:, :6]
            for part, sl in (("proprio8", slice(0, 8)), ("base_action7", slice(8, 15)), ("all15", slice(0, 15))):
                o = obs_all.copy()
                o[:, sl] = mean_obs[sl]
                u1 = actor_out(ag, o)[0][:, :6]
                sens_rows.append({"policy": pol, "replaced_by_mean": part, "n_steps": len(o),
                                  "mean_abs_change_u": rnd(np.abs(u1 - u0).mean()),
                                  "rel_change_l2": rnd(np.linalg.norm(u1 - u0, axis=1).mean() /
                                                       max(np.linalg.norm(u0, axis=1).mean(), 1e-9)),
                                  "u_l2_mean": rnd(np.linalg.norm(u0, axis=1).mean()),
                                  "u_std_across_states_mean": rnd(u0.std(0).mean())})

    # ---- H3: can the observation tell the perturbation direction apart? (frozen-base rollouts)
    for dist in (4, 2):
        cs = [c for c, v in CONDS.items() if v[2] == dist and c in base_obs]
        if len(cs) < 2:
            continue
        for window, lo, hi in (("pre", -10**9, 0), ("post_0_20", 0, 20), ("post_all", 0, 10**9)):
            for feat, sl in (("all15", slice(0, 15)), ("proprio8", slice(0, 8)), ("base_action7", slice(8, 15))):
                X, y, g = [], [], []
                for ci, c in enumerate(cs):
                    d, s = base_obs[c]
                    ps = perturb_steps(d)[s["episode"].astype(int)]
                    rel = s["step"] - ps
                    m = np.isfinite(ps) & (rel >= lo) & (rel < hi)
                    X.append(s["obs"][m][:, sl])
                    y.append(np.full(m.sum(), ci))
                    g.append(s["episode"][m].astype(int))
                X, y, g = np.concatenate(X), np.concatenate(y), np.concatenate(g)
                dec_rows.append({"distance_cm": dist, "classes": "/".join(CONDS[c][3] for c in cs),
                                 "window": window, "features": feat, "n_steps": len(y),
                                 "chance": rnd(1 / len(cs)), "lda_cv_accuracy": rnd(lda_cv(X, y, g))})

    write_csv(out / "B_residual_action_stats_by_dim.csv", stat_rows)
    write_csv(out / "B_residual_summary.csv", summary_rows)
    write_csv(out / "B_condition_mean_residual_post_perturb.csv", cond_rows)
    write_csv(out / "B_same_obs_policy_comparison.csv", cf_summary)
    write_csv(out / "B_state_sensitivity.csv", sens_rows)
    write_csv(out / "B_direction_decodability_lda.csv", dec_rows)

    # ---- figures
    if base_obs:
        pool = np.concatenate([s["obs"] for _, s in base_obs.values()])
        U = {pol: actor_out(ag, pool)[0] for pol, ag in agents.items()}
        fig, axes = plt.subplots(2, 7, figsize=(18, 5), constrained_layout=True)
        for i, name in enumerate(DIMS):
            for row, pol, col in ((0, "ck7k", "#1f77b4"), (1, "final", "#d62728")):
                ax = axes[row, i]
                ax.hist(U[pol][:, i], bins=np.linspace(-1, 1, 41), color=col)
                ax.axvline(U[pol][:, i].mean(), color="k", lw=1)
                ax.set_title(f"{pol} {name}" + (" (no effect)" if name == "grip" else ""), fontsize=9)
                ax.set_yticks([])
        fig.suptitle("Deterministic residual u=tanh(mu) per dimension on identical frozen-base observations "
                     f"(all 9 conditions, {len(pool)} states); black line = mean")
        fig.savefig(figs / "B_residual_hist_same_obs.png", dpi=130)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), constrained_layout=True)
        x = np.arange(7)
        w = .8 / len(U)
        for j, pol in enumerate(U):
            off = (j - (len(U) - 1) / 2) * w
            axes[0].bar(x + off, U[pol].mean(0), w, yerr=U[pol].std(0), capsize=2, color=PCOLOR[pol], label=PNAME[pol])
            axes[1].bar(x + off, (np.abs(U[pol]) > SAT).mean(0), w, color=PCOLOR[pol], label=PNAME[pol])
        for ax, t in ((axes[0], "mean ± std of u (residual before x0.1 scale)"), (axes[1], f"fraction |u| > {SAT}")):
            ax.set_xticks(x, DIMS)
            ax.set_title(t)
            ax.grid(axis="y", alpha=.25)
            ax.legend(fontsize=8)
        axes[0].axhline(0, color="k", lw=.5)
        fig.savefig(figs / "B_residual_bias_saturation.png", dpi=140)
        plt.close(fig)

    dirs4 = [c for c in ("py4", "my4", "px4", "mx4")]
    pols4 = [p for p in CKPTS if all((p, c) in onpol for c in dirs4)]
    if pols4:
        fig, axes = plt.subplots(1, len(pols4), figsize=(6 * len(pols4), 3.8), constrained_layout=True,
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, pol in zip(axes, pols4):
            for j, c in enumerate(dirs4):
                d, s = onpol[(pol, c)]
                m = phase_masks(d, s)["post"]
                ax.bar(np.arange(3) + (j - 1.5) * .2, s["u"][m][:, :3].mean(0), .2, label=CONDS[c][3])
            ax.set_xticks(range(3), DIMS[:3])
            ax.axhline(0, color="k", lw=.5)
            ax.set_title(f"{PNAME[pol]}: mean translational residual after perturbation", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=.25)
        fig.savefig(figs / "B_post_perturb_mean_by_direction.png", dpi=140)
        plt.close(fig)

        fig, axes = plt.subplots(1, len(pols4), figsize=(6 * len(pols4), 3.8), constrained_layout=True,
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, pol in zip(axes, pols4):
            for c in ["0cm"] + dirs4:
                if (pol, c) not in onpol:
                    continue
                d, s = onpol[(pol, c)]
                ps = perturb_steps(d)[s["episode"].astype(int)]
                rel = s["step"] - np.where(np.isfinite(ps), ps, np.nan)
                nrm = RESIDUAL_SCALE * np.linalg.norm(s["u"][:, :6], axis=1)
                bins = np.arange(-30, 151, 5)
                mids, vals = [], []
                for a0, a1 in zip(bins[:-1], bins[1:]):
                    m = (rel >= a0) & (rel < a1)
                    if m.sum() > 20:
                        mids.append((a0 + a1) / 2)
                        vals.append(nrm[m].mean())
                ax.plot(mids, vals, label=CONDS[c][3])
            ax.axvline(0, color="k", lw=.6, ls="--")
            ax.set_xlabel("steps relative to perturbation")
            ax.set_title(f"{PNAME[pol]}: executed residual norm ||0.1*u[0:6]||", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=.25)
        fig.savefig(figs / "B_residual_norm_around_perturbation.png", dpi=140)
        plt.close(fig)
    print(f"B: {len(stat_rows)} dim rows, {len(summary_rows)} summaries, {len(dec_rows)} decodability rows")


# ============================================================== C
def part_c(out, figs):
    for tag, run in RUNS.items():
        if (run / "summary.json").exists():
            training_stability(run, out, figs, "" if tag == "s42" else f"{tag}_")
    compare_runs(out, figs)


def compare_runs(out, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    runs = {t: r for t, r in RUNS.items() if (r / "summary.json").exists()}
    if len(runs) < 2:
        return
    names = {"s42": "7D fixed +y (baseline)", "v6d": "6D fixed +y", "v6drand": "6D random dir",
             "v6drandfix": "6D random dir, fixed coef 0.01"}
    fig, axes = plt.subplots(1, 4, figsize=(20, 3.8), constrained_layout=True)
    for t, r in runs.items():
        tr, ep = pd.read_csv(r / "train.csv"), pd.read_csv(r / "episodes.csv")
        buf = np.load(r / "checkpoints/replay_buffer.npz")
        a = buf["actions"][:int(buf["meta"][1])]
        axes[0].plot(ep.env_step, ep.success.rolling(20, min_periods=5).mean(), label=names[t])
        axes[1].plot(tr.env_step, tr.entropy, label=names[t])
        axes[2].plot(tr.env_step, tr.alpha, label=names[t])
        n6 = RESIDUAL_SCALE * np.linalg.norm(a[:, :6], axis=1)
        idx = np.arange(500, len(n6) + 1, 500)
        axes[3].plot(idx, [n6[i - 500:i].mean() for i in idx], label=names[t])
    for ax, t in zip(axes, ("training success (rolling 20 ep)", "policy entropy",
                            "SAC entropy coefficient (log)", "sampled executed residual norm (500 steps)")):
        ax.set_title(t)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[2].set_yscale("log")
    fig.savefig(figs / "C_runs_comparison.png", dpi=130)
    plt.close(fig)


def training_stability(RUN, out, figs, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    tr = pd.read_csv(RUN / "train.csv")
    ep = pd.read_csv(RUN / "episodes.csv")
    buf = np.load(RUN / "checkpoints/replay_buffer.npz")
    n = int(buf["meta"][1])
    acts, rews, obs = buf["actions"][:n], buf["rewards"][:n].ravel(), buf["obs"][:n]
    cfg = json.loads((RUN / "config.json").read_text())
    lr, a0 = cfg["agent"]["alpha_lr"], cfg["agent"]["init_alpha"]
    target_entropy = -cfg["agent"]["target_entropy_scale"] * cfg["action_spec"]["dim"]

    # checkpoint sweep on a fixed observation set: warmup transitions (zero residual, training condition)
    fixed = obs[:1000]
    sweep = []
    for k in range(1000, 20001, 1000):
        p = RUN / f"checkpoints/ckpt_{k:08d}.pt"
        sd = torch.load(p, map_location="cpu", weights_only=False)
        a_dim = sd["agent"]["action_spec"]["dim"]
        ag = SACAgent(15, ActionSpec(a_dim, -1, 1, "residual"), SACConfig(device="cpu"))
        ag.load_state_dict(sd["agent"])
        u, mu, std = actor_out(ag, fixed)
        with torch.no_grad():
            _, logp, _ = ag.actor.sample(torch.as_tensor(fixed))
        sweep.append({"env_step": k, "grad_updates": sd["trainer"]["grad_updates"],
                      "entropy_coef": rnd(float(sd["agent"]["log_alpha"].exp()), 6),
                      "entropy_coef_maxrate_prediction": rnd(a0 * np.exp(-lr * sd["trainer"]["grad_updates"]), 6),
                      "policy_entropy_est": rnd(-logp.mean().item()),
                      "det_u_abs_mean_dim0_5": rnd(np.abs(u[:, :6]).mean()),
                      "det_sat_frac_dim0_5": rnd((np.abs(u[:, :6]) > SAT).mean()),
                      "det_delta_l2_mean": rnd(RESIDUAL_SCALE * np.linalg.norm(u[:, :6], axis=1).mean()),
                      "pre_tanh_abs_mu_mean": rnd(np.abs(mu[:, :6]).mean()),
                      "policy_std_mean_dim0_5": rnd(std[:, :6].mean()),
                      "constancy_ratio": rnd((u[:, :6].mean(0) ** 2).sum() / max((u[:, :6] ** 2).sum(1).mean(), 1e-12)),
                      **{f"det_u_mean_{d}": rnd(u[:, i].mean()) for i, d in enumerate(DIMS)}})

    # 1000-step windows
    win = []
    for w0 in range(0, n, 1000):
        w1 = w0 + 1000
        e = ep[(ep.env_step > w0) & (ep.env_step <= w1)]
        t = tr[(tr.env_step > w0) & (tr.env_step <= w1)]
        a = acts[w0:w1]
        win.append({"env_step_from": w0, "env_step_to": w1, "episodes_ended": len(e),
                    "train_success_rate": rnd(e.success.mean()) if len(e) else None,
                    "mean_episode_length": rnd(e.length.mean(), 1) if len(e) else None,
                    "success_transitions": int((rews[w0:w1] > 0).sum()),
                    "entropy_coef_end": rnd(t.alpha.iloc[-1], 6) if "alpha" in t and t.alpha.notna().any() else None,
                    "policy_entropy_mean": rnd(t.entropy.mean()) if "entropy" in t and t.entropy.notna().any() else None,
                    "actor_loss_mean": rnd(t.actor_loss.mean()) if t.actor_loss.notna().any() else None,
                    "critic_loss_mean": rnd(t.critic_loss.mean()) if t.critic_loss.notna().any() else None,
                    "q1_mean": rnd(t.q1_mean.mean()) if t.q1_mean.notna().any() else None,
                    "sampled_sat_frac_dim0_5": rnd((np.abs(a[:, :6]) > SAT).mean()),
                    "sampled_sat_frac_grip": rnd((np.abs(a[:, 6]) > SAT).mean()) if a.shape[1] > 6 else None,
                    "sampled_delta_l2_mean": rnd(RESIDUAL_SCALE * np.linalg.norm(a[:, :6], axis=1).mean()),
                    **{f"sampled_u_mean_{d}": rnd(a[:, i].mean()) for i, d in enumerate(DIMS[:a.shape[1]])}})
    write_csv(out / f"{prefix}C_checkpoint_sweep.csv", sweep)
    write_csv(out / f"{prefix}C_training_windows.csv", win)
    write_csv(out / f"{prefix}C_entropy_coef_vs_maxrate.csv", [
        {"env_step": int(r.env_step), "grad_updates": int(r.grad_updates), "entropy_coef": r.alpha,
         "maxrate_prediction": a0 * np.exp(-lr * r.grad_updates), "policy_entropy": r.entropy,
         "target_entropy": target_entropy} for r in tr.itertuples() if not np.isnan(r.alpha)])

    sw = pd.DataFrame(sweep)
    wn = pd.DataFrame(win)
    roll = ep.success.rolling(20, min_periods=5).mean()
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), constrained_layout=True)
    A = axes.ravel()
    A[0].plot(ep.env_step, roll, label="rolling 20 episodes")
    A[0].bar(wn.env_step_to - 500, wn.train_success_rate, 900, alpha=.3, label="per 1000 steps")
    A[0].set_title("Training success (stochastic policy)")
    A[0].set_ylim(0, 1.05)
    A[1].plot(tr.env_step, tr.actor_loss, label="actor")
    A[1].set_title("Actor loss")
    A[2].plot(tr.env_step, tr.critic_loss, label="critic")
    A[2].set_yscale("log")
    A[2].set_title("Critic loss (log)")
    A[3].plot(tr.env_step, tr.alpha, label="observed")
    A[3].plot(tr.env_step, a0 * np.exp(-lr * tr.grad_updates), "--", label="max-rate Adam decay a0*exp(-lr*t)")
    A[3].set_yscale("log")
    A[3].set_title("SAC entropy coefficient (temperature)")
    A[4].plot(tr.env_step, tr.entropy, label="policy entropy (-log pi, batch)")
    A[4].axhline(target_entropy, color="r", ls="--", label=f"target {target_entropy:g}")
    A[4].set_title("Policy entropy (squashed Gaussian; may be < 0)")
    A[5].plot(tr.env_step, tr.q1_mean, label="Q1 mean")
    A[5].plot(tr.env_step, tr.q_target_mean, label="target Q mean")
    A[5].set_title("Q statistics")
    A[6].plot(wn.env_step_to, wn.sampled_sat_frac_dim0_5, "o-", label="sampled |u|>0.99 (dims 0-5)")
    A[6].plot(sw.env_step, sw.det_sat_frac_dim0_5, "s-", label="deterministic, fixed obs")
    A[6].set_title("Residual saturation")
    A[7].plot(wn.env_step_to, wn.sampled_delta_l2_mean, "o-", label="sampled ||0.1*u||")
    A[7].plot(sw.env_step, sw.det_delta_l2_mean, "s-", label="deterministic, fixed obs")
    A[7].set_title("Executed residual norm")
    A[8].bar(wn.env_step_to - 500, wn.success_transitions, 900)
    A[8].set_title("Success (reward=1) transitions per 1000 steps")
    for ax in A:
        ax.axvline(7000, color="gray", ls=":", lw=1)
        ax.set_xlabel("environment steps")
        ax.grid(alpha=.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle(f"Residual-SAC training stability: {RUN.name} (dotted line = 7k)")
    fig.savefig(figs / f"{prefix}C_training_stability.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for i, d in enumerate(DIMS[:a_dim]):
        ax.plot(sw.env_step, sw[f"det_u_mean_{d}"], marker="o", ms=3, label=d + (" (no effect)" if d == "grip" else ""))
    ax.axhline(0, color="k", lw=.5)
    ax.axvline(7000, color="gray", ls=":")
    ax.set_title("Deterministic residual mean per dimension across checkpoints (fixed warmup observations)")
    ax.set_xlabel("checkpoint env step")
    ax.legend(ncol=7, fontsize=8)
    ax.grid(alpha=.25)
    fig.savefig(figs / f"{prefix}C_checkpoint_residual_drift.png", dpi=140)
    plt.close(fig)
    print(f"C: {len(sweep)} checkpoints, {len(win)} windows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parts", nargs="+", default=["A", "B", "C"])
    p.add_argument("--out", default="results/residual_s42_diagnostics")
    args = p.parse_args()
    out = ROOT / args.out
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        {"A": part_a, "B": part_b, "C": part_c}[part](out, figs)


if __name__ == "__main__":
    main()
