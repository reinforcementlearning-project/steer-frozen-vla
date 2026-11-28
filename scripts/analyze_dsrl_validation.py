"""Validation analysis of the DSRL runs (docs/dsrl-validation.md).

Reads results/dsrl_train_v1_s{42,43,44}/ and results/dsrl_validation/reference/ and writes the
tables under docs/evidence/dsrl_validation/. Runs on CPU (the critics are small MLPs).

Usage: "$PY" -B scripts/analyze_dsrl_validation.py [--residual-results DIR]
  --residual-results  results directory holding residual_6d_rand_fixa001_2cm_s<seed>_20k/train.csv;
                      only used for the Q-value reference table, skipped when absent.
"""
import argparse
import csv
import json
import math
import statistics as st
import sys
from math import comb
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl.sac import SACAgent  # noqa: E402

SEEDS = (42, 43, 44)
CONDS = ("+x2", "-x2", "+y2", "-y2")
CANDIDATES = list(range(200, 2001, 100))  # decision index of ckpt 2k..20k control steps
PHASES = (("2k (0 updates)", 200, 200), ("3k-5k", 300, 500), ("6k-10k", 600, 1000),
          ("11k-15k", 1100, 1500), ("16k-20k", 1600, 2000))
OUT = ROOT / "docs/evidence/dsrl_validation"


def run_dir(seed):
    return ROOT / f"results/dsrl_train_v1_s{seed}"


def load_report(path):
    """Episodes of a validation report by condition; checks the report against the rule."""
    d = json.loads(Path(path).read_text())
    if "wall_seconds" not in d or d["post_perturb"] != "policy" or d["args"]["stochastic"]:
        raise SystemExit(f"incomplete or mismatched report: {path}")
    episodes = {c: d["conditions"][c]["episodes"] for c in CONDS}
    for c, eps in episodes.items():
        if [e["seed"] for e in eps] != list(range(1000, 1010)):
            raise SystemExit(f"{path} {c}: not eval seed 1000, 10 episodes")
    return episodes


def flat(episodes):
    return [e for c in CONDS for e in episodes[c]]


def outcomes(episodes):
    return [int(bool(e["success"])) for e in flat(episodes)]


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n)


def write_csv(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {name} ({len(rows)} rows)")


def latent_stats(episodes):
    zs = np.array([z for e in flat(episodes) for z in e["latents"] if z is not None], dtype=float)
    return {"latent_mean_abs": round(float(np.abs(zs).mean()), 3),
            "latent_offset_norm": round(float(np.linalg.norm(zs.mean(0))), 3),  # norm of the per-dim mean
            "latent_sd": round(float(zs.std(0).mean()), 3),
            "latent_frac_near_bound": round(float((np.abs(zs) > 2.4).mean()), 3)}


def checkpoint_table(base):
    rows, reports = [], {}
    for seed in SEEDS:
        run = run_dir(seed)
        train = {int(r["env_step"]): r for r in csv.DictReader(open(run / "train.csv"))}
        for n in CANDIDATES:
            eps = reports[(seed, n)] = load_report(run / f"validation/ckpt_{n:08d}.json")
            v = outcomes(eps)
            row = {"seed": seed, "decision": n, "control_steps": int(train[n]["control_steps"]),
                   "grad_updates": int(train[n]["grad_updates"])}
            for c in CONDS:
                row[f"succ_{c}"] = sum(bool(e["success"]) for e in eps[c])
            row["score"] = sum(v)
            row["perturbation_applied"] = sum(e["dynamic_libero"].get("status") == "applied" for e in flat(eps))
            row["base_only_fails"] = sum(a > b for a, b in zip(v, base))
            row["base_only_succeeds"] = sum(b > a for a, b in zip(v, base))
            row.update(latent_stats(eps))
            rows.append(row)
        best = max((r for r in rows if r["seed"] == seed), key=lambda r: (r["score"], r["decision"]))
        sel = json.loads((run / "validation/selection.json").read_text())
        if sel["selected_decision"] != best["decision"] or sel["selected_score"] != best["score"]:
            raise SystemExit(f"seed {seed}: selection.json does not match the reports")
    write_csv("V_scores_by_checkpoint.csv", rows)
    return rows, reports


def phase_table(rows):
    out = []
    for name, lo, hi in PHASES:
        sel = [r for r in rows if lo <= r["decision"] <= hi]
        row = {"phase": name, "checkpoints": len(sel), "mean_score": round(st.mean(r["score"] for r in sel), 1)}
        for c in CONDS:
            row[f"mean_succ_{c}"] = round(st.mean(r[f"succ_{c}"] for r in sel), 2)
        row["mean_latent_offset_norm"] = round(st.mean(r["latent_offset_norm"] for r in sel), 2)
        out.append(row)
    write_csv("V_phase_means.csv", out)
    trained = [r for r in rows if r["grad_updates"] > 0]
    x, y = [r["latent_offset_norm"] for r in trained], [r["score"] for r in trained]
    print(f"corr(score, latent offset norm) over {len(trained)} trained checkpoints: {np.corrcoef(x, y)[0, 1]:.2f}")
    for lo, hi in ((0, 0.9), (0.9, 1.3), (1.3, 99)):
        sel = [r["score"] for r in trained if lo <= r["latent_offset_norm"] < hi]
        print(f"  offset norm [{lo}, {hi}): n={len(sel)} mean score={st.mean(sel):.1f}")


def reference_tables(base_eps, gauss_eps, reports, rows):
    base = outcomes(base_eps)
    ref = []
    for name, eps in (("base", base_eps), ("gaussian", gauss_eps)):
        row = {"policy": name}
        for c in CONDS:
            row[f"succ_{c}"] = sum(bool(e["success"]) for e in eps[c])
        row["score"] = sum(outcomes(eps))
        row["perturbation_applied"] = sum(e["dynamic_libero"].get("status") == "applied" for e in flat(eps))
        ok = [e["env_steps"] for e in flat(eps) if e["success"]]
        row["mean_steps_when_success"] = round(st.mean(ok), 1)
        row["failures_at_280_steps"] = sum(e["env_steps"] >= 280 for e in flat(eps) if not e["success"])
        ref.append(row)
    write_csv("V_reference.csv", ref)

    selected = {s: max((r for r in rows if r["seed"] == s), key=lambda r: (r["score"], r["decision"]))["decision"]
                for s in SEEDS}
    selected_trained = {s: max((r for r in rows if r["seed"] == s and r["grad_updates"] > 0),
                               key=lambda r: (r["score"], r["decision"]))["decision"] for s in SEEDS}
    policies = [("gaussian", gauss_eps)]
    for s in SEEDS:
        policies.append((f"s{s}_selected_ckpt{selected[s] * 10 // 1000}k", reports[(s, selected[s])]))
        if selected_trained[s] != selected[s]:
            policies.append((f"s{s}_selected_trained_ckpt{selected_trained[s] * 10 // 1000}k",
                             reports[(s, selected_trained[s])]))
    policies += [(f"s{s}_final_ckpt20k", reports[(s, 2000)]) for s in SEEDS]
    paired = []
    for name, eps in policies:
        v = outcomes(eps)
        wins, losses = sum(a > b for a, b in zip(v, base)), sum(b > a for a, b in zip(v, base))
        fl = flat(eps)
        paired.append({"policy": name, "score": sum(v), "base_score": sum(base), "base_only_fails": wins,
                       "base_only_succeeds": losses, "sign_test_p": round(sign_test(wins, losses), 4),
                       "mean_steps_when_success": round(st.mean(e["env_steps"] for e in fl if e["success"]), 1),
                       "failures": len(fl) - sum(v),
                       "failures_at_280_steps": sum(e["env_steps"] >= 280 for e in fl if not e["success"])})
    write_csv("V_paired_vs_base.csv", paired)

    per_episode = []
    for ci, c in enumerate(CONDS):
        for i in range(10):
            k = ci * 10 + i
            row = {"condition": c, "episode": i, "seed": 1000 + i, "base": base[k], "gaussian": outcomes(gauss_eps)[k],
                   "frac_of_57_checkpoints_succeeding": round(st.mean(outcomes(reports[key])[k] for key in reports), 2)}
            for s in SEEDS:
                row[f"s{s}_selected"] = outcomes(reports[(s, selected[s])])[k]
                row[f"s{s}_final"] = outcomes(reports[(s, 2000)])[k]
            per_episode.append(row)
    write_csv("V_episode_outcomes.csv", per_episode)


def critic_table():
    """Q of the saved critics on the replay-buffer transitions, split by how the episode ended."""
    torch.manual_seed(0)
    rows = []
    for seed in SEEDS:
        run = run_dir(seed)
        buf = np.load(run / "checkpoints/replay_buffer.npz")
        obs, next_obs, act, done = (torch.as_tensor(buf[k]) for k in ("obs", "next_obs", "actions", "dones"))
        n = len(obs)
        success = np.full(n, -1)
        last = np.zeros(n, bool)
        for r in csv.DictReader(open(run / "episodes.csv")):
            end, length, ok = int(r["env_step"]), int(r["length"]), int(r["success"])
            if end > n:
                continue
            success[end - length:end] = ok
            last[end - 1] = True
            if int(done[end - 1, 0]) != ok:
                raise SystemExit(f"seed {seed}: done flag and episode outcome disagree at decision {end}")
        for ck in range(300, 2001, 100):
            agent = SACAgent.from_checkpoint(str(run / f"checkpoints/ckpt_{ck:08d}.pt"), device="cpu")
            with torch.no_grad():
                q = torch.min(*agent.critic(obs, agent._to_norm(act))).numpy()[:, 0]
                vs, ents = [], []
                for _ in range(8):  # soft value of the next state under the current policy
                    na, logp, _ = agent.actor.sample(next_obs)
                    vs.append((torch.min(*agent.critic(next_obs, na)) - agent.alpha.detach() * logp).numpy()[:, 0])
                    ents.append(-logp.numpy()[:, 0])
                v, entropy = np.mean(vs, 0), float(np.mean(ents))
            seen = np.arange(n) < ck  # transitions collected up to this checkpoint
            mean = lambda m: round(float(q[m & seen].mean()), 3)  # noqa: E731
            timeout = last & (success == 0) & seen
            rows.append({"seed": seed, "decision": ck, "control_steps_approx": ck * 10,
                         "q_success_episode_transitions": mean(success == 1),
                         "q_failed_episode_transitions": mean(success == 0),
                         "q_final_transition_success": mean(last & (success == 1)),
                         "q_final_transition_timeout": mean(timeout),
                         "soft_value_of_state_at_timeout": round(float(v[timeout].mean()), 3),
                         "frac_states_with_soft_value_above_1": round(float((v[seen] > 1).mean()), 3),
                         "policy_entropy": round(entropy, 2), "alpha": round(float(agent.alpha), 4),
                         "entropy_bonus_perpetuity": round(float(agent.alpha) * entropy * 0.99 / 0.01, 2)})
    write_csv("Q_critic_by_outcome.csv", rows)


def training_table():
    rows = []
    for seed in SEEDS:
        run = run_dir(seed)
        summary = json.loads((run / "summary.json").read_text())
        eps = list(csv.DictReader(open(run / "episodes.csv")))
        train = [r for r in csv.DictReader(open(run / "train.csv")) if r["q1_mean"] not in ("", "nan")]
        late = [r for r in train if int(r["env_step"]) >= 1500]
        row = {"seed": seed, "control_steps": summary["control_steps"], "decisions": summary["decisions"],
               "grad_updates": summary["grad_updates"], "vla_calls": summary["vla_calls"],
               "episodes": len(eps), "training_successes": sum(int(e["success"]) for e in eps),
               "entropy_min": round(min(float(r["entropy"]) for r in train), 2),
               "entropy_final": round(float(train[-1]["entropy"]), 2),
               "q1_final": round(float(train[-1]["q1_mean"]), 2),
               "critic_loss_late": round(st.mean(float(r["critic_loss"]) for r in late), 4)}
        for lo, hi in ((256, 500), (500, 1000), (1000, 1500), (1500, 2100)):
            sel = [r for r in train if lo <= int(r["env_step"]) < hi]
            row[f"q1_mean_decisions_{lo}_{hi}"] = round(st.mean(float(r["q1_mean"]) for r in sel), 2)
        rows.append(row)
    write_csv("T_training_summary.csv", rows)


def residual_reference(results):
    rows = []
    for seed in SEEDS:
        path = Path(results).expanduser() / f"residual_6d_rand_fixa001_2cm_s{seed}_20k/train.csv"
        if not path.exists():
            print(f"skipped Residual reference: {path} not found")
            return
        train = [r for r in csv.DictReader(open(path)) if r.get("q1_mean") not in (None, "", "nan")]
        for lo, hi in ((1000, 5000), (5000, 10000), (10000, 15000), (15000, 20001)):
            sel = [r for r in train if lo <= int(r["env_step"]) < hi]
            entropy = st.mean(float(r["entropy"]) for r in sel)
            rows.append({"seed": seed, "env_steps": f"{lo}-{hi - 1 if hi > 20000 else hi}",
                         "q1_mean": round(st.mean(float(r["q1_mean"]) for r in sel), 2),
                         "policy_entropy": round(entropy, 2), "alpha": round(float(sel[-1]["alpha"]), 4),
                         "entropy_bonus_perpetuity": round(float(sel[-1]["alpha"]) * entropy * 0.99 / 0.01, 2)})
    write_csv("Q_residual_reference.csv", rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--residual-results")
    args = p.parse_args()
    base_eps = load_report(ROOT / "results/dsrl_validation/reference/base_val.json")
    gauss_eps = load_report(ROOT / "results/dsrl_validation/reference/gaussian_val.json")
    rows, reports = checkpoint_table(outcomes(base_eps))
    phase_table(rows)
    reference_tables(base_eps, gauss_eps, reports, rows)
    training_table()
    critic_table()
    if args.residual_results:
        residual_reference(args.residual_results)
    assert not math.isnan(sum(r["score"] for r in rows))


if __name__ == "__main__":
    main()
