"""Frozen-SmolVLA rollouts in Dynamic LIBERO v2 (environment validation, no learning).

Runs the frozen base policy (zero residual, the same code path as
scripts/smoke_rollout.py / eval_residual.py --zero-residual) on one config and
records, per episode: outcome, LIBERO init state, perturbation status and reason,
chunk phase, actual displacement, spatial shift, and whether the VLA re-planned at
multiples of n_action_steps. Optionally logs the per-step EEF-target distance,
robot contact and object height (read from sim data, no side effects) for
offline feasibility analysis of timing windows.

Examples:
  python scripts/run_dynamic_libero_v2.py --set primary --condition +y2 --out results/dlv2/primary_py2.json
  python scripts/run_dynamic_libero_v2.py --preset no_perturbation --episodes 50 --eval-seed 0 \
      --log-distance --out results/dlv2/monitor.json
  python scripts/run_dynamic_libero_v2.py --set validation_timing --condition rand2 \
      --override perturbation.timing=late --out results/dlv2/late_rand2.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.dynamic_libero_config import config_to_dict, load_eval_set, load_preset, make_dynamic_libero
from models.smolvla_wrapper import FrozenSmolVLA
from policies.residual_policy import ResidualEnv


def parse_overrides(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def eef_site_pos(inner):
    return np.asarray(inner.sim.data.get_site_xpos(inner.robots[0].controller.eef_name)).copy()


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--set", help="eval set name in configs/dynamic_libero_v2.yaml")
    g.add_argument("--preset", help="preset name in configs/dynamic_libero_v2.yaml")
    p.add_argument("--condition", help="condition token of --set (e.g. +y2, rand2, 0cm)")
    p.add_argument("--override", action="append", help="dotted key=value, e.g. perturbation.timing=late")
    p.add_argument("--episodes", type=int)
    p.add_argument("--eval-seed", type=int)
    p.add_argument("--log-distance", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        p.error(f"Refusing to overwrite existing report: {out}")
    overrides = parse_overrides(args.override)
    if args.episodes is not None:
        overrides["episodes"] = args.episodes
    if args.eval_seed is not None:
        overrides["eval_seed"] = args.eval_seed
    if args.set:
        if not args.condition:
            p.error("--set needs --condition")
        cfg = load_eval_set(args.set, **overrides)[args.condition]
    else:
        cfg = load_preset(args.preset, **overrides)
    if cfg.episodes is None or cfg.eval_seed is None:
        p.error("episodes and eval seed are required (set them in the config or with --episodes/--eval-seed)")
    torch.manual_seed(cfg.eval_seed)
    np.random.seed(cfg.eval_seed)
    artifacts = json.loads((ROOT / "results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"], cfg.n_action_steps, cfg.eval_seed)
    sim = make_dynamic_libero(cfg)
    env = ResidualEnv(sim, base, max_steps=cfg.max_steps)
    report = {"args": vars(args), "config": config_to_dict(cfg), "artifacts": artifacts,
              "policy": "frozen_base_zero_residual", "n_action_steps": base.config.n_action_steps,
              "episodes": []}
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = {k: [] for k in ("episode", "step", "eef_distance", "robot_contact", "object_dz")}
    start = time.perf_counter()
    try:
        for episode in range(cfg.episodes):
            seed = cfg.episode_seed(episode)
            obs, info = env.reset(seed=seed)
            inner = sim._env.env
            initial_state_hash = hashlib.sha256(np.asarray(inner.sim.get_state().flatten()).tobytes()).hexdigest()
            initial_state_id = sim.init_state_id - 1
            z0 = float(np.asarray(inner.sim.data.get_joint_qpos(sim.joint))[2])
            for step in range(cfg.max_steps):
                obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
                if args.log_distance:
                    pos = np.asarray(inner.sim.data.body_xpos[sim.body_id])
                    for k, v in (("episode", episode), ("step", step + 1),
                                 ("eef_distance", np.linalg.norm(eef_site_pos(inner) - pos)),
                                 ("robot_contact", sim._robot_contact(inner, sim.geom_ids)),
                                 ("object_dz", float(np.asarray(inner.sim.data.get_joint_qpos(sim.joint))[2]) - z0)):
                        traces[k].append(v)
                if terminated or truncated:
                    break
            replans = [t["env_step"] for t in base.trace]
            expected = list(range(0, step + 1, cfg.n_action_steps))
            record = {"episode": episode, "seed": seed, "env_steps": step + 1, "success": info["success"],
                      "initial_state_id": initial_state_id, "initial_state_hash": initial_state_hash,
                      "noise_seed": seed, "vla_replan_steps": replans,
                      "replan_cadence_ok": replans[:len(expected)] == expected[:len(replans)]
                                           and all(r % cfg.n_action_steps == 0 for r in replans),
                      "dynamic_libero": info["dynamic_libero"], "perturb_events": info.get("perturb_events", [])}
            report["episodes"].append(record)
            out.write_text(json.dumps(report, indent=2, default=float))
            dl = info["dynamic_libero"]
            print(json.dumps({"episode": episode, "success": record["success"], "steps": record["env_steps"],
                              "status": dl.get("status"), "reason": dl.get("reason"), "step": dl.get("step"),
                              "phase": dl.get("chunk_phase")}), flush=True)
        if args.log_distance:
            np.savez_compressed(str(out) + ".trace.npz", **{k: np.asarray(v) for k, v in traces.items()})
        report["wall_seconds"] = time.perf_counter() - start
        out.write_text(json.dumps(report, indent=2, default=float))
        print(f"REPORT {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
