"""Evaluate a trained residual SAC actor on top of the frozen SmolVLA.

Mirrors scripts/smoke_rollout.py episode-for-episode (same reset seeds, noise
seeds, perturbation and report schema) so results pair directly with the frozen
base evals (results/eval50_*.json). Only the residual differs: the SAC actor's
deterministic action (tanh of the mean) instead of zeros. Read-only with
respect to the shared SAC code and the training run directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.libero_env import make_libero
from envs.perturbation import DynamicLibero, PerturbationConfig
from models.smolvla_wrapper import FrozenSmolVLA
from policies.residual_policy import ResidualEnv
from rl import SACAgent, SACConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="trainer checkpoint (e.g. <run>/checkpoints/latest.pt)")
    p.add_argument("--zero-residual", action="store_true",
                   help="frozen-base control: residual = 0 (same code path as smoke_rollout.py)")
    p.add_argument("--axis", type=int, default=1, help="perturbation axis (0=x, 1=y); default matches training")
    p.add_argument("--direction", type=int, default=1, help="perturbation sign (+1/-1); default matches training")
    p.add_argument("--scale-override", type=float, default=None,
                   help="sanity check: run the trained actor but execute residual scale VALUE (e.g. 0) instead of --alpha")
    p.add_argument("--residual-off-after-perturb", action="store_true",
                   help="ablation: residual = 0 from the first step after the perturbation is applied")
    p.add_argument("--save-steps", action="store_true",
                   help="also write <out>.steps.npz with per-step obs/residual/base/executed actions")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--n-action-steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stochastic", action="store_true", help="sample instead of deterministic mean action")
    p.add_argument("--target")
    p.add_argument("--distance-cm", type=float, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        p.error("episodes and max-steps must be positive")
    out = ROOT / args.out
    if out.exists():
        p.error(f"Refusing to overwrite existing report: {out}")
    if bool(args.checkpoint) == args.zero_residual:
        p.error("give exactly one of --checkpoint or --zero-residual")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False) if ckpt_path else None
    run_cfg_path = ckpt_path.parent.parent / "base_config.json" if ckpt_path else None
    if run_cfg_path is not None and run_cfg_path.exists():
        run_args = json.loads(run_cfg_path.read_text())["args"]
        for key in ("suite", "task_id", "n_action_steps", "alpha"):
            if run_args[key] != vars(args)[key]:
                p.error(f"{key}={vars(args)[key]} differs from training ({run_args[key]})")

    artifacts = json.loads((ROOT / "results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"],
                        args.n_action_steps, args.seed)
    sim_env = make_libero(args.suite, args.task_id, max_steps=args.max_steps)
    if args.target:
        sim_env = DynamicLibero(sim_env, PerturbationConfig(args.target, args.distance_cm / 100,
                                                            axis=args.axis, direction=args.direction))
    residual_dims = int(state["agent"]["action_spec"]["dim"]) if state is not None else 7
    exec_scale = args.alpha if args.scale_override is None else args.scale_override
    env = ResidualEnv(sim_env, base, exec_scale, max_steps=args.max_steps, residual_dims=residual_dims)
    agent = None
    if state is not None:
        agent = SACAgent(env.obs_dim, env.action_spec, SACConfig(device="cuda"))
        agent.load_state_dict(state["agent"])
        agent.actor.eval()

    report = {"args": vars(args), "artifacts": artifacts,
              "checkpoint": str(ckpt_path) if ckpt_path else "frozen_base_zero_residual",
              "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest() if ckpt_path else None,
              "checkpoint_trainer": {k: v for k, v in state["trainer"].items() if k != "config"} if state else None,
              "deterministic": not args.stochastic, "residual_dims": residual_dims,
              "executed_residual_scale": exec_scale,
              "residual_off_after_perturb": args.residual_off_after_perturb,
              "checkpoint_n_action_steps": base.checkpoint_n_action_steps,
              "n_action_steps": base.config.n_action_steps, "chunk_size": base.config.chunk_size,
              "normalization": str(base.config.normalization_mapping),
              "frozen": all(not x.requires_grad for x in base.policy.parameters()), "episodes": []}
    start = time.perf_counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = {k: [] for k in ("episode", "step", "obs", "u", "base_action", "exec_action")}
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            initial_state = np.asarray(sim_env._env.env.sim.get_state().flatten())
            initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
            initial_state_id = sim_env.init_state_id - 1
            ret, calls, norms = 0., info["vla_calls"], []
            perturbed = False
            for step in range(args.max_steps):
                if agent is None or (args.residual_off_after_perturb and perturbed):
                    residual = np.zeros(residual_dims, dtype=np.float32)
                else:
                    with torch.no_grad():
                        residual = agent.act(obs, deterministic=not args.stochastic)
                next_obs, reward, terminated, truncated, info = env.step(residual)
                perturbed = perturbed or any(e["applied"] for e in info.get("perturb_events", []))
                if args.save_steps:
                    for k, v in (("episode", episode), ("step", step), ("obs", obs), ("u", np.pad(residual, (0, 7 - residual_dims))),
                                 ("base_action", info["base_action"]), ("exec_action", info["exec_action"])):
                        steps[k].append(np.asarray(v, dtype=np.float32))
                calls += info["vla_calls"]
                ret += reward
                norms.append(info["residual_norm"])
                obs = next_obs
                if step % 25 == 0:
                    print(json.dumps({"episode": episode, "step": step + 1, "vla_calls": calls}), flush=True)
                if terminated or truncated:
                    break
            record = {"episode": episode, "env_steps": step + 1, "success": info["success"],
                      "return": ret, "vla_calls": calls, "trace": base.trace,
                      "initial_state_id": initial_state_id, "initial_state_hash": initial_state_hash,
                      "noise_seed": args.seed + episode,
                      "residual_norm_mean": float(np.mean(norms)), "residual_norm_max": float(np.max(norms)),
                      "perturb_events": info.get("perturb_events", [])}
            report["episodes"].append(record)
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in record.items() if k != "trace"}), flush=True)
        if args.save_steps:
            np.savez_compressed(str(out) + ".steps.npz", **{k: np.stack(v) for k, v in steps.items()})
        report["wall_seconds"] = time.perf_counter() - start
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        out.write_text(json.dumps(report, indent=2))
        print(f"REPORT {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
