"""Evaluate a DSRL-SAC noise policy (or its controls) in Dynamic LIBERO v2.

Policies:
  base      Frozen Base: the VLA's own seeded Gaussian noise. Same noise stream and
            init states as scripts/run_dynamic_libero_v2.py, so episodes pair up.
  gaussian  z ~ N(0, 1) clipped to the latent bound: the warm-up prior. Shows what
            the latent parameterisation alone does, before any learning.
  <path>    SAC checkpoint from train_dsrl.py (deterministic unless --stochastic);
            latent layout and observation mode are read from the run directory.

--post-perturb base hands control back to the Frozen Base noise from the first
decision after the object was moved (the "intervention removed" ablation).

Examples:
  python scripts/eval_dsrl.py --policy base --set primary --out results/dsrl_eval/base_primary.json
  python scripts/eval_dsrl.py --policy results/dsrl_s42/checkpoints/latest.pt --set primary \
      --out results/dsrl_eval/s42_primary.json
  python scripts/eval_dsrl.py --policy results/dsrl_s42/checkpoints/latest.pt --set primary \
      --condition=+x2 --condition=-x2 --condition=+y2 --condition=-y2 --eval-seed 1000 --episodes 10 \
      --out results/dsrl_eval/s42_validation.json        # checkpoint selection rule
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.dynamic_libero_config import config_to_dict, load_eval_set, make_dynamic_libero
from models.noise_wrapper import FILLS, NoiseLatentSpec, NoiseSteeredSmolVLA
from policies.noise_policy import OBS_MODES, NoiseLatentEnv
from rl import SACAgent


def wilson(successes, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [centre - half, centre + half]


def summarize(episodes):
    n = len(episodes)
    wins = sum(e["success"] for e in episodes)
    return {"episodes": n, "successes": wins, "success_rate": wins / n if n else float("nan"),
            "success_wilson95": wilson(wins, n),
            "perturbation_applied": sum(e["dynamic_libero"].get("status") == "applied" for e in episodes),
            "vla_calls": sum(e["vla_calls"] for e in episodes),
            "decisions": sum(e["decisions"] for e in episodes),
            "steered_decisions": sum(e["steered_decisions"] for e in episodes)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="base | gaussian | path to a train_dsrl.py checkpoint")
    p.add_argument("--set", required=True, help="eval set name in configs/dynamic_libero_v2.yaml")
    p.add_argument("--condition", action="append", help="condition token(s) of the set; default: all")
    p.add_argument("--episodes", type=int)
    p.add_argument("--eval-seed", type=int)
    p.add_argument("--post-perturb", choices=("policy", "base"), default="policy")
    p.add_argument("--stochastic", action="store_true", help="sample from the SAC policy instead of its mean")
    p.add_argument("--latent-dims", type=int, default=NoiseLatentSpec.dims)
    p.add_argument("--latent-steps", type=int, default=NoiseLatentSpec.steps)
    p.add_argument("--latent-bound", type=float, default=NoiseLatentSpec.bound)
    p.add_argument("--latent-fill", choices=FILLS, default=NoiseLatentSpec.fill)
    p.add_argument("--obs-mode", choices=sorted(OBS_MODES), default="proprio")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = ROOT / args.out
    if out.exists():
        p.error(f"Refusing to overwrite existing report: {out}")
    overrides = {k: v for k, v in (("episodes", args.episodes), ("eval_seed", args.eval_seed)) if v is not None}
    configs = load_eval_set(args.set, **overrides)
    for token in args.condition or []:
        if token not in configs:
            p.error(f"{token!r} is not in {args.set}: {list(configs)}")
    configs = {t: c for t, c in configs.items() if not args.condition or t in args.condition}

    agent, run_config = None, None
    spec = NoiseLatentSpec(args.latent_dims, args.latent_steps, args.latent_bound, args.latent_fill)
    obs_mode = args.obs_mode
    if args.policy not in ("base", "gaussian"):
        checkpoint = Path(args.policy)
        run_config = json.loads((checkpoint.resolve().parents[1] / "base_config.json").read_text())
        spec = NoiseLatentSpec(**run_config["latent"])
        obs_mode = run_config["observation"]
        agent = SACAgent.from_checkpoint(str(checkpoint), device="cuda")
        if agent.action_spec.dim != spec.latent_dim or agent.obs_dim != OBS_MODES[obs_mode]:
            p.error("checkpoint does not match the latent/observation settings of its run directory")
    n_values = {c.n_action_steps for c in configs.values()}
    if len(n_values) != 1:
        p.error("conditions use different n_action_steps")
    artifacts = json.loads((ROOT / "results/artifacts.json").read_text())
    vla = NoiseSteeredSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"], n_values.pop(), 0, spec)
    report = {"args": vars(args), "policy": args.policy, "post_perturb": args.post_perturb,
              "latent": spec.to_dict(), "observation": obs_mode, "artifacts": artifacts,
              "training_run": run_config, "conditions": {}}
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for token, cfg in configs.items():
        # A fresh simulator per condition: LIBERO hands out init states in reset order.
        sim = make_dynamic_libero(cfg)
        env = NoiseLatentEnv(sim, vla, max_steps=cfg.max_steps, obs_mode=obs_mode)
        block = {"config": config_to_dict(cfg), "episodes": []}
        report["conditions"][token] = block
        try:
            for episode in range(cfg.episodes):
                seed = cfg.episode_seed(episode)
                rng = np.random.default_rng([seed, 0xD5121])
                obs, info = env.reset(seed=seed)
                inner = sim._env.env
                state_hash = hashlib.sha256(np.asarray(inner.sim.get_state().flatten()).tobytes()).hexdigest()
                state_id = sim.init_state_id - 1
                latents, steered_flags, calls, done = [], [], 0, False
                while not done:
                    if args.policy == "base" or (args.post_perturb == "base" and env.perturbed):
                        latent = None
                    elif args.policy == "gaussian":
                        latent = env.action_spec.clip(rng.standard_normal(env.action_spec.dim).astype(np.float32))
                    else:
                        latent = agent.act(obs, deterministic=not args.stochastic)
                    obs, _, terminated, truncated, info = env.step(latent)
                    latents.append(None if latent is None else np.round(latent, 4).tolist())
                    steered_flags.append(latent is not None)
                    calls += info["vla_calls"]
                    done = terminated or truncated
                record = {"episode": episode, "seed": seed, "success": info["success"], "env_steps": env.steps,
                          "decisions": env.decisions, "steered_decisions": sum(steered_flags), "vla_calls": calls,
                          "initial_state_id": state_id, "initial_state_hash": state_hash, "noise_seed": seed,
                          "dynamic_libero": info["dynamic_libero"], "perturb_events": info.get("perturb_events", []),
                          "latents": latents}
                block["episodes"].append(record)
                block["summary"] = summarize(block["episodes"])
                out.write_text(json.dumps(report, indent=2, default=float))
                dl = info["dynamic_libero"]
                print(json.dumps({"condition": token, "episode": episode, "success": record["success"],
                                  "steps": env.steps, "status": dl.get("status"), "step": dl.get("step"),
                                  "steered": f"{sum(steered_flags)}/{env.decisions}"}), flush=True)
        finally:
            env.close()
        s = block["summary"]
        print(f"SUMMARY {token}: {s['successes']}/{s['episodes']} = {s['success_rate']:.3f}", flush=True)
    report["wall_seconds"] = time.perf_counter() - start
    report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"REPORT {out}", flush=True)


if __name__ == "__main__":
    main()
