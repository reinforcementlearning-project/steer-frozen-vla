"""SAC sanity check on a simple continuous-control problem (plan §8-1, Stage 0).

Examples
--------
# dependency-free toy task (2-D point reaching), ~1-2 min on CPU
python scripts/sanity_check_sac.py --env point --steps 15000

# 7-D action (same dimension as the Residual action). Success (radius 0.1 in 7-D) is
# rarely reached in 20k steps; check that the return improves (random ~ -100 -> ~ -25).
python scripts/sanity_check_sac.py --env point --dim 7 --steps 30000

# standard benchmark (needs `pip install gymnasium`); random ≈ -1200, solved ≈ -200
python scripts/sanity_check_sac.py --env pendulum --steps 15000

# resume an interrupted run
python scripts/sanity_check_sac.py --env point --steps 30000 --run-dir results/sanity_point --resume
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl import OffPolicyTrainer, SACAgent, SACConfig, TrainerConfig  # noqa: E402
from rl.toy_env import GymnasiumAdapter, PointReachEnv  # noqa: E402


def make_env(name: str, dim: int):
    if name == "point":
        return PointReachEnv(dim=dim)
    if name == "pendulum":
        import gymnasium as gym
        return GymnasiumAdapter(gym.make("Pendulum-v1"))
    raise ValueError(name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", choices=["point", "pendulum"], default="point")
    p.add_argument("--dim", type=int, default=2, help="action dim for --env point")
    p.add_argument("--steps", type=int, default=15_000)
    p.add_argument("--warmup", type=int, default=1_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    run_dir = args.run_dir or f"results/sanity_{args.env}_s{args.seed}"
    env, eval_env = make_env(args.env, args.dim), make_env(args.env, args.dim)
    agent = SACAgent(env.obs_dim, env.action_spec, SACConfig(device=args.device))
    trainer = OffPolicyTrainer(
        env, agent,
        TrainerConfig(total_env_steps=args.steps, warmup_steps=args.warmup, learning_starts=args.warmup,
                      batch_size=args.batch_size, utd_ratio=args.utd, log_every=1_000,
                      eval_every=max(args.steps // 5, 1), eval_episodes=10,
                      checkpoint_every=max(args.steps // 5, 1), keep_checkpoints=False,
                      seed=args.seed, run_dir=run_dir),
        eval_env=eval_env,
    )
    print(f"device={agent.device} obs_dim={env.obs_dim} action={env.action_spec}")
    if args.resume and os.path.exists(os.path.join(trainer.ckpt_dir, "latest.pt")):
        trainer.resume()
    trainer.train()

    before = trainer.evaluate(20)
    reloaded = SACAgent.from_checkpoint(os.path.join(trainer.ckpt_dir, "latest.pt"), device=args.device)
    trainer.agent = reloaded
    after = trainer.evaluate(20)
    print(f"[final] trained : {before}")
    print(f"[final] reloaded: {after}  (should match: checkpoint reload check)")


if __name__ == "__main__":
    main()
