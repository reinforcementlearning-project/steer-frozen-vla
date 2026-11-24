"""Connect the shared SAC trainer to Frozen SmolVLA + (Dynamic) LIBERO."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from envs.libero_env import make_libero
from envs.perturbation import DynamicLibero, PerturbationConfig
from models.smolvla_wrapper import FrozenSmolVLA
from policies.residual_policy import ResidualEnv
from rl import SACAgent, SACConfig, OffPolicyTrainer, TrainerConfig, seed_everything
from scripts.training_monitor import TrainingMonitor, plot_curves


class EpisodeResidualStats:
    """Logging-only wrapper: running per-episode mean of the executed residual norm
    (||alpha * u||, gripper excluded) and the fraction of steps with any |u| > 0.99.
    The trainer copies these info keys into episodes.csv at episode end."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        self._n, self._norm, self._sat = 0, 0., 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._n += 1
        self._norm += float(info.get("residual_norm", 0.))
        self._sat += int(np.any(np.abs(np.asarray(action)) > 0.99))
        info.update(residual_norm_mean=self._norm / self._n, residual_sat_frac=self._sat / self._n)
        return obs, reward, terminated, truncated, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-action-steps", type=int, default=1)
    p.add_argument("--max-episode-steps", type=int, default=280)
    p.add_argument("--total-env-steps", type=int, default=20000)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd-ratio", type=int, default=1)
    p.add_argument("--alpha", type=float, default=.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target")
    p.add_argument("--distance-cm", type=float, default=0)
    p.add_argument("--run-dir", default="results/residual_s0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=1000)
    p.add_argument("--residual-dims", type=int, choices=(6, 7), default=7,
                   help="6 = arm-only residual (SAC target entropy -6); 7 = original (gripper output ignored)")
    p.add_argument("--random-direction", action="store_true",
                   help="draw the perturbation direction per episode from +x/-x/+y/-y (default: fixed +y)")
    p.add_argument("--fixed-entropy-coef", type=float, default=None,
                   help="disable SAC temperature auto-tuning and keep the entropy coefficient fixed at this value "
                        "(SACConfig(autotune_alpha=False, init_alpha=value)); default: auto-tuned from 0.1")
    args = p.parse_args()
    if args.fixed_entropy_coef is not None and not args.fixed_entropy_coef > 0:
        p.error("--fixed-entropy-coef must be positive")
    if min(args.total_env_steps, args.batch_size, args.utd_ratio) < 1:
        p.error("steps, batch-size and utd-ratio must be positive")
    run_dir = Path(args.run_dir)
    if (run_dir / "config.json").exists() and not args.resume:
        p.error("Run already exists; choose another run-dir or --resume")
    if args.resume:
        original = json.loads((run_dir / "base_config.json").read_text())["args"]
        defaults = {"residual_dims": 7, "random_direction": False,  # runs created before these flags
                    "fixed_entropy_coef": None}
        for key in ("suite", "task_id", "n_action_steps", "max_episode_steps", "alpha", "seed", "target", "distance_cm",
                    "residual_dims", "random_direction", "fixed_entropy_coef"):
            if original.get(key, defaults.get(key)) != vars(args)[key]:
                p.error(f"Resume would change {key}; use the original setting")
    seed_everything(args.seed)
    artifacts = json.loads(Path("results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"],
                        args.n_action_steps, args.seed)
    simulator = make_libero(args.suite, args.task_id, max_steps=args.max_episode_steps)
    if args.target:
        directions = ((0, 1), (0, -1), (1, 1), (1, -1)) if args.random_direction else ()
        simulator = DynamicLibero(simulator, PerturbationConfig(args.target, args.distance_cm / 100,
                                                                directions=directions))
    elif args.random_direction:
        p.error("--random-direction needs --target")
    env = ResidualEnv(simulator, base, args.alpha, max_steps=args.max_episode_steps,
                      residual_dims=args.residual_dims)
    sac_config = SACConfig(device="cuda")
    if args.fixed_entropy_coef is not None:
        sac_config = SACConfig(device="cuda", autotune_alpha=False, init_alpha=args.fixed_entropy_coef)
    agent = SACAgent(env.obs_dim, env.action_spec, sac_config)
    env = EpisodeResidualStats(env)
    config = TrainerConfig(total_env_steps=args.total_env_steps,
        warmup_steps=args.warmup_steps, warmup_policy="zero",
        learning_starts=args.warmup_steps, batch_size=args.batch_size, utd_ratio=args.utd_ratio,
        buffer_capacity=max(10000, args.total_env_steps), eval_every=0,
        log_every=args.log_every, checkpoint_every=args.checkpoint_every, run_dir=str(run_dir), seed=args.seed,
        episode_info_keys=["perturb_step", "perturb_dist", "perturb_axis", "perturb_sign",
                           "residual_norm_mean", "residual_sat_frac"])
    # Keep capacity fixed across short resume checks and ordinary 20k-step runs.
    config.buffer_capacity = 100000
    trainer = OffPolicyTrainer(env, agent, config)
    trainer.logger = TrainingMonitor(trainer.logger, run_dir,
                                     optional_keys=("alpha_loss",) if args.fixed_entropy_coef is not None else ())
    metadata = {"args": vars(args), "artifacts": artifacts,
                "gripper_residual": False, "residual_dims": args.residual_dims,
                "sac_target_entropy": agent.target_entropy, "random_direction": args.random_direction,
                "sac_entropy_coef": ("fixed", args.fixed_entropy_coef) if args.fixed_entropy_coef is not None
                else ("auto", sac_config.init_alpha), "observation": "proprioception8+base_action7",
                "vla_budget": "initial query charged to first step", "evaluation_during_training": False}
    metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata["source_sha256_lf"] = {
        str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in [Path(__file__).relative_to(Path.cwd())] +
        [p for directory in ("rl", "envs", "models", "policies", "scripts") for p in Path(directory).glob("*.py")]
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ("resume_config.json" if args.resume else "base_config.json")).write_text(json.dumps(metadata, indent=2))
    try:
        if args.resume:
            trainer.resume()
        actor_before = [x.detach().clone() for x in agent.actor.parameters()]
        critic_before = [x.detach().clone() for x in agent.critic.parameters()]
        metrics = trainer.train()
        plot_curves(run_dir)
        report = {"env_steps": trainer.env_steps, "grad_updates": trainer.grad_updates,
            "vla_calls": trainer.vla_calls, "metrics": metrics,
            "actor_changed": any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())),
            "critic_changed": any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters())),
            "vla_frozen": all(not x.requires_grad and x.grad is None for x in base.policy.parameters())}
        (run_dir / ("resume_summary.json" if args.resume else "summary.json")).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        trainer.logger.close()
        env.close()


if __name__ == "__main__":
    main()
