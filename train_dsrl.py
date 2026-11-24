"""DSRL-SAC: shared SAC trainer + noise-steered Frozen SmolVLA + Dynamic LIBERO v2.

One trainer step is one RL decision (one VLA chunk = ``n_action_steps`` control
steps), so the columns the shared trainer calls ``env_step`` count decisions here.
The budget is still given in control steps, the unit Residual-SAC uses:
``--total-env-steps``, ``--warmup-steps`` and ``--checkpoint-every`` are control
steps and ``control_steps`` is written next to every logged row.

Example (shared training condition, see docs/dynamic_libero_v2.md §7):
    python train_dsrl.py --preset train_v1 --seed 42 --run-dir results/dsrl_train_v1_s42
"""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import torch

from envs.dynamic_libero_config import config_to_dict, load_preset, make_dynamic_libero
from models.noise_wrapper import FILLS, NoiseLatentSpec, NoiseSteeredSmolVLA
from policies.noise_policy import OBS_MODES, NoiseLatentEnv
from rl import OffPolicyTrainer, SACAgent, SACConfig, TrainerConfig, seed_everything
from scripts.training_monitor import LOSS_KEYS

RESUME_KEYS = ("preset", "seed", "total_env_steps", "warmup_steps", "batch_size", "utd_ratio", "gamma",
               "fixed_entropy_coef", "autotune_entropy", "latent_dims", "latent_steps", "latent_bound",
               "latent_fill", "obs_mode")


class ControlStepBudget:
    """Counts control steps and ends the shared trainer's loop once the budget is spent.

    The trainer re-reads ``cfg.total_env_steps`` (in decisions) every iteration, so
    lowering it to the current decision count stops training after this step. The
    last chunk may overshoot the budget by at most ``n_action_steps - 1`` steps.
    Every decision is appended to ``budget.csv`` so a resumed run restores the count.
    """

    def __init__(self, env, budget, path):
        self.env, self.budget, self.path = env, int(budget), Path(path)
        self.total, self.trainer, self.file = 0, None, None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def attach(self, trainer, resumed):
        self.trainer = trainer
        rows = []
        if resumed:
            with self.path.open() as f:
                rows = [r for r in csv.DictReader(f) if int(r["decision"]) <= trainer.env_steps]
            if (int(rows[-1]["decision"]) if rows else 0) != trainer.env_steps:
                raise RuntimeError(f"{self.path} has no row for decision {trainer.env_steps}")
            self.total = int(rows[-1]["control_steps"]) if rows else 0
        self.file = self.path.open("w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=["decision", "control_steps", "vla_calls"])
        self.writer.writeheader()
        self.writer.writerows(rows)
        self.file.flush()
        if self.total >= self.budget:   # resuming a finished run: take no further decision
            trainer.cfg.total_env_steps = trainer.env_steps

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total += info["env_steps"]
        decision = self.trainer.env_steps + 1
        info["control_steps"] = self.total
        self.writer.writerow({"decision": decision, "control_steps": self.total,
                              "vla_calls": self.trainer.vla_calls + info["vla_calls"]})
        self.file.flush()
        if self.total >= self.budget:
            self.trainer.cfg.total_env_steps = decision
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.file is not None:
            self.file.close()
        self.env.close()


class DecisionLogger:
    """Adds control steps to every row and reserves the loss columns before warm-up ends."""

    def __init__(self, logger, budget):
        self.logger, self.budget = logger, budget

    def log(self, stream, row):
        row = {**row, "control_steps": self.budget.total}
        if stream == "train":
            for key in LOSS_KEYS:
                row.setdefault(key, float("nan"))
            if row["grad_updates"] > 0 and not all(math.isfinite(row[k]) for k in LOSS_KEYS if k != "alpha_loss"):
                raise FloatingPointError("Non-finite SAC metrics")
        self.logger.log(stream, row)

    def close(self):
        self.logger.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", default="train_v1", help="training preset in configs/dynamic_libero_v2.yaml")
    p.add_argument("--total-env-steps", type=int, default=20000, help="budget in control steps")
    p.add_argument("--warmup-steps", type=int, default=2560,
                   help="control steps of Gaussian-latent warm-up (raised to batch-size decisions if smaller)")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--utd-ratio", type=int, default=None,
                   help="gradient updates per decision (default: n_action_steps, i.e. about one per control step)")
    p.add_argument("--gamma", type=float, default=0.99, help="discount per decision")
    p.add_argument("--fixed-entropy-coef", type=float, default=0.01,
                   help="fixed SAC entropy coefficient (Residual-SAC's final setting)")
    p.add_argument("--autotune-entropy", action="store_true", help="auto-tune the entropy coefficient from 0.1")
    p.add_argument("--latent-dims", type=int, default=NoiseLatentSpec.dims)
    p.add_argument("--latent-steps", type=int, default=NoiseLatentSpec.steps)
    p.add_argument("--latent-bound", type=float, default=NoiseLatentSpec.bound)
    p.add_argument("--latent-fill", choices=FILLS, default=NoiseLatentSpec.fill)
    p.add_argument("--obs-mode", choices=sorted(OBS_MODES), default="proprio_ref")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", default="results/dsrl_s0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--log-every", type=int, default=100, help="control steps (approximate)")
    p.add_argument("--checkpoint-every", type=int, default=1000, help="control steps (approximate)")
    args = p.parse_args()
    if min(args.total_env_steps, args.batch_size, args.warmup_steps) < 1:
        p.error("steps, warmup-steps and batch-size must be positive")
    if not args.autotune_entropy and not args.fixed_entropy_coef > 0:
        p.error("--fixed-entropy-coef must be positive")
    run_dir = Path(args.run_dir)
    if (run_dir / "config.json").exists() and not args.resume:
        p.error("Run already exists; choose another run-dir or --resume")
    cfg = load_preset(args.preset)
    if cfg.split != "train":
        p.error(f"{args.preset} is a {cfg.split!r} preset; train only on a 'train' preset")
    n = cfg.n_action_steps
    if args.utd_ratio is None:
        args.utd_ratio = n
    if args.utd_ratio < 1:
        p.error("utd-ratio must be positive")
    if args.resume:
        original = json.loads((run_dir / "base_config.json").read_text())["args"]
        for key in RESUME_KEYS:
            if original[key] != vars(args)[key]:
                p.error(f"Resume would change {key}; use the original setting")
    seed_everything(args.seed)
    spec = NoiseLatentSpec(args.latent_dims, args.latent_steps, args.latent_bound, args.latent_fill)
    artifacts = json.loads(Path("results/artifacts.json").read_text())
    vla = NoiseSteeredSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"], n, args.seed, spec)
    env = ControlStepBudget(
        NoiseLatentEnv(make_dynamic_libero(cfg), vla, max_steps=cfg.max_steps, obs_mode=args.obs_mode),
        args.total_env_steps, run_dir / "budget.csv")
    sac = SACConfig(device="cuda", gamma=args.gamma,
                    init_alpha=0.1 if args.autotune_entropy else args.fixed_entropy_coef,
                    autotune_alpha=args.autotune_entropy)
    agent = SACAgent(env.obs_dim, env.action_spec, sac)
    # Until the first gradient step the untrained actor would act; keep the prior instead.
    warmup = max(math.ceil(args.warmup_steps / n), args.batch_size)
    config = TrainerConfig(
        total_env_steps=args.total_env_steps,   # upper bound in decisions; ControlStepBudget ends the run
        warmup_steps=warmup, warmup_policy="gaussian", learning_starts=warmup,
        batch_size=args.batch_size, utd_ratio=args.utd_ratio, buffer_capacity=100000, eval_every=0,
        log_every=max(1, args.log_every // n), checkpoint_every=max(1, args.checkpoint_every // n),
        run_dir=str(run_dir), seed=args.seed,
        episode_info_keys=["control_steps", "perturb_step", "perturb_dist", "perturb_axis", "perturb_sign"])
    trainer = OffPolicyTrainer(env, agent, config)
    trainer.logger = DecisionLogger(trainer.logger, env)
    metadata = {"args": vars(args), "artifacts": artifacts, "environment": config_to_dict(cfg),
                "latent": spec.to_dict(), "observation": args.obs_mode,
                "trainer_step_unit": "RL decision (one VLA chunk); control steps are in budget.csv",
                "warmup_decisions": warmup, "vla_budget": "initial reference query charged to first decision",
                "evaluation_during_training": False}
    metadata["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    metadata["source_sha256_lf"] = {
        str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in [Path(__file__).relative_to(Path.cwd())] +
        [q for directory in ("rl", "envs", "models", "policies") for q in sorted(Path(directory).glob("*.py"))]
    }
    (run_dir / ("resume_config.json" if args.resume else "base_config.json")).write_text(json.dumps(metadata, indent=2))
    try:
        if args.resume:
            trainer.resume()
        env.attach(trainer, args.resume)
        actor_before = [x.detach().clone() for x in agent.actor.parameters()]
        metrics = trainer.train()
        report = {"control_steps": env.total, "decisions": trainer.env_steps, "episodes": trainer.episodes,
                  "grad_updates": trainer.grad_updates, "vla_calls": trainer.vla_calls, "metrics": metrics,
                  "actor_changed": any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())),
                  "vla_frozen": all(not x.requires_grad and x.grad is None for x in vla.policy.parameters())}
        (run_dir / ("resume_summary.json" if args.resume else "summary.json")).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        trainer.logger.close()
        env.close()


if __name__ == "__main__":
    main()
