"""Connect the shared SAC trainer to Frozen SmolVLA + (Dynamic) LIBERO."""
import argparse
import json
from pathlib import Path
import torch
from envs.libero_env import make_libero
from envs.perturbation import DynamicLibero, PerturbationConfig
from models.smolvla_wrapper import FrozenSmolVLA
from policies.residual_policy import ResidualEnv
from rl import SACAgent, SACConfig, OffPolicyTrainer, TrainerConfig, seed_everything


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
    args = p.parse_args()
    if min(args.total_env_steps, args.batch_size, args.utd_ratio) < 1:
        p.error("steps, batch-size and utd-ratio must be positive")
    run_dir = Path(args.run_dir)
    if (run_dir / "config.json").exists() and not args.resume:
        p.error("Run already exists; choose another run-dir or --resume")
    if args.resume:
        original = json.loads((run_dir / "base_config.json").read_text())["args"]
        for key in ("suite", "task_id", "n_action_steps", "max_episode_steps", "alpha", "seed", "target", "distance_cm"):
            if original[key] != vars(args)[key]:
                p.error(f"Resume would change {key}; use the original setting")
    seed_everything(args.seed)
    artifacts = json.loads(Path("results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"],
                        args.n_action_steps, args.seed)
    simulator = make_libero(args.suite, args.task_id, max_steps=args.max_episode_steps)
    if args.target:
        simulator = DynamicLibero(simulator, PerturbationConfig(args.target, args.distance_cm / 100))
    env = ResidualEnv(simulator, base, args.alpha, max_steps=args.max_episode_steps)
    agent = SACAgent(env.obs_dim, env.action_spec, SACConfig(device="cuda"))
    config = TrainerConfig(total_env_steps=args.total_env_steps,
        warmup_steps=args.warmup_steps, warmup_policy="zero",
        learning_starts=args.warmup_steps, batch_size=args.batch_size, utd_ratio=args.utd_ratio,
        buffer_capacity=max(10000, args.total_env_steps), eval_every=0,
        log_every=32, checkpoint_every=0, run_dir=str(run_dir), seed=args.seed,
        episode_info_keys=["perturb_step", "perturb_dist"])
    # Keep capacity fixed across short resume checks and ordinary 20k-step runs.
    config.buffer_capacity = 100000
    trainer = OffPolicyTrainer(env, agent, config)
    metadata = {"args": vars(args), "artifacts": artifacts,
                "gripper_residual": False, "observation": "proprioception8+base_action7",
                "vla_budget": "initial query charged to first step", "evaluation_during_training": False}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / ("resume_config.json" if args.resume else "base_config.json")).write_text(json.dumps(metadata, indent=2))
    try:
        if args.resume:
            trainer.resume()
        actor_before = [x.detach().clone() for x in agent.actor.parameters()]
        critic_before = [x.detach().clone() for x in agent.critic.parameters()]
        metrics = trainer.train()
        report = {"env_steps": trainer.env_steps, "grad_updates": trainer.grad_updates,
            "vla_calls": trainer.vla_calls, "metrics": metrics,
            "actor_changed": any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters())),
            "critic_changed": any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters())),
            "vla_frozen": all(not x.requires_grad and x.grad is None for x in base.policy.parameters())}
        (run_dir / ("resume_summary.json" if args.resume else "summary.json")).write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
