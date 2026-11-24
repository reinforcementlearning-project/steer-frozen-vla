"""Run actual frozen-policy episodes, optionally one shared SAC update."""
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
from rl import ActionSpec, ReplayBuffer, SACAgent, SACConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=280)
    p.add_argument("--n-action-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--update", action="store_true")
    p.add_argument("--target")
    p.add_argument("--distance-cm", type=float, default=0)
    p.add_argument("--out", default="results/smoke.json")
    args = p.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        p.error("episodes and max-steps must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    artifacts = json.loads((ROOT / "results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"],
                        args.n_action_steps, args.seed)
    sim_env = make_libero(args.suite, args.task_id, max_steps=args.max_steps)
    if args.target:
        sim_env = DynamicLibero(sim_env, PerturbationConfig(args.target, args.distance_cm / 100))
    env = ResidualEnv(sim_env, base, max_steps=args.max_steps)
    replay = ReplayBuffer(15, 7, max(args.max_steps * args.episodes, 256), args.seed)
    agent = SACAgent(15, ActionSpec(7, -1, 1, "residual"), SACConfig(device="cuda")) if args.update else None
    report = {"args": vars(args), "artifacts": artifacts,
              "checkpoint_n_action_steps": base.checkpoint_n_action_steps,
              "n_action_steps": base.config.n_action_steps, "chunk_size": base.config.chunk_size,
              "normalization": str(base.config.normalization_mapping),
              "frozen": all(not x.requires_grad for x in base.policy.parameters()), "episodes": []}
    start = time.perf_counter()
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            initial_state = np.asarray(sim_env._env.env.sim.get_state().flatten())
            initial_state_hash = hashlib.sha256(initial_state.tobytes()).hexdigest()
            initial_state_id = sim_env.init_state_id - 1
            ret, calls = 0., info["vla_calls"]
            for step in range(args.max_steps):
                residual = agent.act(obs) if agent is not None else np.zeros(7, dtype=np.float32)
                next_obs, reward, terminated, truncated, info = env.step(residual)
                replay.add(obs, residual, reward, next_obs, terminated)
                calls += info["vla_calls"]
                ret += reward
                obs = next_obs
                if step % 25 == 0:
                    print(json.dumps({"episode": episode, "step": step + 1, "vla_calls": calls}), flush=True)
                if terminated or truncated:
                    break
            record = {"episode": episode, "env_steps": step + 1, "success": info["success"],
                      "return": ret, "vla_calls": calls, "trace": base.trace,
                      "initial_state_id": initial_state_id, "initial_state_hash": initial_state_hash,
                      "noise_seed": args.seed + episode,
                      "perturb_events": info.get("perturb_events", [])}
            report["episodes"].append(record)
            out.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in record.items() if k != "trace"}), flush=True)
        if agent is not None:
            actor_before = [x.detach().clone() for x in agent.actor.parameters()]
            critic_before = [x.detach().clone() for x in agent.critic.parameters()]
            report["update"] = agent.update(replay.sample(min(32, len(replay)), "cuda"))
            report["actor_changed"] = any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
            report["critic_changed"] = any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters()))
            report["frozen_grad_none"] = all(x.grad is None for x in base.policy.parameters())
            ckpt = ROOT / "results/residual_smoke.pt"
            agent.save(str(ckpt))
            restored = SACAgent.from_checkpoint(str(ckpt))
            report["reload_action_equal"] = bool(np.allclose(agent.act(obs, True), restored.act(obs, True)))
            assert report["actor_changed"] and report["critic_changed"] and report["frozen_grad_none"] and report["reload_action_equal"]
        report["wall_seconds"] = time.perf_counter() - start
        report["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        out.write_text(json.dumps(report, indent=2))
        print(f"REPORT {out}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
