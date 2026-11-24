"""Audit actual simulator reward, control cadence and replay transition units."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.libero_env import make_libero
from envs.perturbation import DynamicLibero, PerturbationConfig
from models.smolvla_wrapper import FrozenSmolVLA
from policies.residual_policy import ResidualEnv
from rl import ReplayBuffer


def main():
    artifacts = json.loads((ROOT / "results/artifacts.json").read_text())
    base = FrozenSmolVLA(artifacts["checkpoint"], artifacts["backbone_path"], 10, 41)
    simulator = DynamicLibero(make_libero(), PerturbationConfig("alphabet_soup_1", .02))
    env = ResidualEnv(simulator, base)
    replay = ReplayBuffer(15, 7, 280, 41)
    rows = []
    try:
        obs, _ = env.reset(seed=41)
        for step in range(280):
            action = np.zeros(7, np.float32)
            if step < 21:
                action[0] = .5 if step % 2 == 0 else -.5
            nxt, reward, terminated, truncated, info = env.step(action)
            expected = np.clip(np.asarray(info["base_action"]) + .1*action, -1, 1)
            assert np.allclose(expected, info["exec_action"])
            inner = simulator._env.env
            assert reward == float(inner._check_success())
            replay.add(obs, action, reward, nxt, terminated)
            assert len(replay) == step + 1
            assert replay.dones[step,0] == float(terminated)
            assert replay.rewards[step,0] == reward
            assert np.array_equal(replay.actions[step], action)
            rows.append({"env_step": step+1, "reward": reward, "terminated": terminated,
                "truncated": truncated, "vla_calls": info["vla_calls"],
                "residual_norm": info["residual_norm"], "action": action.tolist()})
            obs = nxt
            if terminated or truncated:
                break
        calls = [t["env_step"] for t in base.trace]
        assert calls == list(range(0, calls[-1]+1, 10))
        report = {"passed": True, "reward": "float(task_success), sparse 0/1, no shaping or residual penalty",
            "n_action_steps": 10, "transition_unit": "one control step", "residual_period_steps": 1,
            "replay_size": len(replay), "vla_query_steps": calls, "success": info["success"],
            "events": info["perturb_events"], "transitions": rows,
            "time_limit_bootstraps": "replay dones stores terminated only"}
        out = ROOT / "results/residual_contract_audit.json"
        out.write_text(json.dumps(report, indent=2))
        print(json.dumps({k:v for k,v in report.items() if k != "transitions"}), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
