"""Validate a completed short run before launching a longer experiment."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch


def validate(root, expected_steps):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    config = json.loads((root / "config.json").read_text())
    assert summary["env_steps"] == expected_steps
    assert summary["actor_changed"] and summary["critic_changed"] and summary["vla_frozen"]
    assert summary["grad_updates"] > 0
    with (root / "train.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if int(row["grad_updates"]) > 0:
            assert all(math.isfinite(float(row[k])) for k in ("actor_loss", "critic_loss", "alpha"))
    checkpoint = torch.load(root / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["trainer"]["env_steps"] == expected_steps
    assert checkpoint["trainer"]["grad_updates"] == summary["grad_updates"]
    buffer = np.load(root / "checkpoints/replay_buffer.npz")
    count = int(buffer["meta"][1])
    assert count == expected_steps  # short run has not filled the FIFO
    assert set(np.unique(buffer["rewards"])).issubset({0., 1.})
    assert set(np.unique(buffer["dones"])).issubset({0., 1.})
    assert np.isfinite(buffer["obs"]).all() and np.isfinite(buffer["next_obs"]).all()
    assert np.max(np.abs(buffer["actions"])) <= 1
    assert (root / "learning_curves.png").stat().st_size > 0
    return {"passed": True, "env_steps": count, "updates": summary["grad_updates"],
            "positive_reward_transitions": int(np.sum(buffer["rewards"] > 0)),
            "log_rows": len(rows), "batch_size": config["trainer"]["batch_size"],
            "finite_losses": True, "checkpoint_and_replay_verified": True}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--expected-steps", type=int, required=True)
    args = p.parse_args()
    print(json.dumps(validate(args.run_dir, args.expected_steps), indent=2))
