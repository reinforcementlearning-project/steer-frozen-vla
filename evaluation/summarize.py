"""Aggregate actual episode logs with Wilson 95% intervals and event coverage."""
import argparse
import json
import math
from pathlib import Path


def wilson(successes, total):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Require 0 <= successes <= total, total > 0")
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    margin = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-margin), min(1., center+margin)]


def summarize(path):
    data = json.loads(Path(path).read_text())
    episodes = data["episodes"]
    count = len(episodes)
    successes = sum(e["success"] for e in episodes)
    applied = sum(any(x["applied"] for x in e.get("perturb_events", [])) for e in episodes)
    trace = [t for e in episodes for t in e["trace"]]
    return {"source": str(path), "episodes": count, "successes": successes,
        "success_rate": successes/count, "wilson95": wilson(successes, count),
        "perturbation_applied_episodes": applied, "distance_cm": data["args"].get("distance_cm"),
        "n_action_steps": data["n_action_steps"], "env_steps": sum(e["env_steps"] for e in episodes),
        "vla_calls": sum(e["vla_calls"] for e in episodes),
        "mean_inference_seconds": sum(t["inference_seconds"] for t in trace)/len(trace),
        "chunk_shapes": sorted({str(t["chunk_shape"]) for t in trace}),
        "noise_shapes": sorted({str(t["noise_shape"]) for t in trace}),
        "wall_seconds": data.get("wall_seconds"), "peak_vram_bytes": data.get("peak_vram_bytes"),
        "run_complete": "wall_seconds" in data,
        "unique_initial_states": len({e["initial_state_hash"] for e in episodes if "initial_state_hash" in e}),
        "pilot_only": count < 50}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    print(json.dumps([summarize(p) for p in args.reports], indent=2))
