"""Verify the completed 0/2/4cm evaluations use the same episodes and seeds."""
import argparse
import json
from pathlib import Path
import numpy as np


def validate(paths):
    reports = [json.loads(Path(p).read_text()) for p in paths]
    reference = reports[0]
    assert all("wall_seconds" in r for r in reports), "Evaluation is still running"
    assert all(len(r["episodes"]) == 50 for r in reports), "Require 50 episodes per condition"
    assert all(r["n_action_steps"] == reference["n_action_steps"] for r in reports)
    signatures = lambda r: [(e["initial_state_id"], e["initial_state_hash"], e["noise_seed"]) for e in r["episodes"]]
    expected = signatures(reference)
    assert all(signatures(r) == expected for r in reports), "Initial states or noise seeds differ"
    coverage = []
    for report in reports:
        applied = 0
        for episode in report["episodes"]:
            events = episode["perturb_events"]
            assert len(events) <= 1
            for event in events:
                assert not event["robot_contact_before"]
                if event["applied"]:
                    applied += 1
                    assert np.isclose(event["actual_displacement_m"], event["config"]["distance_m"])
                    assert np.isclose(event["before"][2], event["after"][2])
        coverage.append({"distance_cm": report["args"]["distance_cm"], "applied": applied, "total": 50})
    return {"paired_initial_states_and_noise_seeds": True, "unique_initial_states": len({x[1] for x in expected}),
            "initial_state_ids": [x[0] for x in expected], "coverage": coverage}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    print(json.dumps(validate(args.reports), indent=2))
