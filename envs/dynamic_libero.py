"""Dynamic LIBERO v2: algorithm-agnostic object perturbation around a LeRobot LiberoEnv.

Extends the v1 rule (envs/perturbation.py, kept unchanged) with
  * timing windows on the real EEF-target distance (legacy / early / middle / late),
  * an optional VLA action-chunk phase gate and per-event chunk-phase logging,
  * per-episode, seed-reproducible direction and magnitude sampling,
  * an evaluation-only initial-position shift of the target (spatial generalization),
  * an explicit reason for every episode without an applied perturbation.

With ``timing="legacy"``, one fixed or seed-drawn direction and ``chunk_phase=None``
the simulator calls, their order and the moved pose are identical to v1, so legacy
results reproduce exactly (tests/test_dynamic_libero_v2.py, docs/dynamic_libero_v2.md).

Simulator coordinates are used only to schedule and validate the disturbance; they
are never added to the observation returned to the learning policy.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np

from envs.dynamic_libero_config import DIRECTIONS, DynamicLiberoConfig, chunk_phase_index

DIRECTION_SEED_SALT = 0x5EED   # same stream as envs/perturbation.py random directions
DISTANCE_SEED_SALT = 0xD157
SPATIAL_SEED_SALT = 0x51A7
BLOCKERS = ("robot_contact", "object_lifted", "chunk_phase_mismatch")

# ---------------------------------------------------------------- info contract (stable)
# Keys added to the `info` dict of every reset()/step() (reset: only "dynamic_libero").
# The first five are the v1 keys (envs/perturbation.py) so existing code keeps working.
INFO_KEYS = ("perturb_events", "perturb_step", "perturb_dist", "perturb_axis", "perturb_sign",
             "dynamic_libero")
# info["dynamic_libero"]: per-episode status; the value in the episode's last step is final.
STATUS_KEYS = ("config", "split", "episode_seed", "choice", "spatial", "status")
STATUS_VALUES = ("disabled", "not_triggered", "applied", "rejected")
# extra keys: applied/rejected -> reason, step, chunk_phase, eef_distance, diagnostics
#             not_triggered    -> reason, diagnostics
NOT_TRIGGERED_REASONS = ("not_checked_yet_or_episode_ended_before_earliest_step", "window_not_reached",
                         "window_skipped_between_steps", "blocked_robot_contact", "blocked_object_lifted",
                         "blocked_chunk_phase_mismatch")
REJECTION_REASONS = ("conservative_workspace_boundary", "new_penetration_or_robot_contact")
# info["perturb_events"][0] (at most one event per episode)
EVENT_KEYS = ("step", "eef_distance", "applied", "rejection", "before", "after", "direction", "distance_m",
              "timing", "window_m", "chunk_phase", "steps_until_replan_obs", "chunk_phase_target",
              "last_reported_replan_step", "chunk_phase_from_reported_replan", "robot_contact_before", "config")
APPLIED_EVENT_KEYS = ("actual_displacement_m", "actual_displacement_vector_m")


def _rng(seed, salt):
    return np.random.default_rng(None if seed is None else [int(seed), salt])


class DynamicLiberoV2:
    def __init__(self, env, config: DynamicLiberoConfig):
        self.env, self.config = env, config
        self.pert = config.perturbation
        self.spatial = config.spatial
        self.n = config.n_action_steps
        self.phase_target = chunk_phase_index(self.pert.chunk_phase, self.n) if self.pert else None
        self.replan_steps = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------ helpers
    @property
    def _inner(self):
        return self.env._env.env

    def _robot_contact(self, inner, geom_ids):
        for contact in inner.sim.data.contact[:inner.sim.data.ncon]:
            pair = (int(contact.geom1), int(contact.geom2))
            if not any(x in geom_ids for x in pair):
                continue
            other = pair[1] if pair[0] in geom_ids else pair[0]
            name = inner.sim.model.geom_id2name(other) or ""
            if name.startswith(("robot", "gripper")):
                return True
        return False

    def _penetrating(self, sim, geom_ids, tol):
        return any(c.dist < -tol and (int(c.geom1) in geom_ids or int(c.geom2) in geom_ids)
                   for c in sim.data.contact[:sim.data.ncon])

    def notify_replan(self, env_step: int):
        """Optional hook: a policy wrapper reports the env step whose observation it re-planned from.

        Used only for logging/verification of the chunk-phase assumption (replans at multiples
        of n_action_steps); the perturbation schedule itself does not depend on it.
        """
        self.replan_steps.append(int(env_step))

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.steps, self.attempted, self.events, self.replan_steps = 0, False, [], []
        self.episode_seed = seed
        inner = self._inner
        target = self.config.target
        if target not in inner.objects_dict:
            raise ValueError(f"Target {target} not found: {list(inner.objects_dict)}")
        obj = inner.objects_dict[target]
        self.joint = obj.joints[0]
        if np.asarray(inner.sim.data.get_joint_qpos(self.joint)).shape != (7,):
            raise ValueError("Only free-joint movable objects are supported")
        self.body_id = inner.sim.model.body_name2id(obj.root_body)
        self.geom_ids = {inner.sim.model.geom_name2id(g) for g in obj.contact_geoms}
        self.spatial_record = {"mode": self.spatial.mode, "applied": False}
        if self.spatial.mode != "none":
            obs = self._apply_spatial(seed, inner, obs)
        self.initial_z = float(np.asarray(inner.sim.data.get_joint_qpos(self.joint))[2])
        self.diag = {"steps_checked": 0, "steps_in_window": 0, "first_window_step": None,
                     "min_distance_m": None, "blocked": {b: 0 for b in BLOCKERS}}
        self.episode_choice = None
        if self.pert is not None:
            dirs = self.pert.directions
            d_name = dirs[int(_rng(seed, DIRECTION_SEED_SALT).integers(len(dirs)))] if len(dirs) > 1 else dirs[0]
            dists = self.pert.distances_m
            dist = dists[int(_rng(seed, DISTANCE_SEED_SALT).integers(len(dists)))] if len(dists) > 1 else dists[0]
            axis, sign = DIRECTIONS[d_name]
            self.episode_choice = {"direction": d_name, "axis": axis, "sign": sign, "distance_m": float(dist)}
        info = dict(info)
        info["dynamic_libero"] = self.status()
        return obs, info

    # ------------------------------------------------------------------ spatial
    def _init_region(self, inner):
        pp = inner.parsed_problem
        for pred in pp["initial_state"]:
            if len(pred) == 3 and pred[0] == "on" and pred[1] == self.config.target:
                ranges = pp["regions"].get(pred[2], {}).get("ranges") or []
                if ranges:
                    return pred[2], [float(v) for v in ranges[0]]
        return None, None

    def _apply_spatial(self, seed, inner, obs):
        s, sim = self.spatial, inner.sim
        region_name, rng_xy = self._init_region(inner)
        if region_name is None:
            raise ValueError(f"No BDDL 'on' init region with ranges for {self.config.target}")
        x0, y0, x1, y1 = rng_xy
        m = s.region_margin_m
        state0 = sim.get_state()
        pose0 = np.asarray(sim.data.get_joint_qpos(self.joint)).copy()
        rng = _rng(seed, SPATIAL_SEED_SALT)
        rejections: Dict[str, int] = {}
        # LIBERO's shipped init states do not satisfy every BDDL init predicate (e.g. task 0:
        # cream_cheese, butter and basket "on" predicates are already false). The shift must not
        # break any predicate that held before it, and the target's own predicate must hold.
        preds = [tuple(q) for q in inner.parsed_problem["initial_state"]]
        held_before = {q: bool(inner._eval_predicate(list(q))) for q in preds}
        target_pred = next(q for q in preds if len(q) == 3 and q[0] == "on" and q[1] == self.config.target)
        from lerobot.envs.libero import get_libero_dummy_action
        for attempt in range(s.max_attempts):
            offset = rng.uniform(-s.max_offset_m, s.max_offset_m, size=2)
            reason = None
            new_xy = pose0[:2] + offset
            if np.linalg.norm(offset) < s.min_offset_m:
                reason = "below_min_offset"
            elif not (x0 + m <= new_xy[0] <= x1 - m and y0 + m <= new_xy[1] <= y1 - m):
                reason = "outside_bddl_init_region"
            else:
                moved = pose0.copy()
                moved[:2] = new_xy
                sim.data.set_joint_qpos(self.joint, moved)
                sim.data.set_joint_qvel(self.joint, np.zeros(6))
                sim.forward()
                if self._penetrating(sim, self.geom_ids, 0.003):
                    reason = "penetration"
                elif self._robot_contact(inner, self.geom_ids):
                    reason = "robot_contact"
                else:
                    for _ in range(s.settle_steps):
                        self.env._env.step(get_libero_dummy_action())
                    after = np.asarray(sim.data.get_joint_qpos(self.joint)).copy()
                    if np.linalg.norm(after[:2] - new_xy) > s.settle_tolerance_m or abs(after[2] - pose0[2]) > 0.01:
                        reason = "unstable_after_settle"
                    elif not inner._eval_predicate(list(target_pred)) or any(
                            held_before[q] and not inner._eval_predicate(list(q)) for q in preds):
                        reason = "bddl_initial_predicate_broken"
                    elif inner._check_success():
                        reason = "goal_already_satisfied"
            if reason is None:
                after = np.asarray(sim.data.get_joint_qpos(self.joint)).copy()
                self.spatial_record = {"mode": s.mode, "applied": True, "attempts": attempt + 1,
                                       "rejections": rejections, "offset_m": offset.tolist(),
                                       "before": pose0[:3].tolist(), "after_settle": after[:3].tolist(),
                                       "region": region_name, "region_xy": rng_xy,
                                       "init_predicates_false_before_shift": [list(q) for q in preds if not held_before[q]],
                                       "seed": None if seed is None else int(seed), "salt": SPATIAL_SEED_SALT}
                return self.env._format_raw_obs(inner._get_observations(force_update=True))
            rejections[reason] = rejections.get(reason, 0) + 1
            sim.set_state(state0)
            sim.forward()
        raise RuntimeError(f"No valid initial position for {self.config.target} after {s.max_attempts} "
                           f"attempts (seed {seed}): {rejections}")

    # ------------------------------------------------------------------ step
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        p, choice = self.pert, self.episode_choice
        if p is not None and not self.attempted and not terminated and self.steps >= p.earliest_step:
            inner = self._inner
            sim = inner.sim
            position = np.asarray(sim.data.body_xpos[self.body_id])
            eef = np.asarray(inner._get_observations()["robot0_eef_pos"])
            distance = float(np.linalg.norm(eef - position))
            pose = np.asarray(sim.data.get_joint_qpos(self.joint)).copy()
            contact = self._robot_contact(inner, self.geom_ids)
            d = self.diag
            d["steps_checked"] += 1
            d["min_distance_m"] = distance if d["min_distance_m"] is None else min(d["min_distance_m"], distance)
            lo, hi = p.window
            phase = self.steps % self.n
            if lo <= distance <= hi:
                d["steps_in_window"] += 1
                if d["first_window_step"] is None:
                    d["first_window_step"] = self.steps
                lifted = abs(pose[2] - self.initial_z) >= p.lift_tolerance_m
                if contact:
                    d["blocked"]["robot_contact"] += 1
                elif lifted:
                    d["blocked"]["object_lifted"] += 1
                elif self.phase_target is not None and phase != self.phase_target:
                    d["blocked"]["chunk_phase_mismatch"] += 1
                else:
                    obs = self._attempt(inner, sim, pose, distance, contact, phase, choice)
        info.update(perturb_events=list(self.events), perturb_step=self.events[0]["step"] if self.events else -1,
                    perturb_dist=choice["distance_m"] if choice else 0.0,
                    perturb_axis=choice["axis"] if choice else -1, perturb_sign=choice["sign"] if choice else 0,
                    dynamic_libero=self.status())
        return obs, reward, terminated, truncated, info

    def _attempt(self, inner, sim, pose, distance, contact, phase, choice):
        p = self.pert
        self.attempted = True
        velocity = np.asarray(sim.data.get_joint_qvel(self.joint)).copy()
        moved = pose.copy()
        moved[choice["axis"]] += choice["sign"] * choice["distance_m"]
        last_replan = self.replan_steps[-1] if self.replan_steps else None
        event = {"step": self.steps, "config": {**asdict(p), "axis": choice["axis"], "direction": choice["sign"],
                                                 "distance_m": choice["distance_m"]},
                 "before": pose[:3].tolist(), "after": moved[:3].tolist(),
                 "eef_distance": distance, "robot_contact_before": contact,
                 "applied": False, "rejection": None,
                 "timing": p.timing if p.window_m is None else "custom", "window_m": list(p.window),
                 "direction": choice["direction"], "distance_m": choice["distance_m"],
                 "chunk_phase": phase, "steps_until_replan_obs": (self.n - phase) % self.n,
                 "chunk_phase_target": self.phase_target,
                 "last_reported_replan_step": last_replan,
                 "chunk_phase_from_reported_replan": None if last_replan is None else (self.steps - last_replan) % self.n}
        if np.max(np.abs(moved[:2])) > p.workspace_limit_m:
            event["rejection"] = "conservative_workspace_boundary"
        else:
            sim.data.set_joint_qpos(self.joint, moved)
            sim.data.set_joint_qvel(self.joint, velocity if choice["distance_m"] == 0 else np.zeros(6))
            sim.forward()
            if self._penetrating(sim, self.geom_ids, p.penetration_tolerance_m) or self._robot_contact(inner, self.geom_ids):
                sim.data.set_joint_qpos(self.joint, pose)
                sim.data.set_joint_qvel(self.joint, velocity)
                sim.forward()
                event["rejection"] = "new_penetration_or_robot_contact"
            else:
                event["applied"] = True
                actual = np.asarray(sim.data.get_joint_qpos(self.joint))[:3] - pose[:3]
                event["actual_displacement_m"] = float(np.linalg.norm(actual))
                event["actual_displacement_vector_m"] = actual.tolist()
        self.events.append(event)
        # Force fresh camera/state observations before the next policy query.
        return self.env._format_raw_obs(inner._get_observations(force_update=True))

    # ------------------------------------------------------------------ status
    def status(self) -> Dict[str, Any]:
        """Per-episode perturbation status; the value in the last step's info is final."""
        out: Dict[str, Any] = {"config": self.config.name, "split": self.config.split,
                               "episode_seed": None if self.episode_seed is None else int(self.episode_seed),
                               "choice": self.episode_choice, "spatial": self.spatial_record}
        if self.pert is None:
            out.update(status="disabled")
            return out
        d = self.diag
        out.update(diagnostics={k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()})
        if self.events:
            e = self.events[0]
            out.update(status="applied" if e["applied"] else "rejected", reason=e["rejection"],
                       step=e["step"], chunk_phase=e["chunk_phase"], eef_distance=e["eef_distance"])
            return out
        lo, hi = self.pert.window
        if d["steps_checked"] == 0:
            reason = "not_checked_yet_or_episode_ended_before_earliest_step"
        elif d["steps_in_window"] == 0:
            reason = ("window_not_reached" if d["min_distance_m"] > hi else "window_skipped_between_steps")
        else:
            reason = "blocked_" + max(BLOCKERS, key=lambda b: d["blocked"][b])
        out.update(status="not_triggered", reason=reason)
        return out

    def close(self):
        self.env.close()
