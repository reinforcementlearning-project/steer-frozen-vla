"""One pre-contact object translation, with an audit log and collision rejection.

Simulator object coordinates are used only to schedule/validate the disturbance;
they are never returned to the learning policy.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np


@dataclass
class PerturbationConfig:
    target: str
    distance_m: float = 0.02
    earliest_step: int = 10
    approach_min_m: float = 0.05
    approach_max_m: float = 0.18
    axis: int = 1
    direction: int = 1
    # Optional per-episode direction randomization: a tuple of (axis, direction)
    # pairs, one drawn uniformly per episode from the reset seed. Empty = fixed
    # (axis, direction) above, which keeps the original behaviour.
    directions: tuple = ()

    def __post_init__(self):
        if not self.target or not 0 <= self.distance_m <= 0.10:
            raise ValueError("Specify a target and a distance between 0 and 0.10 m")
        self.directions = tuple(tuple(int(v) for v in d) for d in self.directions)
        for axis, direction in ((self.axis, self.direction),) + self.directions:
            if axis not in (0, 1) or direction not in (-1, 1):
                raise ValueError("Translation must be along +/- x or y")


class DynamicLibero:
    def __init__(self, env, config):
        self.env, self.config = env, config

    def __getattr__(self, name):
        return getattr(self.env, name)

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

    def reset(self, *, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self.steps, self.attempted, self.events = 0, False, []
        self.episode_config = self.config
        if self.config.directions:
            # Deterministic in the episode seed (resume-safe); independent of the env's own RNG.
            rng = np.random.default_rng(None if seed is None else [int(seed), 0x5EED])
            axis, direction = self.config.directions[int(rng.integers(len(self.config.directions)))]
            self.episode_config = replace(self.config, axis=axis, direction=direction)
        inner = self.env._env.env
        if self.config.target not in inner.objects_dict:
            raise ValueError(f"Target {self.config.target} not found: {list(inner.objects_dict)}")
        obj = inner.objects_dict[self.config.target]
        self.joint = obj.joints[0]
        pose = np.asarray(inner.sim.data.get_joint_qpos(self.joint))
        if pose.shape != (7,):
            raise ValueError("Only free-joint movable objects are supported")
        self.initial_z = float(pose[2])
        self.body_id = inner.sim.model.body_name2id(obj.root_body)
        self.geom_ids = {inner.sim.model.geom_name2id(g) for g in obj.contact_geoms}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        cfg = self.episode_config
        if not self.attempted and not terminated and self.steps >= cfg.earliest_step:
            inner = self.env._env.env
            sim = inner.sim
            position = np.asarray(sim.data.body_xpos[self.body_id])
            eef = np.asarray(inner._get_observations()["robot0_eef_pos"])
            distance = float(np.linalg.norm(eef - position))
            pose = np.asarray(sim.data.get_joint_qpos(self.joint)).copy()
            contact = self._robot_contact(inner, self.geom_ids)
            if cfg.approach_min_m <= distance <= cfg.approach_max_m and not contact and abs(pose[2] - self.initial_z) < 0.015:
                self.attempted = True
                velocity = np.asarray(sim.data.get_joint_qvel(self.joint)).copy()
                moved = pose.copy()
                moved[cfg.axis] += cfg.direction * cfg.distance_m
                event = {"step": self.steps, "config": asdict(cfg),
                         "before": pose[:3].tolist(), "after": moved[:3].tolist(),
                         "eef_distance": distance, "robot_contact_before": contact,
                         "applied": False, "rejection": None}
                if np.max(np.abs(moved[:2])) > 0.45:
                    event["rejection"] = "conservative_workspace_boundary"
                else:
                    sim.data.set_joint_qpos(self.joint, moved)
                    sim.data.set_joint_qvel(self.joint, velocity if cfg.distance_m == 0 else np.zeros(6))
                    sim.forward()
                    penetrating = any(c.dist < -0.003 and
                        (int(c.geom1) in self.geom_ids or int(c.geom2) in self.geom_ids)
                        for c in sim.data.contact[:sim.data.ncon])
                    if penetrating or self._robot_contact(inner, self.geom_ids):
                        sim.data.set_joint_qpos(self.joint, pose)
                        sim.data.set_joint_qvel(self.joint, velocity)
                        sim.forward()
                        event["rejection"] = "new_penetration_or_robot_contact"
                    else:
                        event["applied"] = True
                        event["actual_displacement_m"] = float(np.linalg.norm(
                            np.asarray(sim.data.get_joint_qpos(self.joint))[:3] - pose[:3]))
                self.events.append(event)
                # Force fresh camera/state observations before the next policy query.
                obs = self.env._format_raw_obs(inner._get_observations(force_update=True))
        info.update(perturb_events=list(self.events), perturb_step=self.events[0]["step"] if self.events else -1,
                    perturb_dist=cfg.distance_m, perturb_axis=cfg.axis,
                    perturb_sign=cfg.direction)
        return obs, reward, terminated, truncated, info

    def close(self):
        self.env.close()
