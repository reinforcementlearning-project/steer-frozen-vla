"""Dependency-free continuous-control env for SAC sanity tests.

PointReach: a point in [-1,1]^d must reach a random goal. The action is a
velocity in [-1,1]^d (scaled by 0.1). This mirrors the LIBERO setting: the
episode has a time limit, a binary ``success`` flag in ``info``, and a
configurable action dimension (7 like Residual, or the DSRL latent size).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .interface import ActionSpec


class PointReachEnv:
    def __init__(self, dim: int = 2, max_steps: int = 50, success_radius: float = 0.1,
                 dense_reward: bool = True) -> None:
        self.dim = dim
        self.max_steps = max_steps
        self.success_radius = success_radius
        self.dense_reward = dense_reward
        self.obs_dim = 2 * dim
        self.action_spec = ActionSpec(dim=dim, low=-1.0, high=1.0, name="velocity")
        self._rng = np.random.default_rng()
        self.pos = np.zeros(dim, dtype=np.float32)
        self.goal = np.zeros(dim, dtype=np.float32)
        self.t = 0

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal - self.pos]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.pos = self._rng.uniform(-1, 1, self.dim).astype(np.float32)
        self.goal = self._rng.uniform(-1, 1, self.dim).astype(np.float32)
        self.t = 0
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        a = self.action_spec.clip(np.asarray(action, dtype=np.float32))
        self.pos = np.clip(self.pos + 0.1 * a, -1.0, 1.0)
        self.t += 1
        dist = float(np.linalg.norm(self.goal - self.pos))
        success = dist < self.success_radius
        reward = (-dist + (10.0 if success else 0.0)) if self.dense_reward else float(success)
        terminated = success
        truncated = (not terminated) and self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"success": success, "dist": dist, "vla_calls": 0}


class GymnasiumAdapter:
    """Wrap a gymnasium env so its ``info`` always has a ``success`` key (False)."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_spec = ActionSpec.from_gym_box(env.action_space)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32).ravel(), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        info.setdefault("success", False)
        return np.asarray(obs, dtype=np.float32).ravel(), float(r), bool(term), bool(trunc), info
