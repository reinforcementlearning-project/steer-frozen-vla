"""Shared policy interface for Residual-SAC and DSRL-SAC.

Both methods use the same SAC core. The only difference is *what* the action
means, so the agent is parameterised by an :class:`ActionSpec`:

* Residual-SAC : ``ActionSpec(dim=7, low=-1, high=1, name="residual")``
  (the residual Δa is later scaled by α and added to the frozen VLA action)
* DSRL-SAC     : ``ActionSpec(dim=latent_dim, low=-b, high=b, name="noise_latent")``
  (the latent is mapped to the flow-matching initial noise of SmolVLA)

Environments are expected to follow the gymnasium API and return a flat
``np.ndarray`` observation (the env / VLA wrapper is responsible for turning
LIBERO dict observations into a state vector).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


@dataclass
class ActionSpec:
    """Continuous, box-bounded action space of arbitrary dimension."""

    dim: int
    low: ArrayLike = -1.0
    high: ArrayLike = 1.0
    name: str = "action"
    low_arr: np.ndarray = field(init=False, repr=False)
    high_arr: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError(f"ActionSpec.dim must be positive, got {self.dim}")
        self.low_arr = np.broadcast_to(np.asarray(self.low, dtype=np.float32), (self.dim,)).copy()
        self.high_arr = np.broadcast_to(np.asarray(self.high, dtype=np.float32), (self.dim,)).copy()
        if np.any(self.high_arr <= self.low_arr):
            raise ValueError("ActionSpec.high must be strictly greater than low in every dimension")

    @property
    def center(self) -> np.ndarray:
        return (self.high_arr + self.low_arr) / 2.0

    @property
    def scale(self) -> np.ndarray:
        return (self.high_arr - self.low_arr) / 2.0

    def clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.low_arr, self.high_arr)

    def sample_uniform(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low_arr, self.high_arr).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "low": self.low_arr.tolist(),
            "high": self.high_arr.tolist(),
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionSpec":
        return cls(dim=int(d["dim"]), low=d["low"], high=d["high"], name=d.get("name", "action"))

    @classmethod
    def from_gym_box(cls, space: Any, name: str = "action") -> "ActionSpec":
        """Build from a ``gymnasium.spaces.Box`` (1-D)."""
        return cls(dim=int(np.prod(space.shape)), low=space.low.ravel(), high=space.high.ravel(), name=name)


@runtime_checkable
class RLAgent(Protocol):
    """Minimal interface every off-policy agent must provide to the trainer."""

    obs_dim: int
    action_spec: ActionSpec

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Single observation (obs_dim,) -> action (action_dim,) in env scale."""

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """One gradient step; returns scalar metrics for logging."""

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@runtime_checkable
class GymLikeEnv(Protocol):
    """gymnasium-style env. ``info`` may contain ``success`` and ``vla_calls``."""

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]: ...

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]: ...
