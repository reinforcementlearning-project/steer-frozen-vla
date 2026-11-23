"""Fixed-capacity FIFO replay buffer (numpy storage, torch sampling)."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, seed: Optional[int] = None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        # `dones` = terminated only (NOT truncated): a time-limit cut must still bootstrap.
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, terminated: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty replay buffer")
        idx = self.rng.integers(0, self.size, size=batch_size)

        def t(x: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(x[idx], device=device)

        return {
            "obs": t(self.obs),
            "actions": t(self.actions),
            "rewards": t(self.rewards),
            "next_obs": t(self.next_obs),
            "dones": t(self.dones),
        }

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        n = self.size
        np.savez_compressed(
            path,
            obs=self.obs[:n] if n < self.capacity else self.obs,
            next_obs=self.next_obs[:n] if n < self.capacity else self.next_obs,
            actions=self.actions[:n] if n < self.capacity else self.actions,
            rewards=self.rewards[:n] if n < self.capacity else self.rewards,
            dones=self.dones[:n] if n < self.capacity else self.dones,
            meta=np.array([self.ptr, self.size, self.capacity, self.obs_dim, self.action_dim], dtype=np.int64),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        ptr, size, capacity, obs_dim, action_dim = (int(x) for x in data["meta"])
        if (obs_dim, action_dim) != (self.obs_dim, self.action_dim):
            raise ValueError(
                f"Buffer shape mismatch: file has obs={obs_dim}, act={action_dim}; "
                f"buffer has obs={self.obs_dim}, act={self.action_dim}"
            )
        if capacity != self.capacity:
            raise ValueError(f"Buffer capacity mismatch: file {capacity} vs buffer {self.capacity}")
        n = data["obs"].shape[0]
        self.obs[:n] = data["obs"]
        self.next_obs[:n] = data["next_obs"]
        self.actions[:n] = data["actions"]
        self.rewards[:n] = data["rewards"]
        self.dones[:n] = data["dones"]
        self.ptr, self.size = ptr, size
