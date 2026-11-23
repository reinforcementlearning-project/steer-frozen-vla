"""Soft Actor-Critic (Haarnoja et al., 2018) with automatic entropy tuning.

Shared by Residual-SAC (action-space residual) and DSRL-SAC (noise-latent
action). The action dimension / bounds come from :class:`ActionSpec`, so the
same class is used for both — only the spec changes.

Internally the actor works in a normalised [-1, 1] space (tanh-squashed
Gaussian). ``act()`` returns actions in env scale; the replay buffer stores
env-scale actions and ``update()`` re-normalises them before feeding the critic.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .interface import ActionSpec

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


@dataclass
class SACConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    # Entropy coefficient
    init_alpha: float = 0.1
    autotune_alpha: bool = True
    target_entropy_scale: float = 1.0  # target_entropy = -scale * action_dim
    # Update schedule (inside one `update()` call)
    actor_update_freq: int = 1  # delayed policy update (1 = every critic step)
    target_update_freq: int = 1
    grad_clip_norm: Optional[float] = None
    device: str = "auto"

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def mlp(in_dim: int, out_dim: int, hidden_dims: List[int]) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    """Outputs normalised actions in (-1, 1)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.trunk = mlp(obs_dim, 2 * action_dim, hidden_dims)
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (action_norm, log_prob [B,1], deterministic_action_norm)."""
        mu, log_std = self(obs)
        std = log_std.exp()
        u = mu + std * torch.randn_like(mu)  # reparameterisation
        a = torch.tanh(u)
        # log N(u; mu, std) - sum log(1 - tanh(u)^2), numerically stable form
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        log_prob -= (2.0 * (np.log(2.0) - u - F.softplus(-2.0 * u))).sum(-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, 1, hidden_dims)
        self.q2 = mlp(obs_dim + action_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, action_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action_norm], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, obs_dim: int, action_spec: ActionSpec, config: Optional[SACConfig] = None) -> None:
        self.obs_dim = int(obs_dim)
        self.action_spec = action_spec
        self.config = config or SACConfig()
        self.device = self.config.resolve_device()
        act_dim = action_spec.dim
        hd = self.config.hidden_dims

        self.actor = TanhGaussianActor(self.obs_dim, act_dim, hd).to(self.device)
        self.critic = TwinQ(self.obs_dim, act_dim, hd).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        self.target_entropy = -self.config.target_entropy_scale * act_dim
        self.log_alpha = torch.tensor(np.log(self.config.init_alpha), dtype=torch.float32,
                                      device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)

        self._center = torch.as_tensor(action_spec.center, device=self.device)
        self._scale = torch.as_tensor(action_spec.scale, device=self.device)
        self.num_updates = 0

    # ------------------------------------------------------------------ helpers
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _to_env(self, a_norm: torch.Tensor) -> torch.Tensor:
        return self._center + self._scale * a_norm

    def _to_norm(self, a_env: torch.Tensor) -> torch.Tensor:
        return torch.clamp((a_env - self._center) / self._scale, -1.0, 1.0)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32), device=self.device)
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)
        a, _, a_det = self.actor.sample(obs_t)
        out = self._to_env(a_det if deterministic else a).cpu().numpy()
        return out[0] if single else out

    # ------------------------------------------------------------------ learning
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        cfg = self.config
        obs = batch["obs"].to(self.device)
        act = self._to_norm(batch["actions"].to(self.device))
        rew = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["dones"].to(self.device)
        alpha = self.alpha.detach()

        # ---- critic
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_a)
            target_v = torch.min(tq1, tq2) - alpha * next_logp
            target_q = rew + cfg.gamma * (1.0 - done) * target_v
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if cfg.grad_clip_norm:
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip_norm)
        self.critic_opt.step()

        metrics: Dict[str, float] = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q_target_mean": target_q.mean().item(),
        }

        # ---- actor & alpha (optionally delayed)
        if self.num_updates % cfg.actor_update_freq == 0:
            new_a, logp, _ = self.actor.sample(obs)
            self.critic.requires_grad_(False)
            q_pi = torch.min(*self.critic(obs, new_a))
            self.critic.requires_grad_(True)
            actor_loss = (alpha * logp - q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if cfg.grad_clip_norm:
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip_norm)
            self.actor_opt.step()
            metrics["actor_loss"] = actor_loss.item()
            metrics["entropy"] = -logp.mean().item()

            if cfg.autotune_alpha:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                metrics["alpha_loss"] = alpha_loss.item()

        # ---- target network (Polyak)
        if self.num_updates % cfg.target_update_freq == 0:
            with torch.no_grad():
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.lerp_(p, cfg.tau)

        self.num_updates += 1
        metrics["alpha"] = self.alpha.item()
        return metrics

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> Dict[str, Any]:
        return {
            "obs_dim": self.obs_dim,
            "action_spec": self.action_spec.to_dict(),
            "config": asdict(self.config),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        if sd["obs_dim"] != self.obs_dim or sd["action_spec"]["dim"] != self.action_spec.dim:
            raise ValueError(
                f"Checkpoint dims (obs={sd['obs_dim']}, act={sd['action_spec']['dim']}) do not match "
                f"agent (obs={self.obs_dim}, act={self.action_spec.dim})"
            )
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        self.num_updates = int(sd["num_updates"])

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "auto") -> "SACAgent":
        """Rebuild an agent (dims, spec, config) purely from a checkpoint file."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if "agent" in sd:  # trainer checkpoint
            sd = sd["agent"]
        cfg = SACConfig(**{**sd["config"], "device": device})
        agent = cls(sd["obs_dim"], ActionSpec.from_dict(sd["action_spec"]), cfg)
        agent.load_state_dict(sd)
        return agent
