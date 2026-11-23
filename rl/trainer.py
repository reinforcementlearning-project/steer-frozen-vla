"""Generic off-policy training loop shared by train_residual.py / train_dsrl.py.

Responsibilities (plan §8-1):
  * replay warm-up, update frequency / UTD ratio, batch size
  * logging of env steps, VLA calls, optimizer updates, reward, success, losses
  * evaluation on fixed seeds
  * checkpoint save + resume (agent, optimizers, replay buffer, RNG, counters)

The env must follow the gymnasium API with a flat ``np.ndarray`` observation.
Optional ``info`` keys:
  * ``success``   (bool)  task success, OR-ed over the episode
  * ``vla_calls`` (int)   number of frozen-VLA forward passes made in this step
  * anything in ``TrainerConfig.episode_info_keys`` is logged at episode end
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from .interface import GymLikeEnv
from .logger import CSVLogger, fmt_metrics
from .replay_buffer import ReplayBuffer
from .sac import SACAgent

WarmupPolicy = Union[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]]


@dataclass
class TrainerConfig:
    total_env_steps: int = 100_000
    # Replay warm-up: actions are taken from `warmup_policy` for the first N env steps.
    #   "uniform"  : U(low, high)               — standard SAC
    #   "gaussian" : N(0, 1) clipped to bounds  — DSRL: equals the frozen VLA's own noise prior
    #   "zero"     : all-zeros                  — Residual: equals the frozen VLA's own action
    warmup_steps: int = 1_000
    warmup_policy: str = "uniform"
    learning_starts: int = 1_000  # first gradient step once buffer has this many transitions
    batch_size: int = 256
    buffer_capacity: int = 1_000_000
    update_every: int = 1  # run updates every N env steps ...
    utd_ratio: int = 1  # ... doing `update_every * utd_ratio` gradient steps each time
    # Logging / evaluation / checkpointing (all in env steps; 0 disables)
    log_every: int = 1_000
    eval_every: int = 10_000
    eval_episodes: int = 10
    eval_seed: int = 10_000  # eval episode i uses reset(seed=eval_seed + i)
    checkpoint_every: int = 10_000
    keep_checkpoints: bool = True  # also keep ckpt_<step>.pt besides latest.pt
    save_buffer: bool = True
    # Misc
    seed: int = 0
    run_dir: str = "results/sac_run"
    success_key: str = "success"
    vla_calls_key: str = "vla_calls"
    episode_info_keys: List[str] = field(default_factory=list)
    verbose: bool = True


class OffPolicyTrainer:
    def __init__(
        self,
        env: GymLikeEnv,
        agent: SACAgent,
        config: TrainerConfig,
        eval_env: Optional[GymLikeEnv] = None,
        warmup_fn: Optional[Callable[[np.ndarray, np.random.Generator], np.ndarray]] = None,
    ) -> None:
        self.env = env
        self.eval_env = eval_env if eval_env is not None else env
        self.agent = agent
        self.cfg = config
        self.warmup_fn = warmup_fn
        self.buffer = ReplayBuffer(agent.obs_dim, agent.action_spec.dim, config.buffer_capacity, seed=config.seed)
        self.rng = np.random.default_rng(config.seed)
        self.logger = CSVLogger(config.run_dir)
        self.ckpt_dir = os.path.join(config.run_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        # Budget counters (plan §9: env steps / VLA calls / optimizer updates)
        self.env_steps = 0
        self.vla_calls = 0
        self.grad_updates = 0
        self.episodes = 0
        self.wall_time = 0.0  # accumulated across resumes
        self._first_reset_done = False

        seed_everything(config.seed)
        with open(os.path.join(config.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"trainer": asdict(config), "agent": asdict(agent.config),
                       "obs_dim": agent.obs_dim, "action_spec": agent.action_spec.to_dict()}, f, indent=2)

    # ------------------------------------------------------------------ acting
    def _warmup_action(self, obs: np.ndarray) -> np.ndarray:
        spec = self.agent.action_spec
        if self.warmup_fn is not None:
            return spec.clip(np.asarray(self.warmup_fn(obs, self.rng), dtype=np.float32))
        p = self.cfg.warmup_policy
        if p == "uniform":
            return spec.sample_uniform(self.rng)
        if p == "gaussian":
            return spec.clip(self.rng.standard_normal(spec.dim).astype(np.float32))
        if p == "zero":
            return spec.clip(np.zeros(spec.dim, dtype=np.float32))
        raise ValueError(f"Unknown warmup_policy: {p!r}")

    def _reset(self) -> np.ndarray:
        if not self._first_reset_done:
            obs, _ = self.env.reset(seed=self.cfg.seed)
            self._first_reset_done = True
        else:
            obs, _ = self.env.reset()
        return np.asarray(obs, dtype=np.float32)

    # ------------------------------------------------------------------ main loop
    def train(self) -> Dict[str, float]:
        cfg = self.cfg
        obs = self._reset()
        ep_ret, ep_len, ep_success = 0.0, 0, False
        recent_returns: List[float] = []
        recent_success: List[float] = []
        last_metrics: Dict[str, float] = {}
        t_start = time.time() - self.wall_time

        while self.env_steps < cfg.total_env_steps:
            # ---- act
            if self.env_steps < cfg.warmup_steps:
                action = self._warmup_action(obs)
            else:
                action = self.agent.act(obs, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            self.buffer.add(obs, action, float(reward), next_obs, bool(terminated))

            self.env_steps += 1
            self.vla_calls += int(info.get(cfg.vla_calls_key, 0))
            ep_ret += float(reward)
            ep_len += 1
            ep_success = ep_success or bool(info.get(cfg.success_key, False))
            obs = next_obs

            # ---- learn
            if len(self.buffer) >= max(cfg.learning_starts, cfg.batch_size) and self.env_steps % cfg.update_every == 0:
                for _ in range(cfg.update_every * cfg.utd_ratio):
                    last_metrics = self.agent.update(self.buffer.sample(cfg.batch_size, self.agent.device))
                    self.grad_updates += 1

            # ---- episode end
            if terminated or truncated:
                self.episodes += 1
                row = {"env_step": self.env_steps, "episode": self.episodes, "return": ep_ret,
                       "length": ep_len, "success": int(ep_success), "vla_calls": self.vla_calls,
                       "grad_updates": self.grad_updates, "wall_time": time.time() - t_start}
                for k in cfg.episode_info_keys:
                    row[k] = info.get(k)
                self.logger.log("episodes", row)
                recent_returns = (recent_returns + [ep_ret])[-20:]
                recent_success = (recent_success + [float(ep_success)])[-20:]
                obs = self._reset()
                ep_ret, ep_len, ep_success = 0.0, 0, False

            # ---- periodic logging
            if cfg.log_every and self.env_steps % cfg.log_every == 0:
                self.wall_time = time.time() - t_start
                row = {"env_step": self.env_steps, "grad_updates": self.grad_updates, "vla_calls": self.vla_calls,
                       "episodes": self.episodes, "wall_time": self.wall_time,
                       "return_avg20": float(np.mean(recent_returns)) if recent_returns else float("nan"),
                       "success_avg20": float(np.mean(recent_success)) if recent_success else float("nan"),
                       "buffer_size": len(self.buffer), "sps": self.env_steps / max(self.wall_time, 1e-9)}
                if torch.cuda.is_available():
                    row["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                row.update(last_metrics)
                self.logger.log("train", row)
                if cfg.verbose:
                    print("[train] " + fmt_metrics(row, ["env_step", "episodes", "return_avg20", "success_avg20",
                                                         "critic_loss", "actor_loss", "alpha", "sps"]), flush=True)

            # ---- evaluation
            if cfg.eval_every and self.env_steps % cfg.eval_every == 0:
                ev = self.evaluate(cfg.eval_episodes)
                ev.update({"env_step": self.env_steps, "grad_updates": self.grad_updates, "vla_calls": self.vla_calls})
                self.logger.log("eval", ev)
                if cfg.verbose:
                    print("[eval]  " + fmt_metrics(ev, ["env_step", "return_mean", "success_rate", "length_mean"]), flush=True)
                if self.eval_env is self.env:  # eval interrupted the training episode
                    obs = self._reset()
                    ep_ret, ep_len, ep_success = 0.0, 0, False

            # ---- checkpoint
            if cfg.checkpoint_every and self.env_steps % cfg.checkpoint_every == 0:
                self.wall_time = time.time() - t_start
                self.save_checkpoint()

        self.wall_time = time.time() - t_start
        if not (cfg.checkpoint_every and self.env_steps % cfg.checkpoint_every == 0):  # not already saved
            self.save_checkpoint()
        return last_metrics

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, n_episodes: int, deterministic: bool = True) -> Dict[str, float]:
        returns, lengths, successes = [], [], []
        for i in range(n_episodes):
            obs, _ = self.eval_env.reset(seed=self.cfg.eval_seed + i)
            done, ret, length, success = False, 0.0, 0, False
            while not done:
                action = self.agent.act(np.asarray(obs, dtype=np.float32), deterministic=deterministic)
                obs, r, term, trunc, info = self.eval_env.step(action)
                ret += float(r)
                length += 1
                success = success or bool(info.get(self.cfg.success_key, False))
                done = term or trunc
            returns.append(ret)
            lengths.append(length)
            successes.append(float(success))
        return {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
                "success_rate": float(np.mean(successes)), "length_mean": float(np.mean(lengths)),
                "n_episodes": n_episodes}

    # ------------------------------------------------------------------ checkpointing
    def save_checkpoint(self) -> str:
        state = {
            "agent": self.agent.state_dict(),
            "trainer": {
                "env_steps": self.env_steps, "vla_calls": self.vla_calls, "grad_updates": self.grad_updates,
                "episodes": self.episodes, "wall_time": self.wall_time, "config": asdict(self.cfg),
            },
            "rng": {
                "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "trainer_rng": self.rng.bit_generator.state, "buffer_rng": self.buffer.rng.bit_generator.state,
            },
        }
        latest = os.path.join(self.ckpt_dir, "latest.pt")
        torch.save(state, latest + ".tmp")
        os.replace(latest + ".tmp", latest)  # atomic: never leaves a half-written latest.pt
        if self.cfg.keep_checkpoints:
            torch.save(state, os.path.join(self.ckpt_dir, f"ckpt_{self.env_steps:08d}.pt"))
        if self.cfg.save_buffer:
            self.buffer.save(os.path.join(self.ckpt_dir, "replay_buffer.npz.tmp.npz"))
            os.replace(os.path.join(self.ckpt_dir, "replay_buffer.npz.tmp.npz"),
                       os.path.join(self.ckpt_dir, "replay_buffer.npz"))
        if self.cfg.verbose:
            print(f"[ckpt]  saved {latest} (env_step={self.env_steps})", flush=True)
        return latest

    def resume(self, path: Optional[str] = None) -> None:
        """Restore everything from a trainer checkpoint (default: <run_dir>/checkpoints/latest.pt).

        The in-progress episode is not restored: training restarts from a fresh
        ``env.reset()`` (seeded from the restored RNG state).
        """
        path = path or os.path.join(self.ckpt_dir, "latest.pt")
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.agent.load_state_dict(state["agent"])
        t = state["trainer"]
        self.env_steps, self.vla_calls = t["env_steps"], t["vla_calls"]
        self.grad_updates, self.episodes, self.wall_time = t["grad_updates"], t["episodes"], t["wall_time"]
        r = state["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
        self.rng.bit_generator.state = r["trainer_rng"]
        self.buffer.rng.bit_generator.state = r["buffer_rng"]
        buf_path = os.path.join(os.path.dirname(path), "replay_buffer.npz")
        if os.path.exists(buf_path):
            self.buffer.load(buf_path)
        elif self.cfg.verbose:
            print(f"[resume] no replay buffer at {buf_path}; continuing with an empty buffer", flush=True)
        self._first_reset_done = True
        if self.cfg.verbose:
            print(f"[resume] {path} -> env_step={self.env_steps}, grad_updates={self.grad_updates}, "
                  f"buffer={len(self.buffer)}", flush=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
