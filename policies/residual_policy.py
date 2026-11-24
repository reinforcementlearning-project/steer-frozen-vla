"""Gym-like residual interface for the team's shared SAC implementation.

SACAgent.act already applies tanh. This adapter therefore uses alpha * u with
u in [-1, 1], avoiding a second tanh. Gripper correction is disabled by default.
"""
import numpy as np
from rl import ActionSpec


class ResidualEnv:
    obs_dim = 15  # proprioception(8) + unnormalized base action(7), no oracle coordinates

    def __init__(self, env, base, alpha=0.1, residual_gripper=False, max_steps=280):
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be nonnegative and finite")
        self.env, self.base = env, base
        self.alpha, self.residual_gripper = alpha, residual_gripper
        self.max_steps = max_steps
        self.action_spec = ActionSpec(7, -1, 1, "residual")

    def _observe(self, obs):
        state, self.base_action = self.base.action(obs, self.env.task_description, self.steps)
        return np.concatenate((state, self.base_action)).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        # Use the trainer's checkpointed numpy RNG for fresh episodes, including
        # after resume; the policy's private CUDA generator is reseeded here.
        episode_seed = int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
        self.base.reset(episode_seed)
        obs, info = self.env.reset(seed=episode_seed)
        state = self._observe(obs)
        # The shared trainer ignores reset info. Charge initial precomputation to
        # the first control step so its VLA budget includes every forward pass.
        self.pending_reset_calls = self.base.calls
        return state, {**info, "success": False, "vla_calls": 0,
                       "reset_vla_calls": self.base.calls, "episode_seed": episode_seed}

    def step(self, action):
        u = np.asarray(action, dtype=np.float32)
        if u.shape != (7,) or not np.isfinite(u).all() or np.any(np.abs(u) > 1.00001):
            raise ValueError("Residual must be finite (7,) in [-1, 1]")
        delta = self.alpha * np.clip(u, -1, 1)
        if not self.residual_gripper:
            delta[-1] = 0
        used_base_action = self.base_action.copy()
        executed = np.clip(used_base_action + delta, -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = self.env.step(executed)
        self.steps += 1
        truncated = bool(truncated or (self.steps >= self.max_steps and not terminated))
        before = self.base.calls
        if terminated or truncated:
            # Terminal states are never acted on; do not consume another VLA query.
            state = self.base.observation(obs, self.env.task_description)["observation.state"][0].numpy()
            next_obs = np.concatenate((state, np.zeros(7, dtype=np.float32))).astype(np.float32)
            if truncated:
                # Time-limit transitions bootstrap, so provide the real next base action.
                next_obs = self._observe(obs)
        else:
            next_obs = self._observe(obs)
        step_calls = self.base.calls - before + self.pending_reset_calls
        self.pending_reset_calls = 0
        info.update(success=bool(info.get("is_success", False)), vla_calls=step_calls,
                    env_steps=1, base_action=used_base_action.tolist(), exec_action=executed.tolist(),
                    residual_norm=float(np.linalg.norm(delta)))
        return next_obs, float(reward), bool(terminated), truncated, info

    def close(self):
        self.env.close()
