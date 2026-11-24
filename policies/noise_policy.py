"""Gym-like DSRL interface for the team's shared SAC implementation.

One RL step = one VLA action chunk. SAC picks a noise latent z, the frozen VLA
turns it into ``n_action_steps`` actions, and all of them are executed before the
next decision. The simulator is the same object Residual-SAC uses
(``make_dynamic_libero(cfg)``), so both methods see identical perturbations.

Observation (no simulator object coordinates):
  * ``proprio_ref`` (15): proprioception(8) + first action of a zero-noise
    reference chunk(7). Same content as ResidualEnv's observation; costs one extra
    VLA forward pass per decision, which is counted in ``info["vla_calls"]``.
  * ``proprio`` (8): proprioception only, one VLA forward pass per decision.

``step(None)`` uses the VLA's own seeded Gaussian noise, i.e. the Frozen Base.
"""
import numpy as np

OBS_MODES = {"proprio_ref": 15, "proprio": 8}


class NoiseLatentEnv:
    def __init__(self, env, vla, max_steps=280, obs_mode="proprio_ref", n_action_steps=None):
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {sorted(OBS_MODES)}")
        self.env, self.vla = env, vla
        self.max_steps, self.obs_mode = max_steps, obs_mode
        self.n = n_action_steps or vla.config.n_action_steps
        self.obs_dim = OBS_MODES[obs_mode]
        self.action_spec = vla.spec.action_spec()

    def _observe(self):
        task = self.env.task_description
        if self.obs_mode == "proprio":
            return self.vla.state(self.raw_obs, task)
        state, reference = self.vla.chunk(self.raw_obs, task, self.steps, reference=True)
        return np.concatenate((state, reference[0])).astype(np.float32)

    def _terminal_observation(self):
        # Terminal states are never acted on; do not spend a VLA query on them.
        state = self.vla.state(self.raw_obs, self.env.task_description)
        return np.concatenate((state, np.zeros(self.obs_dim - state.size, dtype=np.float32))).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        self.steps, self.decisions, self.perturbed = 0, 0, False
        # Fresh episodes take their seed from the trainer's checkpointed numpy RNG,
        # as ResidualEnv does; the VLA's private generator is reseeded here.
        episode_seed = int(seed) if seed is not None else int(np.random.randint(0, 2**31 - 1))
        self.vla.reset(episode_seed)
        self.raw_obs, info = self.env.reset(seed=episode_seed)
        obs = self._observe()
        # The shared trainer ignores reset info, so the initial reference query is
        # charged to the first decision.
        self.pending_reset_calls = self.vla.calls
        return obs, {**info, "success": False, "vla_calls": 0,
                     "reset_vla_calls": self.vla.calls, "episode_seed": episode_seed}

    def step(self, action):
        latent = None
        if action is not None:
            latent = np.asarray(action, dtype=np.float32)
            if latent.shape != (self.action_spec.dim,) or not np.isfinite(latent).all():
                raise ValueError(f"Noise latent must be finite with shape ({self.action_spec.dim},)")
        if hasattr(self.env, "notify_replan"):
            self.env.notify_replan(self.steps)
        before = self.vla.calls
        _, chunk = self.vla.chunk(self.raw_obs, self.env.task_description, self.steps, latent=latent)
        total, executed, terminated, truncated, info = 0., 0, False, False, {}
        for planned in chunk[:self.n]:
            # ResidualEnv clips the executed action to [-1, 1]; keep the same units.
            self.raw_obs, reward, terminated, truncated, info = self.env.step(
                np.clip(planned, -1, 1).astype(np.float32))
            self.steps += 1
            executed += 1
            total += float(reward)
            truncated = bool(truncated or (self.steps >= self.max_steps and not terminated))
            if terminated or truncated:
                break
        self.decisions += 1
        self.perturbed = info.get("perturb_step", -1) != -1
        # Time-limit transitions bootstrap, so they get a real next observation.
        next_obs = self._terminal_observation() if terminated else self._observe()
        calls = self.vla.calls - before + self.pending_reset_calls
        self.pending_reset_calls = 0
        info.update(success=bool(info.get("is_success", False)), vla_calls=calls, env_steps=executed,
                    decisions=self.decisions, steered=latent is not None,
                    latent_abs_mean=float(np.abs(latent).mean()) if latent is not None else float("nan"))
        return next_obs, total, bool(terminated), truncated, info

    def close(self):
        self.env.close()
