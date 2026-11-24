"""Frozen SmolVLA whose flow-matching initial noise can be set from outside (DSRL).

``FrozenSmolVLA`` draws the initial noise ``eps ~ N(0, I)`` of shape
``(1, chunk_size, max_action_dim)`` itself. ``NoiseSteeredSmolVLA`` adds
``chunk()``, which builds that noise from a low-dimensional latent chosen by SAC:

    z (steps * dims,)  ->  noise[:steps, :dims] = z
                           noise[steps:, :dims] = z[-1]      (last row repeated)
                           noise[:, dims:]      = eps or 0   (``fill``)

* ``latent=None`` uses ``eps`` unchanged: one generator draw per chunk, the same
  stream and therefore the same actions as ``FrozenSmolVLA`` (the Frozen Base).
* ``reference=True`` uses zero noise and does not touch the generator. It gives a
  deterministic "what would the VLA do here" action that the RL env can observe.

The weights stay frozen; nothing here is differentiated.
"""
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch

from models.smolvla_wrapper import FrozenSmolVLA
from rl import ActionSpec

FILLS = ("gaussian", "zero")


@dataclass(frozen=True)
class NoiseLatentSpec:
    """How a SAC action is laid out inside the (chunk_size, max_action_dim) noise tensor."""

    # Measured on LIBERO-Object task 0 (n_action_steps=10): noise dim j moves action dim j
    # almost linearly (about 0.03 per unit for xyz when repeated over the chunk), dims >= 7
    # have ~1/4 of the effect of dims 0-6, and noise for chunk steps >= n_action_steps has
    # no effect on the executed actions.
    dims: int = 6          # leading noise dims set by the latent (6 = arm, 7 = arm + gripper, 32 = all)
    steps: int = 1         # leading chunk steps set independently; the last one is repeated to the end
    bound: float = 2.5     # latent range [-bound, bound], in units of the N(0, 1) noise
    fill: str = "gaussian"  # noise dims >= dims: seeded N(0, 1) draw (base prior) or zero

    def __post_init__(self):
        if self.dims < 1 or self.steps < 1:
            raise ValueError("dims and steps must be positive")
        if not np.isfinite(self.bound) or self.bound <= 0:
            raise ValueError("bound must be positive and finite")
        if self.fill not in FILLS:
            raise ValueError(f"fill must be one of {FILLS}")

    @property
    def latent_dim(self) -> int:
        return self.dims * self.steps

    def action_spec(self) -> ActionSpec:
        return ActionSpec(self.latent_dim, -self.bound, self.bound, "noise_latent")

    def check(self, chunk_size: int, max_action_dim: int) -> None:
        if self.steps > chunk_size or self.dims > max_action_dim:
            raise ValueError(f"latent {self.steps}x{self.dims} does not fit noise {chunk_size}x{max_action_dim}")

    def to_dict(self):
        return asdict(self)


def expand_latent(latent, base_noise: torch.Tensor, spec: NoiseLatentSpec) -> torch.Tensor:
    """latent (steps * dims,) + base draw (1, chunk, max_dim) -> initial noise of the same shape."""
    if base_noise.ndim != 3 or base_noise.shape[0] != 1:
        raise ValueError(f"base noise must be (1, chunk, max_dim), got {tuple(base_noise.shape)}")
    spec.check(base_noise.shape[1], base_noise.shape[2])
    z = torch.as_tensor(np.asarray(latent, dtype=np.float32), device=base_noise.device)
    if z.shape != (spec.latent_dim,) or not torch.isfinite(z).all():
        raise ValueError(f"latent must be finite with shape ({spec.latent_dim},)")
    if torch.any(z.abs() > spec.bound * 1.00001):
        raise ValueError(f"latent outside [-{spec.bound}, {spec.bound}]")
    z = z.reshape(spec.steps, spec.dims).to(base_noise.dtype)
    noise = base_noise.clone() if spec.fill == "gaussian" else torch.zeros_like(base_noise)
    noise[0, :spec.steps, :spec.dims] = z
    noise[0, spec.steps:, :spec.dims] = z[-1]
    return noise


class NoiseSteeredSmolVLA(FrozenSmolVLA):
    def __init__(self, checkpoint, backbone, n_action_steps=10, seed=0, spec=None):
        super().__init__(checkpoint, backbone, n_action_steps, seed, noise_mode="random")
        self.spec = spec or NoiseLatentSpec()
        self.noise_shape = (1, self.config.chunk_size, self.config.max_action_dim)
        self.spec.check(*self.noise_shape[1:])

    def state(self, obs, task) -> np.ndarray:
        """Proprioceptive state the VLA itself receives (no simulator object coordinates)."""
        return self.observation(obs, task)["observation.state"][0].numpy().astype(np.float32)

    @torch.no_grad()
    def chunk(self, obs, task, env_step, latent=None, reference=False):
        """One VLA forward pass. Returns (state (8,), actions (n_action_steps, 7), unclipped)."""
        if reference and latent is not None:
            raise ValueError("a reference chunk takes no latent")
        observation = self.observation(obs, task)
        state = observation["observation.state"][0].numpy().astype(np.float32)
        processed = self.pre(observation)
        if reference:
            noise, kind = torch.zeros(self.noise_shape, device="cuda"), "reference"
        else:
            # Same draw, in the same order, as FrozenSmolVLA.action().
            noise = torch.randn(self.noise_shape, device="cuda", generator=self.generator)
            kind = "base"
            if latent is not None:
                noise, kind = expand_latent(latent, noise, self.spec), "steered"
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = self.policy.predict_action_chunk(processed, noise=noise)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        actions = self.post(out).detach().cpu().numpy()[0]
        if actions.shape != (self.config.chunk_size, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid action chunk: {actions.shape}")
        self.calls += 1
        self.trace.append({"env_step": env_step, "kind": kind, "inference_seconds": elapsed,
                           "noise_norm": float(noise.norm()),
                           "action_min": float(actions.min()), "action_max": float(actions.max())})
        return state, actions[:self.config.n_action_steps].astype(np.float32).copy()
