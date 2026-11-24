"""Frozen SmolVLA, checkpoint processors, and an explicit action execution queue."""
from collections import deque
import time
import numpy as np
import torch
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.processor import LiberoProcessorStep


def batch_observation(value):
    if isinstance(value, dict):
        return {k: batch_observation(v) for k, v in value.items()}
    return np.expand_dims(np.asarray(value), 0)


class FrozenSmolVLA:
    def __init__(self, checkpoint, backbone, n_action_steps=1, seed=0, noise_mode="random"):
        if noise_mode not in ("random", "zero"):
            raise ValueError(noise_mode)
        config = SmolVLAConfig.from_pretrained(checkpoint)
        self.checkpoint_n_action_steps = config.n_action_steps
        if not 1 <= n_action_steps <= config.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        config.n_action_steps = n_action_steps
        config.vlm_model_name = str(backbone)
        config.load_vlm_weights = False  # Full policy checkpoint already includes VLM weights.
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to("cuda").eval()
        self.policy.requires_grad_(False)
        self.config = config
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=checkpoint,
            preprocessor_overrides={"tokenizer_processor": {"tokenizer_name": str(backbone)}})
        self.env_processor = LiberoProcessorStep()
        self.noise_mode = noise_mode
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        self.queue = deque()
        self.calls = 0
        self.trace = []

    def reset(self, seed=None):
        self.policy.reset()
        self.queue.clear()
        self.calls = 0
        self.trace = []
        if seed is not None:
            self.generator.manual_seed(seed)

    def observation(self, obs, task):
        observation = preprocess_observation(batch_observation(obs))
        observation = self.env_processor.observation(observation)
        observation["task"] = [task]
        return observation

    @torch.no_grad()
    def action(self, obs, task, env_step):
        observation = self.observation(obs, task)
        state = observation["observation.state"][0].numpy().astype(np.float32)
        if not self.queue:
            processed = self.pre(observation)
            shape = (1, self.config.chunk_size, self.config.max_action_dim)
            noise = (torch.randn(shape, device="cuda", generator=self.generator)
                     if self.noise_mode == "random" else torch.zeros(shape, device="cuda"))
            torch.cuda.synchronize()
            start = time.perf_counter()
            chunk = self.policy.predict_action_chunk(processed, noise=noise)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            actions = self.post(chunk).detach().cpu().numpy()[0]
            if actions.shape != (self.config.chunk_size, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid action chunk: {actions.shape}")
            self.queue.extend(actions[:self.config.n_action_steps])
            self.calls += 1
            self.trace.append({"env_step": env_step, "chunk_shape": list(chunk.shape),
                "noise_shape": list(noise.shape), "inference_seconds": elapsed,
                "action_min": float(actions.min()), "action_max": float(actions.max()),
                "state_shape": list(processed["observation.state"].shape)})
        return state, self.queue.popleft().copy()
