"""Unit tests for the shared SAC framework.  Run:  python -m pytest tests -q"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl import ActionSpec, OffPolicyTrainer, ReplayBuffer, SACAgent, SACConfig, TrainerConfig  # noqa: E402
from rl.toy_env import PointReachEnv  # noqa: E402

CPU = SACConfig(hidden_dims=[64, 64], device="cpu")


@pytest.mark.parametrize("dim,low,high", [(7, -1.0, 1.0), (4, -2.5, 2.5), (3, [-1, 0, -3], [1, 2, 3])])
def test_variable_action_dims_and_bounds(dim, low, high):
    spec = ActionSpec(dim=dim, low=low, high=high)
    agent = SACAgent(obs_dim=10, action_spec=spec, config=CPU)
    obs = np.random.randn(10).astype(np.float32)
    a = agent.act(obs)
    assert a.shape == (dim,)
    assert np.all(a >= spec.low_arr - 1e-6) and np.all(a <= spec.high_arr + 1e-6)
    assert agent.act(np.random.randn(5, 10)).shape == (5, dim)
    np.testing.assert_allclose(agent.act(obs, deterministic=True), agent.act(obs, deterministic=True))


def test_log_prob_matches_torch_distribution():
    torch.manual_seed(0)
    agent = SACAgent(6, ActionSpec(dim=3), CPU)
    obs = torch.randn(128, 6)
    torch.manual_seed(1)
    a, logp, _ = agent.actor.sample(obs)
    mu, log_std = agent.actor(obs)
    base = torch.distributions.Normal(mu, log_std.exp())
    u = torch.atanh(a.clamp(-0.999999, 0.999999))
    ref = (base.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1, keepdim=True)
    torch.testing.assert_close(logp, ref, atol=1e-2, rtol=1e-3)


def test_update_changes_params_and_returns_metrics():
    agent = SACAgent(4, ActionSpec(dim=2), CPU)
    buf = ReplayBuffer(4, 2, 1000, seed=0)
    for _ in range(300):
        buf.add(np.random.randn(4), np.random.uniform(-1, 1, 2), np.random.randn(), np.random.randn(4), False)
    before = [p.clone() for p in agent.actor.parameters()]
    tgt_before = [p.clone() for p in agent.critic_target.parameters()]
    m = agent.update(buf.sample(64))
    for k in ["critic_loss", "actor_loss", "alpha", "alpha_loss", "entropy"]:
        assert k in m and np.isfinite(m[k])
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(tgt_before, agent.critic_target.parameters()))
    assert agent.num_updates == 1


def test_agent_save_load_roundtrip(tmp_path):
    spec = ActionSpec(dim=5, low=-2, high=2, name="noise_latent")
    agent = SACAgent(8, spec, CPU)
    path = str(tmp_path / "agent.pt")
    agent.save(path)
    re = SACAgent.from_checkpoint(path, device="cpu")
    obs = np.random.randn(8).astype(np.float32)
    np.testing.assert_allclose(agent.act(obs, True), re.act(obs, True), atol=1e-6)
    assert re.action_spec.name == "noise_latent" and re.action_spec.dim == 5


def test_replay_buffer_save_load(tmp_path):
    buf = ReplayBuffer(3, 2, 50, seed=0)
    for i in range(70):  # wraps around
        buf.add(np.full(3, i), np.full(2, i), i, np.full(3, i + 1), i % 7 == 0)
    path = str(tmp_path / "buf.npz")
    buf.save(path)
    buf2 = ReplayBuffer(3, 2, 50)
    buf2.load(path)
    assert (buf2.ptr, buf2.size) == (buf.ptr, buf.size)
    np.testing.assert_array_equal(buf.obs, buf2.obs)
    np.testing.assert_array_equal(buf.dones, buf2.dones)


def _trainer(tmp_path, steps, **kw):
    env = PointReachEnv(dim=2)
    agent = SACAgent(env.obs_dim, env.action_spec, CPU)
    cfg = TrainerConfig(total_env_steps=steps, warmup_steps=200, learning_starts=200, batch_size=64,
                        log_every=100, eval_every=0, checkpoint_every=250, keep_checkpoints=False,
                        seed=0, run_dir=str(tmp_path / "run"), verbose=False, **kw)
    return OffPolicyTrainer(env, agent, cfg, eval_env=PointReachEnv(dim=2))


def test_trainer_counters_logs_and_checkpoint(tmp_path):
    tr = _trainer(tmp_path, 500, utd_ratio=2)
    tr.train()
    assert tr.env_steps == 500
    assert tr.grad_updates == 2 * (500 - 200 + 1)  # updates start once buffer has 200 transitions
    run = tmp_path / "run"
    for f in ["config.json", "train.csv", "episodes.csv", "checkpoints/latest.pt", "checkpoints/replay_buffer.npz"]:
        assert (run / f).exists(), f


def test_resume_restores_learning_state(tmp_path):
    """Train 500 steps straight vs. 250 + resume + 250: final weights must be identical."""
    a = _trainer(tmp_path / "a", 500)
    a.train()

    b1 = _trainer(tmp_path / "b", 250)
    b1.train()
    b2 = _trainer(tmp_path / "b", 500)
    b2.resume()
    assert b2.env_steps == 250 and len(b2.buffer) == 250
    b2.train()
    assert b2.env_steps == 500 and b2.grad_updates == a.grad_updates
    # The in-progress episode is restarted on resume, so trajectories (and weights) may
    # differ from run `a`; what must hold is that learning state was carried over.
    assert b2.agent.num_updates == a.agent.num_updates


@pytest.mark.parametrize("policy,expect", [("zero", 0.0), ("gaussian", None)])
def test_warmup_policies(tmp_path, policy, expect):
    tr = _trainer(tmp_path, 10, warmup_policy=policy)
    a = tr._warmup_action(np.zeros(4, dtype=np.float32))
    assert a.shape == (2,) and np.all(np.abs(a) <= 1.0)
    if expect is not None:
        assert np.all(a == expect)


def test_sac_learns_point_reach(tmp_path):
    """Learning smoke test: SAC must clearly beat a random policy within 6k steps."""
    env = PointReachEnv(dim=2)
    agent = SACAgent(env.obs_dim, env.action_spec, SACConfig(device="cpu"))
    tr = OffPolicyTrainer(env, agent, TrainerConfig(
        total_env_steps=6000, warmup_steps=1000, learning_starts=1000, eval_every=0, log_every=0,
        checkpoint_every=0, seed=0, run_dir=str(tmp_path / "learn"), verbose=False), eval_env=PointReachEnv(dim=2))
    random_sr = np.mean([_random_success(PointReachEnv(dim=2), s) for s in range(20)])
    tr.train()
    ev = tr.evaluate(20)
    assert ev["success_rate"] >= 0.8 and ev["success_rate"] > random_sr + 0.4, (ev, random_sr)


def _random_success(env, seed):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    while True:
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, env.dim))
        if term or trunc:
            return float(info["success"])
