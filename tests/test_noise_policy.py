"""DSRL noise latent layout, chunk-level env contract and control-step budget (no GPU, no simulator)."""
import csv
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("lerobot")
import torch

from models.noise_wrapper import NoiseLatentSpec, expand_latent
from policies.noise_policy import NoiseLatentEnv
from train_dsrl import ControlStepBudget


def test_latent_is_repeated_over_the_chunk_and_other_dims_keep_the_base_draw():
    base = torch.randn(1, 50, 32)
    spec = NoiseLatentSpec(dims=7, steps=1, bound=2.5)
    z = np.linspace(-2, 2, 7).astype(np.float32)
    noise = expand_latent(z, base, spec)
    assert noise.shape == base.shape
    np.testing.assert_allclose(noise[0, :, :7].numpy(), np.tile(z, (50, 1)))
    assert torch.equal(noise[0, :, 7:], base[0, :, 7:])
    zero = expand_latent(z, base, NoiseLatentSpec(dims=7, fill="zero"))
    assert torch.count_nonzero(zero[0, :, 7:]) == 0
    assert torch.equal(base, base.clone())  # the base draw itself is not modified in place


def test_multi_step_latent_sets_leading_steps_and_repeats_the_last():
    base = torch.zeros(1, 50, 32)
    spec = NoiseLatentSpec(dims=3, steps=2, bound=1.0)
    assert spec.latent_dim == 6 and spec.action_spec().dim == 6
    noise = expand_latent(np.array([.1, .2, .3, .4, .5, .6]), base, spec)
    np.testing.assert_allclose(noise[0, 0, :3].numpy(), [.1, .2, .3])
    np.testing.assert_allclose(noise[0, 1:, :3].numpy(), np.tile([.4, .5, .6], (49, 1)))


@pytest.mark.parametrize("latent", [np.zeros(6), np.full(7, np.nan), np.full(7, 3.0)])
def test_invalid_latent_rejected(latent):
    with pytest.raises(ValueError):
        expand_latent(latent, torch.zeros(1, 50, 32), NoiseLatentSpec(dims=7, bound=2.5))


def test_spec_validation():
    spec = NoiseLatentSpec()
    assert spec.action_spec().name == "noise_latent"
    np.testing.assert_allclose(spec.action_spec().high_arr, spec.bound)
    with pytest.raises(ValueError):
        NoiseLatentSpec(dims=33).check(50, 32)
    with pytest.raises(ValueError):
        NoiseLatentSpec(fill="uniform")
    with pytest.raises(ValueError):
        NoiseLatentSpec(bound=0)


class FakeVLA:
    spec = NoiseLatentSpec(dims=7)
    config = SimpleNamespace(n_action_steps=10)

    def reset(self, seed=None):
        self.calls, self.seen, self.seed = 0, [], seed

    def state(self, obs, task):
        return np.full(8, obs["t"], np.float32)

    def chunk(self, obs, task, env_step, latent=None, reference=False):
        self.calls += 1
        self.seen.append((env_step, "reference" if reference else "base" if latent is None else "steered"))
        value = 0.25 if reference else 2.0   # 2.0 must be clipped before execution
        return self.state(obs, task), np.full((10, 7), value, np.float32)


class FakeSim:
    task_description = "test"

    def __init__(self, success_at=None, perturb_at=None):
        self.success_at, self.perturb_at = success_at, perturb_at

    def reset(self, seed=None):
        self.t, self.executed, self.replans = 0, [], []
        return {"t": 0}, {}

    def notify_replan(self, step):
        self.replans.append(step)

    def step(self, action):
        self.t += 1
        self.executed.append(action)
        done = self.success_at == self.t
        info = {"is_success": done,
                "perturb_step": self.perturb_at if self.perturb_at and self.t >= self.perturb_at else -1}
        return {"t": self.t}, float(done), done, False, info

    def close(self):
        pass


def test_one_decision_executes_one_clipped_chunk_and_counts_vla_calls():
    sim, vla = FakeSim(), FakeVLA()
    env = NoiseLatentEnv(sim, vla, max_steps=280)
    obs, info = env.reset(seed=3)
    assert obs.shape == (15,) == (env.obs_dim,) and vla.seed == 3
    np.testing.assert_allclose(obs[8:], 0.25)            # reference action is part of the observation
    assert info["vla_calls"] == 0 and info["reset_vla_calls"] == 1
    nxt, reward, terminated, truncated, info = env.step(np.zeros(7, np.float32))
    assert sim.t == 10 and info["env_steps"] == 10 and info["decisions"] == 1
    assert np.max(sim.executed) == 1.0                  # 2.0 clipped to the action range
    assert info["vla_calls"] == 3                       # reset reference + steered chunk + next reference
    assert vla.seen == [(0, "reference"), (0, "steered"), (10, "reference")]
    assert sim.replans == [0] and info["steered"] and not terminated and not truncated
    np.testing.assert_allclose(nxt[:8], 10)
    _, _, _, _, info = env.step(None)                   # Frozen Base noise for this decision
    assert vla.seen[3] == (10, "base") and not info["steered"] and info["vla_calls"] == 2


def test_proprio_mode_spends_one_vla_call_per_decision():
    vla = FakeVLA()
    env = NoiseLatentEnv(FakeSim(), vla, obs_mode="proprio")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,) and vla.calls == 0
    assert env.step(np.zeros(7))[4]["vla_calls"] == 1


def test_success_ends_the_chunk_early_without_another_vla_query():
    sim, vla = FakeSim(success_at=13), FakeVLA()
    env = NoiseLatentEnv(sim, vla)
    env.reset(seed=0)
    env.step(np.zeros(7))
    calls = vla.calls
    nxt, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert terminated and not truncated and reward == 1.0 and info["success"]
    assert sim.t == 13 and info["env_steps"] == 3
    assert vla.calls == calls + 1                       # steered chunk only, no terminal reference
    np.testing.assert_allclose(nxt[8:], 0)


def test_time_limit_truncates_and_bootstraps_from_a_real_observation():
    sim, vla = FakeSim(), FakeVLA()
    env = NoiseLatentEnv(sim, vla, max_steps=15)
    env.reset(seed=0)
    env.step(np.zeros(7))
    nxt, _, terminated, truncated, info = env.step(np.zeros(7))
    assert truncated and not terminated and sim.t == 15 and info["env_steps"] == 5
    np.testing.assert_allclose(nxt[8:], 0.25)
    assert vla.seen[-1] == (15, "reference")


def test_perturbed_flag_follows_the_environment():
    env = NoiseLatentEnv(FakeSim(perturb_at=14), FakeVLA())
    env.reset(seed=0)
    env.step(np.zeros(7))
    assert not env.perturbed
    env.step(np.zeros(7))
    assert env.perturbed


@pytest.mark.parametrize("latent", [np.zeros(6), np.full(7, np.inf)])
def test_env_rejects_malformed_latent(latent):
    env = NoiseLatentEnv(FakeSim(), FakeVLA())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(latent)


class ChunkEnv:
    def __init__(self, lengths):
        self.lengths = iter(lengths)

    def step(self, action):
        return None, 0., False, False, {"env_steps": next(self.lengths), "vla_calls": 2}

    def close(self):
        pass


def run_decisions(budget, trainer, count):
    for _ in range(count):
        if trainer.env_steps >= trainer.cfg.total_env_steps:
            break
        info = budget.step(None)[4]
        trainer.env_steps += 1
        trainer.vla_calls += info["vla_calls"]


def test_budget_stops_the_trainer_on_control_steps_and_survives_resume(tmp_path):
    path = tmp_path / "budget.csv"
    trainer = SimpleNamespace(env_steps=0, vla_calls=0, cfg=SimpleNamespace(total_env_steps=1000))
    budget = ControlStepBudget(ChunkEnv([10, 10, 4, 10, 10, 10]), 35, path)
    budget.attach(trainer, resumed=False)
    run_decisions(budget, trainer, 3)
    assert budget.total == 24 and trainer.cfg.total_env_steps == 1000
    budget.close()

    # resume from a checkpoint taken after decision 2; decision 3 is replayed
    trainer = SimpleNamespace(env_steps=2, vla_calls=4, cfg=SimpleNamespace(total_env_steps=1000))
    budget = ControlStepBudget(ChunkEnv([10, 10, 10]), 35, path)
    budget.attach(trainer, resumed=True)
    assert budget.total == 20
    run_decisions(budget, trainer, 10)
    budget.close()
    assert budget.total == 40 and trainer.env_steps == 4 == trainer.cfg.total_env_steps
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [int(r["control_steps"]) for r in rows] == [10, 20, 30, 40]
    assert [int(r["vla_calls"]) for r in rows] == [2, 4, 6, 8]
