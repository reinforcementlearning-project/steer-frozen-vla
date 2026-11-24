"""Verify action units, gripper isolation, transition alignment and time limits."""
from types import SimpleNamespace
import numpy as np
import pytest
from policies.residual_policy import ResidualEnv


class Base:
    def reset(self, seed=None):
        self.calls = 0

    def action(self, obs, task, step):
        self.calls += 1
        return np.full(8, step, np.float32), np.full(7, step / 10, np.float32)

    def observation(self, obs, task):
        import torch
        return {"observation.state": torch.zeros(1, 8)}


class Env:
    task_description = "test"

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        self.executed = action
        return {}, 0., False, False, {"is_success": False}


def test_zero_residual_and_transition_alignment():
    raw = Env()
    env = ResidualEnv(raw, Base(), max_steps=2)
    state, _ = env.reset(seed=0)
    expected = state[-7:].copy()
    nxt, _, terminated, truncated, info = env.step(np.zeros(7))
    np.testing.assert_array_equal(raw.executed, expected)
    np.testing.assert_array_equal(info["base_action"], expected)
    assert not terminated and not truncated
    assert info["vla_calls"] == 2  # initial query + next-state query
    np.testing.assert_allclose(nxt[-7:], .1)
    _, _, terminated, truncated, _ = env.step(np.zeros(7))
    assert truncated and not terminated


def test_squashed_sac_action_not_double_squashed_and_gripper_preserved():
    raw = Env()
    env = ResidualEnv(raw, Base(), alpha=.1)
    env.reset()
    env.step(np.ones(7, np.float32))
    np.testing.assert_allclose(raw.executed[:6], .1)
    assert raw.executed[-1] == 0


@pytest.mark.parametrize("value", [np.zeros(6), np.full(7, np.nan), np.full(7, 2.)])
def test_invalid_residual_rejected(value):
    env = ResidualEnv(Env(), Base())
    env.reset()
    with pytest.raises(ValueError):
        env.step(value)
