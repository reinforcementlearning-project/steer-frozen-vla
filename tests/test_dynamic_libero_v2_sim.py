"""Dynamic LIBERO v2 on the real LIBERO simulator (no VLA, scripted EEF approach).

Opt-in because each test builds a rendering LIBERO env:
    DLV2_SIM=1 python -m pytest -q tests/test_dynamic_libero_v2_sim.py
(run after `source scripts/server_env.sh <gpu>` so EGL rendering and LIBERO paths are set).
"""
import hashlib
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("DLV2_SIM") != "1", reason="set DLV2_SIM=1 to run simulator tests")


@pytest.fixture(scope="module")
def libero():
    from envs.libero_env import make_libero
    env = make_libero("libero_object", 0, max_steps=280)
    yield env
    env.close()


def approach_action(sim_env, height=0.03, gain=8.0, limit=0.6):
    """Move the EEF towards a point `height` above the target body centre, gripper open."""
    inner = sim_env._env.env
    obj = inner.objects_dict["alphabet_soup_1"]
    target = np.asarray(inner.sim.data.body_xpos[inner.sim.model.body_name2id(obj.root_body)]) + [0, 0, height]
    eef = np.asarray(inner.sim.data.get_site_xpos(inner.robots[0].controller.eef_name))
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip(gain * (target - eef), -limit, limit)
    a[6] = -1.0
    return a


def rollout(env, seed, steps=120):
    obs, info = env.reset(seed=seed)
    for _ in range(steps):
        obs, _, terminated, _, info = env.step(approach_action(env))
        if terminated:
            break
    return info


def state_hash(env):
    return hashlib.sha256(np.asarray(env._env.env.sim.get_state().flatten()).tobytes()).hexdigest()


def test_eef_site_matches_observation(libero):
    libero.reset(seed=0)
    inner = libero._env.env
    site = np.asarray(inner.sim.data.get_site_xpos(inner.robots[0].controller.eef_name))
    # the observable is refreshed at control rate; differences are micrometres
    np.testing.assert_allclose(site, inner._get_observations()["robot0_eef_pos"], atol=1e-4)


def test_target_starts_inside_its_bddl_region_in_world_xy(libero):
    """Region ranges are world xy (assumption used by the spatial mode); checked on all 50 init states.

    Not every object satisfies its region in LIBERO's shipped init states (e.g. basket_1 lies up
    to ~0.5 cm outside floor_bin_region), so only the target is asserted.
    """
    inner = libero._env.env
    for sid in range(50):
        libero.init_state_id = sid
        libero.reset(seed=sid)
        pp = inner.parsed_problem
        pred = next(p for p in pp["initial_state"] if p[0] == "on" and p[1] == "alphabet_soup_1")
        x0, y0, x1, y1 = pp["regions"][pred[2]]["ranges"][0]
        body = inner.objects_dict["alphabet_soup_1"].root_body
        x, y, _ = inner.sim.data.body_xpos[inner.sim.model.body_name2id(body)]
        assert x0 <= x <= x1 and y0 <= y <= y1, sid


@pytest.mark.parametrize("axis,sign,name,dist", [(1, 1, "+y", .02), (0, -1, "-x", .04), (1, 1, "+y", 0.0)])
def test_legacy_mode_reproduces_v1_on_the_simulator(libero, axis, sign, name, dist):
    from envs.dynamic_libero import DynamicLiberoV2
    from envs.dynamic_libero_config import DynamicLiberoConfig, PerturbationSpec
    from envs.perturbation import DynamicLibero, PerturbationConfig
    v1 = DynamicLibero(libero, PerturbationConfig("alphabet_soup_1", dist, axis=axis, direction=sign))
    libero.init_state_id = 3  # same LIBERO init state for both wrappers
    i1 = rollout(v1, seed=3)
    h1 = state_hash(libero)
    v2 = DynamicLiberoV2(libero, DynamicLiberoConfig(perturbation=PerturbationSpec(directions=[name], distances_m=[dist])))
    libero.init_state_id = 3
    i2 = rollout(v2, seed=3)
    h2 = state_hash(libero)
    e1, e2 = i1["perturb_events"][0], i2["perturb_events"][0]
    for k in ("step", "before", "after", "eef_distance", "applied", "rejection"):
        assert e1[k] == e2[k], k
    assert h1 == h2  # identical full simulator state after the same scripted rollout


@pytest.mark.parametrize("timing", ["early", "middle", "late"])
def test_timing_windows_trigger_inside_window_on_simulator(libero, timing):
    from envs.dynamic_libero_config import TIMING_WINDOWS_M, load_preset, make_dynamic_libero
    cfg = load_preset("legacy_y2", perturbation={"timing": timing, "directions": ["-x"], "distances_m": [0.02]})
    env = make_dynamic_libero(cfg, libero_env=libero)
    info = rollout(env, seed=1)
    lo, hi = TIMING_WINDOWS_M[timing]
    s = info["dynamic_libero"]
    assert s["status"] == "applied", s
    e = info["perturb_events"][0]
    assert lo <= e["eef_distance"] <= hi
    np.testing.assert_allclose(e["actual_displacement_vector_m"], [-0.02, 0, 0], atol=1e-6)


def test_chunk_phase_gate_on_simulator(libero):
    from envs.dynamic_libero_config import load_preset, make_dynamic_libero
    for phase, idx in (("pre_replan", 0), ("post_replan", 1), ("mid_chunk", 5)):
        env = make_dynamic_libero(load_preset("legacy_y2", perturbation={"chunk_phase": phase}), libero_env=libero)
        e = rollout(env, seed=2)["perturb_events"][0]
        assert e["applied"] and e["step"] % 10 == idx == e["chunk_phase"]


def test_spatial_mode_valid_reproducible_and_leaves_libero_init_state_untouched(libero):
    from envs.dynamic_libero_config import load_eval_set, load_preset, make_dynamic_libero
    plain = make_dynamic_libero(load_preset("no_perturbation"), libero_env=libero)
    spatial_cfg = load_eval_set("ood_spatial")["0cm"]
    spatial = make_dynamic_libero(spatial_cfg, libero_env=libero)
    for seed in range(6):
        # LIBERO cycles init states per reset; reuse a fresh index for both runs
        libero.init_state_id = seed
        plain.reset(seed=seed)
        before = state_hash(libero)
        libero.init_state_id = seed
        _, info = spatial.reset(seed=seed)
        rec = info["dynamic_libero"]["spatial"]
        assert rec["applied"], rec
        x0, y0, x1, y1 = rec["region_xy"]
        x, y, _ = rec["after_settle"]
        assert x0 <= x <= x1 and y0 <= y <= y1
        assert np.max(np.abs(rec["offset_m"])) <= spatial_cfg.spatial.max_offset_m
        inner = libero._env.env
        false_before = {tuple(p) for p in rec["init_predicates_false_before_shift"]}
        for p in inner.parsed_problem["initial_state"]:
            if tuple(p) not in false_before:
                assert inner._eval_predicate(p), p
        assert ("on", "alphabet_soup_1", "floor_target_object_region") not in false_before
        assert not inner._check_success()
        libero.init_state_id = seed
        _, info2 = spatial.reset(seed=seed)
        assert info2["dynamic_libero"]["spatial"]["offset_m"] == rec["offset_m"]
        libero.init_state_id = seed
        plain.reset(seed=seed)
        assert state_hash(libero) == before  # original init state path unchanged


class FakeVLA:
    """Stand-in for FrozenSmolVLA (chunked, re-plans every n steps) so the interface test needs no GPU."""

    def __init__(self, sim, n=10):
        self.sim, self.n, self.calls, self.queue = sim, n, 0, []

    def reset(self, seed):
        self.calls, self.queue = 0, []

    def observation(self, obs, task):
        import torch
        return {"observation.state": torch.zeros(1, 8)}

    def action(self, obs, task, env_step):
        if not self.queue:
            self.calls += 1
            self.queue = [approach_action(self.sim) for _ in range(self.n)]
            if hasattr(self.sim, "notify_replan"):
                self.sim.notify_replan(env_step)
        return np.zeros(8, dtype=np.float32), self.queue.pop(0)


class ChunkLatentEnv:
    """Minimal DSRL-style env: one RL action (noise latent) per VLA chunk, same simulator object."""

    def __init__(self, sim, vla, n=10, max_steps=280):
        self.sim, self.vla, self.n, self.max_steps = sim, vla, n, max_steps

    def reset(self, seed=None):
        self.t = 0
        self.vla.reset(seed)
        obs, info = self.sim.reset(seed=seed)
        return np.zeros(8, dtype=np.float32), info

    def step(self, latent):
        total, info, terminated = 0.0, {}, False
        for _ in range(self.n):
            _, a = self.vla.action(None, None, self.t)
            _, r, terminated, _, info = self.sim.step(a)
            self.t += 1
            total += r
            if terminated or self.t >= self.max_steps:
                break
        return np.zeros(8, dtype=np.float32), total, terminated, self.t >= self.max_steps, info


def test_residual_and_dsrl_style_wrappers_share_one_simulator_interface(libero):
    from envs.dynamic_libero_config import load_preset, make_dynamic_libero
    from policies.residual_policy import ResidualEnv
    cfg = load_preset("train_v1")
    sim = make_dynamic_libero(cfg, libero_env=libero)
    # Residual-SAC side (unchanged policies/residual_policy.py)
    res = ResidualEnv(sim, FakeVLA(sim), max_steps=cfg.max_steps)
    res.reset(seed=4)
    for _ in range(60):
        _, _, term, trunc, info = res.step(np.zeros(7, dtype=np.float32))
        if term or trunc:
            break
    s_res = info["dynamic_libero"]
    # DSRL-SAC side (chunk-level latent actions)
    dsrl = ChunkLatentEnv(sim, FakeVLA(sim), max_steps=cfg.max_steps)
    dsrl.reset(seed=4)
    for _ in range(6):
        _, _, term, trunc, info = dsrl.step(np.zeros(32, dtype=np.float32))
        if term or trunc:
            break
    s_dsrl = info["dynamic_libero"]
    assert s_res["choice"] == s_dsrl["choice"]  # same seed -> same sampled direction/distance
    for s in (s_res, s_dsrl):
        assert s["config"] == "train_v1" and s["split"] == "train" and s["status"] in ("applied", "rejected", "not_triggered")
    e = info["perturb_events"][0]
    assert e["chunk_phase_from_reported_replan"] == e["chunk_phase"]  # replans happen at multiples of n
