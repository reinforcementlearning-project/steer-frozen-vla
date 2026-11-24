"""Dynamic LIBERO v2 wrapper and config: simulator-free tests with a scripted fake LIBERO."""
from types import SimpleNamespace as NS

import numpy as np
import pytest

from envs.dynamic_libero import (APPLIED_EVENT_KEYS, EVENT_KEYS, INFO_KEYS, NOT_TRIGGERED_REASONS,
                                  REJECTION_REASONS, STATUS_KEYS, STATUS_VALUES, DynamicLiberoV2)
from envs.dynamic_libero_config import (DynamicLiberoConfig, PerturbationSpec, chunk_phase_index,
                                        config_from_dict, config_to_dict, load_eval_set, load_preset,
                                        parse_condition)
from envs.perturbation import DynamicLibero, PerturbationConfig


class Data:
    def __init__(self):
        self.pose = np.array([0., 0., .04, 1., 0., 0., 0.])
        self.velocity = np.full(6, .001)
        self.contact = []

    @property
    def ncon(self):
        return len(self.contact)

    @property
    def body_xpos(self):
        return self.pose[None, :3]

    def get_joint_qpos(self, joint):
        return self.pose.copy()

    def get_joint_qvel(self, joint):
        return self.velocity.copy()

    def set_joint_qpos(self, joint, value):
        self.pose = value.copy()

    def set_joint_qvel(self, joint, value):
        self.velocity = value.copy()


class FakeLibero:
    """EEF approaches the object along -y at `speed` m/step from `start` m; optional contact/lift steps."""

    def __init__(self, start=.25, speed=.01, contact_steps=(), lift_from=None, penetration=False):
        self.data = Data()
        self.start, self.speed = start, speed
        self.contact_steps, self.lift_from, self.penetration = set(contact_steps), lift_from, penetration
        self.t, self.obs_calls = 0, []
        data = self.data

        def forward():
            if penetration:
                data.contact = [NS(geom1=5, geom2=88, dist=-.01)] if abs(data.pose[0]) + abs(data.pose[1]) > 0 else []

        sim = NS(data=data, forward=forward, model=NS(
            body_name2id=lambda _: 0, geom_name2id=lambda _: 5,
            geom_id2name=lambda x: "robot0_gripper" if x == 99 else "table"))
        self.inner = NS(sim=sim, objects_dict={"target": NS(joints=["joint"], root_body="body", contact_geoms=["geom"])},
                        _get_observations=self._get_observations)
        self._env = NS(env=self.inner)

    def eef(self):
        return np.array([0., self.start - self.speed * self.t, .04])

    def _get_observations(self, force_update=False):
        self.obs_calls.append((self.t, force_update))
        return {"robot0_eef_pos": self.eef()}

    def _format_raw_obs(self, obs):
        return {"eef": obs["robot0_eef_pos"].copy()}

    def reset(self, **kwargs):
        self.t = 0
        self.data.pose = np.array([0., 0., .04, 1., 0., 0., 0.])
        self.data.contact = []
        return {"eef": self.eef()}, {}

    def step(self, action):
        self.t += 1
        self.data.contact = [NS(geom1=5, geom2=99, dist=0)] if self.t in self.contact_steps else []
        if self.lift_from is not None and self.t >= self.lift_from:
            self.data.pose[2] = .04 + .02
        return {"eef": self.eef()}, 0., False, False, {}


def cfg(**pert):
    return DynamicLiberoConfig(name="test", target="target", perturbation=PerturbationSpec(**pert))


def run(env, steps=40, seed=0):
    env.reset(seed=seed)
    info = {}
    for _ in range(steps):
        _, _, _, _, info = env.step(np.zeros(7))
    return info


# ------------------------------------------------------------------ 1. legacy reproduction
@pytest.mark.parametrize("axis,sign,name", [(1, 1, "+y"), (1, -1, "-y"), (0, 1, "+x"), (0, -1, "-x")])
@pytest.mark.parametrize("dist", [0.0, .02, .04])
@pytest.mark.parametrize("seed", [0, 7])
def test_legacy_mode_matches_v1_wrapper_exactly(axis, sign, name, dist, seed):
    a, b = FakeLibero(), FakeLibero()
    v1 = DynamicLibero(a, PerturbationConfig("target", dist, axis=axis, direction=sign))
    v2 = DynamicLiberoV2(b, cfg(directions=[name], distances_m=[dist]))
    i1, i2 = run(v1, seed=seed), run(v2, seed=seed)
    e1, e2 = i1["perturb_events"][0], i2["perturb_events"][0]
    for k in ("step", "before", "after", "eef_distance", "applied", "rejection", "actual_displacement_m"):
        assert e1[k] == e2[k], k
    np.testing.assert_array_equal(a.data.pose, b.data.pose)
    assert a.obs_calls == b.obs_calls  # identical simulator observation calls and order
    for k in ("perturb_step", "perturb_dist", "perturb_axis", "perturb_sign"):
        if k in i1:  # main's v1 emits step/dist; newer v1 also axis/sign
            assert i1[k] == i2[k]


# ------------------------------------------------------------------ 2. timing windows
@pytest.mark.parametrize("timing,lo,hi", [("early", .15, .18), ("middle", .10, .15), ("late", .05, .10),
                                          ("legacy", .05, .18)])
def test_trigger_happens_inside_the_configured_window(timing, lo, hi):
    env = DynamicLiberoV2(FakeLibero(start=.30, speed=.004), cfg(timing=timing))
    info = run(env, steps=80)
    e = info["perturb_events"][0]
    assert lo <= e["eef_distance"] <= hi and e["applied"]
    assert e["eef_distance"] > hi - .004 - 1e-9  # first eligible step of the window
    assert info["dynamic_libero"]["status"] == "applied"


def test_window_not_reached_is_reported():
    env = DynamicLiberoV2(FakeLibero(start=.30, speed=.001), cfg(timing="late"))
    info = run(env, steps=50)
    assert info["perturb_events"] == []
    s = info["dynamic_libero"]
    assert s["status"] == "not_triggered" and s["reason"] == "window_not_reached"


def test_window_skipped_between_steps_is_reported():
    env = DynamicLiberoV2(FakeLibero(start=.20, speed=.06), cfg(window_m=[.11, .12], earliest_step=1))
    info = run(env, steps=4)
    assert info["dynamic_libero"]["reason"] == "window_skipped_between_steps"


# ------------------------------------------------------------------ 3. contact / lifted / collisions
def test_robot_contact_blocks_and_is_counted():
    fake = FakeLibero(start=.19, speed=.001, contact_steps=range(1, 100))
    env = DynamicLiberoV2(fake, cfg(timing="early"))
    info = run(env, steps=40)
    s = info["dynamic_libero"]
    assert info["perturb_events"] == [] and s["reason"] == "blocked_robot_contact"
    assert s["diagnostics"]["blocked"]["robot_contact"] > 0


def test_lifted_object_is_never_moved():
    env = DynamicLiberoV2(FakeLibero(start=.19, speed=.001, lift_from=1), cfg(timing="early"))
    info = run(env, steps=40)
    assert info["dynamic_libero"]["reason"] == "blocked_object_lifted"


def test_penetration_rolls_back_and_is_recorded():
    fake = FakeLibero(penetration=True)
    env = DynamicLiberoV2(fake, cfg())
    pose = fake.data.pose.copy()
    info = run(env)
    np.testing.assert_array_equal(fake.data.pose, pose)
    assert info["dynamic_libero"]["status"] == "rejected"
    assert info["dynamic_libero"]["reason"] == "new_penetration_or_robot_contact"


def test_workspace_boundary_rejection():
    fake = FakeLibero()
    fake.data.pose[0] = .44
    fake.reset = lambda **kw: ({"eef": fake.eef()}, {})
    env = DynamicLiberoV2(fake, cfg(directions=["+x"], distances_m=[.02], window_m=[0., 1.]))
    pose = fake.data.pose.copy()
    info = run(env)
    np.testing.assert_array_equal(fake.data.pose, pose)
    assert info["dynamic_libero"]["reason"] == "conservative_workspace_boundary"


def test_at_most_one_attempt_per_episode():
    env = DynamicLiberoV2(FakeLibero(), cfg())
    info = run(env, steps=60)
    assert len(info["perturb_events"]) == 1


# ------------------------------------------------------------------ 4. seed reproducibility
def test_direction_and_distance_sampling_is_seed_reproducible_and_covers_all():
    c = cfg(directions=["+x", "-x", "+y", "-y"], distances_m=[0.0, .02, .04])
    env = DynamicLiberoV2(FakeLibero(), c)
    seen = []
    for seed in range(200):
        env.reset(seed=seed)
        seen.append((env.episode_choice["direction"], env.episode_choice["distance_m"]))
    again = []
    for seed in range(200):
        env.reset(seed=seed)
        again.append((env.episode_choice["direction"], env.episode_choice["distance_m"]))
    assert seen == again
    assert {d for d, _ in seen} == {"+x", "-x", "+y", "-y"} and {m for _, m in seen} == {0.0, .02, .04}


def test_direction_stream_matches_residual_random_direction_runs():
    # The Residual-SAC random-direction runs (branch analysis/residual-s42-diagnostics,
    # train_residual.py --random-direction) draw index rng([seed, 0x5EED]).integers(4) over
    # ((0, 1), (0, -1), (1, 1), (1, -1)); train_v1 must reproduce that stream.
    legacy = ((0, 1), (0, -1), (1, 1), (1, -1))
    v2 = DynamicLiberoV2(FakeLibero(), load_preset("train_v1", target="target"))
    for seed in range(200):
        v2.reset(seed=seed)
        expected = legacy[int(np.random.default_rng([seed, 0x5EED]).integers(4))]
        assert (v2.episode_choice["axis"], v2.episode_choice["sign"]) == expected


# ------------------------------------------------------------------ 5. displacement matches config
@pytest.mark.parametrize("direction,axis,sign", [("+x", 0, 1), ("-x", 0, -1), ("+y", 1, 1), ("-y", 1, -1)])
@pytest.mark.parametrize("dist", [0.0, .02, .04])
def test_actual_displacement_vector_matches_direction_and_size(direction, axis, sign, dist):
    fake = FakeLibero()
    env = DynamicLiberoV2(fake, cfg(directions=[direction], distances_m=[dist]))
    old = fake.data.pose.copy()
    e = run(env)["perturb_events"][0]
    expected = np.zeros(3)
    expected[axis] = sign * dist
    np.testing.assert_allclose(e["actual_displacement_vector_m"], expected, atol=1e-12)
    assert e["actual_displacement_m"] == pytest.approx(dist)
    np.testing.assert_array_equal(fake.data.pose[2:], old[2:])  # height and orientation unchanged
    if dist == 0:
        np.testing.assert_array_equal(fake.data.velocity, np.full(6, .001))  # 0cm keeps velocity (v1 rule)


# ------------------------------------------------------------------ 8. chunk phase
@pytest.mark.parametrize("phase,idx", [("pre_replan", 0), ("post_replan", 1), ("mid_chunk", 5), (3, 3)])
def test_chunk_phase_gate_and_record(phase, idx):
    env = DynamicLiberoV2(FakeLibero(start=.30, speed=.004), cfg(timing="legacy", chunk_phase=phase))
    e = run(env, steps=80)["perturb_events"][0]
    assert e["step"] % 10 == idx and e["chunk_phase"] == idx
    assert e["steps_until_replan_obs"] == (10 - idx) % 10


def test_chunk_phase_mismatch_reason_when_window_too_short():
    env = DynamicLiberoV2(FakeLibero(start=.30, speed=.004), cfg(window_m=[.171, .179], chunk_phase="pre_replan"))
    info = run(env, steps=80)
    assert info["dynamic_libero"]["reason"] == "blocked_chunk_phase_mismatch"


def test_reported_replans_are_logged_next_to_modular_phase():
    env = DynamicLiberoV2(FakeLibero(), cfg())
    env.reset(seed=0)
    for t in range(40):
        if t % 10 == 0:
            env.notify_replan(t)
        _, _, _, _, info = env.step(np.zeros(7))
    e = info["perturb_events"][0]
    assert e["chunk_phase_from_reported_replan"] == e["chunk_phase"]


def test_phase_names():
    assert [chunk_phase_index(p, 10) for p in ("pre_replan", "post_replan", "mid_chunk", None)] == [0, 1, 5, None]
    with pytest.raises(ValueError):
        chunk_phase_index(10, 10)


# ------------------------------------------------------------------ config / presets
def test_presets_and_eval_sets_load_and_round_trip():
    train = load_preset("train_v1")
    assert train.split == "train" and train.perturbation.directions == ("+x", "-x", "+y", "-y")
    assert train.perturbation.window == (.05, .18) and train.max_steps == 280
    primary = load_eval_set("primary")
    assert list(primary) == ["0cm", "+x2", "-x2", "+y2", "-y2"]
    assert all(c.split == "eval_primary" and c.episodes == 50 and c.eval_seed == 0 for c in primary.values())
    assert primary["0cm"].perturbation.distances_m == (0.0,)
    assert load_eval_set("timing_late")["+y2"].perturbation.window == (.05, .10)
    assert load_eval_set("horizon400")["-y4"].max_steps == 400
    assert load_eval_set("ood_spatial")["+x2"].spatial.mode == "region_uniform"
    for c in (train, *primary.values()):
        assert config_from_dict(config_to_dict(c)) == c
    assert load_preset("no_perturbation").perturbation is None


def test_splits_are_disjoint():
    names = {"primary": "eval_primary", "distance_4cm": "eval_secondary", "timing_late": "eval_secondary",
             "ood_spatial": "eval_ood", "horizon400": "diagnostic"}
    for s, split in names.items():
        assert {c.split for c in load_eval_set(s).values()} == {split}


def test_invalid_configs_rejected():
    for bad in ({"directions": ["+z"]}, {"distances_m": [.2]}, {"timing": "soon"}, {"window_m": [.2, .1]}):
        with pytest.raises(ValueError):
            PerturbationSpec(**bad)
    with pytest.raises(ValueError):
        DynamicLiberoConfig(split="test")
    with pytest.raises(ValueError):
        parse_condition("+q2")


def test_no_perturbation_config_is_a_pass_through():
    fake = FakeLibero()
    env = DynamicLiberoV2(fake, DynamicLiberoConfig(target="target", perturbation=None))
    info = run(env)
    assert info["perturb_events"] == [] and info["dynamic_libero"]["status"] == "disabled"
    assert fake.obs_calls == []


# ------------------------------------------------------------------ info contract
def _check_status(s):
    assert set(STATUS_KEYS) <= set(s) and s["status"] in STATUS_VALUES
    if s["status"] == "not_triggered":
        assert s["reason"] in NOT_TRIGGERED_REASONS
    if s["status"] == "rejected":
        assert s["reason"] in REJECTION_REASONS
    if s["status"] in ("applied", "rejected"):
        assert {"step", "chunk_phase", "eef_distance", "diagnostics"} <= set(s)


@pytest.mark.parametrize("fake,c", [
    (lambda: FakeLibero(), lambda: cfg()),                                             # applied
    (lambda: FakeLibero(penetration=True), lambda: cfg()),                            # rejected
    (lambda: FakeLibero(start=.30, speed=.001), lambda: cfg(timing="late")),          # not triggered
    (lambda: FakeLibero(), lambda: DynamicLiberoConfig(target="target", perturbation=None)),  # disabled
])
def test_info_contract_keys(fake, c):
    env = DynamicLiberoV2(fake(), c())
    _, info = env.reset(seed=0)
    _check_status(info["dynamic_libero"])
    for _ in range(40):
        _, _, _, _, info = env.step(np.zeros(7))
        assert set(INFO_KEYS) <= set(info)
        _check_status(info["dynamic_libero"])
    for e in info["perturb_events"]:
        assert set(EVENT_KEYS) <= set(e)
        if e["applied"]:
            assert set(APPLIED_EVENT_KEYS) <= set(e)
