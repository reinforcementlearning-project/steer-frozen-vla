"""Guard against moving contacted objects or retaining invalid physics states."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
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


def make_env(distance=.02, robot_contact=False, penetration=False):
    data = Data()
    if robot_contact:
        data.contact = [NS(geom1=5, geom2=99, dist=0)]
    def forward():
        if penetration:
            data.contact = [NS(geom1=5, geom2=88, dist=-.01)] if data.pose[1] > 0 else []
    sim = NS(data=data, forward=forward, model=NS(body_name2id=lambda _: 0,
        geom_name2id=lambda _: 5, geom_id2name=lambda x: "robot0_gripper" if x == 99 else "table"))
    inner = NS(sim=sim, objects_dict={"target": NS(joints=["joint"], root_body="body", contact_geoms=["geom"])},
               _get_observations=lambda **kwargs: {"robot0_eef_pos": np.array([0., .10, .04])})
    raw = NS(_env=NS(env=inner), reset=lambda **kwargs: ({}, {}),
             step=lambda action: ({}, 0., False, False, {}), _format_raw_obs=lambda obs: obs)
    env = DynamicLibero(raw, PerturbationConfig("target", distance, earliest_step=1))
    env.reset(seed=0)
    return env, data


def test_exact_translation_preserves_height_and_orientation():
    env, data = make_env()
    old = data.pose.copy()
    _, _, _, _, info = env.step(np.zeros(7))
    assert data.pose[1] == pytest.approx(.02)
    np.testing.assert_array_equal(data.pose[2:], old[2:])
    assert info["perturb_events"][0]["applied"]
    assert info["perturb_events"][0]["actual_displacement_m"] == pytest.approx(.02)
    env.step(np.zeros(7))
    assert len(env.events) == 1


def test_contacted_object_is_never_moved():
    env, data = make_env(robot_contact=True)
    old = data.pose.copy()
    env.step(np.zeros(7))
    np.testing.assert_array_equal(data.pose, old)
    assert env.events == []


def test_new_penetration_rolls_back_pose_and_velocity():
    env, data = make_env(penetration=True)
    pose, velocity = data.pose.copy(), data.velocity.copy()
    env.step(np.zeros(7))
    np.testing.assert_array_equal(data.pose, pose)
    np.testing.assert_array_equal(data.velocity, velocity)
    assert not env.events[0]["applied"]


def test_zero_distance_does_not_reset_velocity():
    env, data = make_env(distance=0.)
    old = data.velocity.copy()
    env.step(np.zeros(7))
    np.testing.assert_array_equal(data.velocity, old)
