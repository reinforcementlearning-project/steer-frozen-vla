"""Shared LIBERO construction; no duplicated simulator implementation."""
from lerobot.envs.libero import LiberoEnv, _get_suite


def make_libero(suite="libero_object", task_id=0, episode_index=0, max_steps=280):
    # hf-libero otherwise ignores config.assets during discovery and downloads
    # another copy to ~/.cache/libero. Reuse the configured assets in this process.
    import libero.libero as libero
    libero._assets_path_cache = libero.get_libero_path("assets")
    return LiberoEnv(task_suite=_get_suite(suite), task_id=task_id,
        task_suite_name=suite, obs_type="pixels_agent_pos", episode_index=episode_index,
        episode_length=max_steps, observation_height=256, observation_width=256,
        control_freq=20, control_mode="relative", num_steps_wait=10)
