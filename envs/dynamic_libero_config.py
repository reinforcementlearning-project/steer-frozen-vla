"""Dynamic LIBERO v2 configuration, presets and the shared environment factory.

One config object fully describes the simulator side of an experiment (task,
horizon, perturbation timing/direction/magnitude, optional initial-position
shift). It does not know anything about the RL algorithm: Residual-SAC and
DSRL-SAC both build their simulator with ``make_dynamic_libero(cfg)`` and wrap
it with their own policy-side env (``policies/residual_policy.ResidualEnv`` or a
noise-latent env).

Presets and evaluation sets live in ``configs/dynamic_libero_v2.yaml``:

    from envs.dynamic_libero_config import load_preset, load_eval_set, make_dynamic_libero
    cfg = load_preset("train_v1")                    # training distribution
    sim = make_dynamic_libero(cfg)
    for name, cfg in load_eval_set("primary").items():  # official comparison conditions
        ...
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "dynamic_libero_v2.yaml"

# name -> (axis, sign); axis 0 = world x, 1 = world y
DIRECTIONS: Dict[str, Tuple[int, int]] = {"+x": (0, 1), "-x": (0, -1), "+y": (1, 1), "-y": (1, -1)}
# EEF-to-target distance windows (metres). "legacy" is the v1 rule (envs/perturbation.py).
TIMING_WINDOWS_M: Dict[str, Tuple[float, float]] = {
    "legacy": (0.05, 0.18),
    "early": (0.15, 0.18),
    "middle": (0.10, 0.15),
    "late": (0.05, 0.10),
}
# Chunk phase = number of env steps completed modulo n_action_steps when the object is moved.
# With the queue-based SmolVLA wrapper the VLA re-plans from the observation taken after every
# n_action_steps-th step (env_step 0, 10, 20, ...), so phase 0 means "the next observation is a
# re-planning observation" and phase k>0 means the current chunk still has n-k actions to execute
# before the VLA can see the moved object.
CHUNK_PHASE_NAMES = ("pre_replan", "post_replan", "mid_chunk")
SPLITS = ("legacy", "train", "eval_primary", "eval_secondary", "eval_ood", "diagnostic")


def chunk_phase_index(phase, n_action_steps: int) -> Optional[int]:
    if phase is None:
        return None
    if isinstance(phase, str):
        if phase == "pre_replan":
            return 0
        if phase == "post_replan":
            return 1 % n_action_steps
        if phase == "mid_chunk":
            return n_action_steps // 2
        raise ValueError(f"Unknown chunk phase {phase!r}; use one of {CHUNK_PHASE_NAMES} or an int")
    phase = int(phase)
    if not 0 <= phase < n_action_steps:
        raise ValueError(f"chunk_phase must be in [0, {n_action_steps})")
    return phase


@dataclass(frozen=True)
class PerturbationSpec:
    """One pre-contact translation of the target object per episode."""

    timing: str = "legacy"
    directions: Tuple[str, ...] = ("+y",)       # >1 entry: one drawn per episode from the reset seed
    distances_m: Tuple[float, ...] = (0.02,)    # >1 entry: one drawn per episode from the reset seed
    window_m: Optional[Tuple[float, float]] = None  # custom [min, max] EEF distance; overrides timing
    chunk_phase: Any = None                     # None | "pre_replan" | "post_replan" | "mid_chunk" | int
    earliest_step: int = 10
    lift_tolerance_m: float = 0.015            # object counts as lifted beyond this z change
    workspace_limit_m: float = 0.45            # conservative |x|,|y| bound for the moved object
    penetration_tolerance_m: float = 0.003

    def __post_init__(self):
        object.__setattr__(self, "directions", tuple(str(d) for d in self.directions))
        object.__setattr__(self, "distances_m", tuple(float(d) for d in self.distances_m))
        if self.window_m is not None:
            object.__setattr__(self, "window_m", tuple(float(v) for v in self.window_m))
        if self.timing not in TIMING_WINDOWS_M and self.window_m is None:
            raise ValueError(f"Unknown timing {self.timing!r}; use {sorted(TIMING_WINDOWS_M)} or window_m")
        if not self.directions or any(d not in DIRECTIONS for d in self.directions):
            raise ValueError(f"directions must be a non-empty subset of {sorted(DIRECTIONS)}")
        if not self.distances_m or any(not 0 <= d <= 0.10 for d in self.distances_m):
            raise ValueError("distances_m must be non-empty with values in [0, 0.10] m")
        lo, hi = self.window
        if not 0 <= lo < hi:
            raise ValueError("window must satisfy 0 <= min < max")
        if self.earliest_step < 1:
            raise ValueError("earliest_step must be >= 1")
        if self.chunk_phase is not None and not isinstance(self.chunk_phase, (str, int)):
            raise ValueError("chunk_phase must be None, a phase name or an int")

    @property
    def window(self) -> Tuple[float, float]:
        return tuple(self.window_m) if self.window_m is not None else TIMING_WINDOWS_M[self.timing]


@dataclass(frozen=True)
class SpatialSpec:
    """Optional OOD-evaluation-only shift of the target's initial xy position (off by default).

    Official training and evaluation keep LIBERO's original init states (mode "none").

    The LIBERO init state is loaded unchanged; the target is then translated by an
    offset drawn from the reset seed, kept only if it stays inside its BDDL init
    region, creates no penetration or robot contact, stays put during a short
    settle, keeps the target's BDDL "on" predicate true, breaks no BDDL initial
    predicate that held before the shift, and does not satisfy the goal.
    """

    mode: str = "none"              # "none" | "region_uniform"
    max_offset_m: float = 0.025     # per-axis offset drawn from U(-max, max)
    min_offset_m: float = 0.0       # reject offsets with smaller norm
    region_margin_m: float = 0.0    # keep the new xy this far inside the BDDL region
    max_attempts: int = 50
    settle_steps: int = 10
    settle_tolerance_m: float = 0.005

    def __post_init__(self):
        if self.mode not in ("none", "region_uniform"):
            raise ValueError("spatial.mode must be 'none' or 'region_uniform'")
        if self.max_offset_m < 0 or self.min_offset_m < 0 or self.min_offset_m > self.max_offset_m:
            raise ValueError("need 0 <= min_offset_m <= max_offset_m")
        if self.max_attempts < 1 or self.settle_steps < 0:
            raise ValueError("max_attempts >= 1 and settle_steps >= 0 required")


@dataclass(frozen=True)
class DynamicLiberoConfig:
    name: str = "custom"
    split: str = "legacy"
    suite: str = "libero_object"
    task_id: int = 0
    target: str = "alphabet_soup_1"
    max_steps: int = 280
    n_action_steps: int = 10
    episodes: Optional[int] = None    # evaluation episodes (None for training presets)
    eval_seed: Optional[int] = None   # episode i uses reset/noise seed eval_seed + i
    perturbation: Optional[PerturbationSpec] = field(default_factory=PerturbationSpec)
    spatial: SpatialSpec = field(default_factory=SpatialSpec)
    description: str = ""

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        if self.max_steps < 1 or self.n_action_steps < 1:
            raise ValueError("max_steps and n_action_steps must be positive")
        if self.perturbation is not None:
            chunk_phase_index(self.perturbation.chunk_phase, self.n_action_steps)

    def episode_seed(self, episode: int) -> int:
        if self.eval_seed is None:
            raise ValueError(f"{self.name} has no eval_seed (training preset)")
        return int(self.eval_seed) + int(episode)


def config_to_dict(cfg: DynamicLiberoConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    if cfg.perturbation is not None:
        d["perturbation"]["window"] = list(cfg.perturbation.window)
        d["perturbation"]["chunk_phase_index"] = chunk_phase_index(cfg.perturbation.chunk_phase, cfg.n_action_steps)
    return d


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_from_dict(d: Dict[str, Any]) -> DynamicLiberoConfig:
    d = dict(d)
    known = {f.name for f in fields(DynamicLiberoConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    pert = d.get("perturbation", {})
    if pert is not None:
        pert = {k: v for k, v in pert.items() if k not in ("window", "chunk_phase_index")}
        d["perturbation"] = PerturbationSpec(**pert)
    d["spatial"] = SpatialSpec(**(d.get("spatial") or {}))
    return DynamicLiberoConfig(**d)


def _read_yaml(path) -> Dict[str, Any]:
    import yaml
    with open(path or DEFAULT_CONFIG_PATH) as fh:
        doc = yaml.safe_load(fh)
    if doc.get("version") != 2:
        raise ValueError("expected a version: 2 Dynamic LIBERO config file")
    return doc


def parse_condition(token: str) -> Dict[str, Any]:
    """'0cm' | '+y2' | '-x4' | 'rand2' (all four directions, drawn per episode)."""
    if token == "0cm":
        return {"directions": ["+y"], "distances_m": [0.0]}
    if token.startswith("rand"):
        return {"directions": ["+x", "-x", "+y", "-y"], "distances_m": [float(token[4:]) / 100]}
    if token[:2] in DIRECTIONS:
        return {"directions": [token[:2]], "distances_m": [float(token[2:]) / 100]}
    raise ValueError(f"Unknown condition token {token!r}")


def list_presets(path=None):
    doc = _read_yaml(path)
    return sorted(doc.get("presets", {})), sorted(doc.get("eval_sets", {}))


def load_preset(name: str, path=None, **overrides) -> DynamicLiberoConfig:
    doc = _read_yaml(path)
    if name not in doc.get("presets", {}):
        raise KeyError(f"Unknown preset {name!r}; available: {sorted(doc.get('presets', {}))}")
    d = _merge(_merge(doc["defaults"], doc["presets"][name]), overrides)
    return config_from_dict({"name": name, **d})


def load_eval_set(set_name: str, path=None, **overrides) -> Dict[str, DynamicLiberoConfig]:
    """All conditions of one evaluation set, keyed by condition token (e.g. '+y2')."""
    doc = _read_yaml(path)
    sets = doc.get("eval_sets", {})
    if set_name not in sets:
        raise KeyError(f"Unknown eval set {set_name!r}; available: {sorted(sets)}")
    spec = dict(sets[set_name])
    conditions = spec.pop("conditions")
    base = _merge(doc["defaults"], spec)
    out = {}
    for token in conditions:
        d = _merge(base, {"perturbation": parse_condition(token)})
        d = _merge(d, overrides)
        out[token] = config_from_dict({"name": f"{set_name}/{token}", **d})
    return out


def make_dynamic_libero(cfg: DynamicLiberoConfig, libero_env=None):
    """Simulator for one config: LeRobot LiberoEnv wrapped by DynamicLiberoV2 (per env step)."""
    from envs.dynamic_libero import DynamicLiberoV2
    if libero_env is None:
        from envs.libero_env import make_libero
        libero_env = make_libero(cfg.suite, cfg.task_id, max_steps=cfg.max_steps)
    return DynamicLiberoV2(libero_env, cfg)


__all__ = ["DIRECTIONS", "TIMING_WINDOWS_M", "CHUNK_PHASE_NAMES", "PerturbationSpec", "SpatialSpec",
           "DynamicLiberoConfig", "chunk_phase_index", "config_to_dict", "config_from_dict",
           "parse_condition", "list_presets", "load_preset", "load_eval_set", "make_dynamic_libero", "replace"]
