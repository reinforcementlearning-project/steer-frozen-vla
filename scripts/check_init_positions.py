"""Read-only: object positions after reset for every LIBERO init state, per task."""
import sys, json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from envs.libero_env import make_libero
suite = sys.argv[1]; tasks = [int(t) for t in sys.argv[2].split(",")]; n = int(sys.argv[3])
out = {}
for t in tasks:
    env = make_libero(suite, t)
    pos = {}
    for i in range(n):
        env.reset(seed=i)
        inner = env._env.env
        for name, obj in inner.objects_dict.items():
            bid = inner.sim.model.body_name2id(obj.root_body)
            pos.setdefault(name, []).append(np.asarray(inner.sim.data.body_xpos[bid]).copy())
        eef = np.asarray(inner._get_observations()["robot0_eef_pos"])
        pos.setdefault("_eef", []).append(eef.copy())
    lang = getattr(env, "task_description", None) or getattr(env, "task", "")
    rows = {k: {"xy_std_cm": (np.array(v)[:, :2].std(0) * 100).round(2).tolist(),
                "xy_range_cm": (np.ptp(np.array(v)[:, :2], 0) * 100).round(2).tolist(),
                "unique_xy_mm": len({tuple(np.round(p[:2] * 1000).astype(int)) for p in v})} for k, v in pos.items()}
    out[f"{suite}/{t}"] = {"task": str(lang), "n_init": n, "objects": rows}
    print(json.dumps({f"{suite}/{t}": out[f"{suite}/{t}"]}), flush=True)
    env.close()
Path(sys.argv[4]).write_text(json.dumps(out, indent=1))
