"""Write only project-local LIBERO path config, using installed assets read-only."""
import os
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
base = Path(os.environ.get("RF_BASE", str(Path.home() / "rtx3090-migration/sy")))
package = base / ".venv/lib/python3.12/site-packages/libero/libero"
paths = {"benchmark_root": package, "bddl_files": package / "bddl_files",
         "init_states": package / "init_files", "assets": base / ".cache/libero/assets",
         "datasets": Path.home() / "rf/data"}
for key in ("benchmark_root", "bddl_files", "init_states", "assets"):
    if not paths[key].is_dir():
        raise FileNotFoundError(paths[key])
out = root / "configs/libero/config.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists():
    raise FileExistsError(f"Refusing to replace {out}")
out.write_text(yaml.safe_dump({k: str(v) for k, v in paths.items()}))
print(out)
