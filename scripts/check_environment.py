"""Read-only dependency and source provenance report; never loads model weights."""
import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
import torch


def main():
    report = {"python": sys.version, "packages": {}, "source_sha256": {},
              "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    for name in ("torch", "transformers", "lerobot", "hf-libero", "robosuite", "mujoco", "num2words", "docopt"):
        report["packages"][name] = importlib.metadata.version(name)
    for name in ("lerobot.policies.smolvla.modeling_smolvla", "lerobot.policies.smolvla.smolvlm_with_expert",
                 "lerobot.envs.libero", "lerobot.processor.env_processor"):
        path = Path(importlib.import_module(name).__file__)
        report["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
