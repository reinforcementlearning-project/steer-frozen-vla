"""Download a pinned public checkpoint into this project's private HF cache."""
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
BACKBONE = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
BACKBONE_REVISION = "4cde9d208e19519c36917aa170fffc7e381aa0ad"

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    checkpoint = snapshot_download(MODEL, revision=REVISION)
    backbone_revision = BACKBONE_REVISION
    # The full fine-tuned policy contains the backbone weights; only its config/tokenizer are needed.
    backbone = snapshot_download(BACKBONE, revision=backbone_revision,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.onnx", "*.tflite"])
    record = {"model": MODEL, "revision": REVISION, "checkpoint": checkpoint,
              "backbone": BACKBONE, "backbone_revision": backbone_revision,
              "backbone_path": backbone, "hf_home": os.environ.get("HF_HOME")}
    out = root / "results" / "artifacts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
