#!/usr/bin/env python3
"""Export a model state_dict and manifest from a trusted project checkpoint."""
import argparse
import json
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()

if not args.checkpoint.is_file():
    raise FileNotFoundError(args.checkpoint)

# Only load checkpoints created by your own trusted training run.
ckpt = torch.load(str(args.checkpoint), map_location="cpu")
if "model" not in ckpt:
    raise KeyError("Checkpoint has no 'model' state_dict.")

args.output_dir.mkdir(parents=True, exist_ok=True)
weights = args.output_dir / "model_state_dict.pth"
torch.save(ckpt["model"], str(weights))

manifest = {
    "source_checkpoint": args.checkpoint.name,
    "saved_epoch_index": ckpt.get("epoch"),
    "best_test_top1": ckpt.get("max_test_acc1"),
    "top5_at_best_test_top1": ckpt.get("test_acc5_at_max_test_acc1"),
    "beta_scheduler_state": ckpt.get("beta_scheduler_state"),
    "weights_file": weights.name,
    "format": "PyTorch state_dict",
    "note": "Load with the matching project model implementation."
}
(args.output_dir / "checkpoint_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print("Exported:", weights)
