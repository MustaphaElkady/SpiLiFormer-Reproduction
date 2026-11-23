"""Stage an official SpikingJelly frame dataset from already extracted recordings."""
import argparse
import fcntl
import os
import tempfile
from pathlib import Path

import numpy as np
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

CLASSES = "airplane automobile bird cat deer dog frog horse ship truck".split()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / "READY-official-t16").is_file():
            print("Reusing prepared dataset:", root)
            return
        if root.exists():
            raise RuntimeError(f"Choose a fresh preparation directory; already exists: {root}")
        with tempfile.TemporaryDirectory(dir=root.parent) as tmp:
            stage = Path(tmp) / "data"
            for name in CLASSES:
                files = sorted((args.raw / name).glob("*.aedat"))
                if len(files) != 1000:
                    raise ValueError(f"{name}: found {len(files)} original recordings, expected 1000")
                dest = stage / "events_np" / name
                dest.mkdir(parents=True)
                for src in files:
                    CIFAR10DVS.read_aedat_save_to_np(str(src), str(dest / (src.stem + ".npz")))
            dataset = CIFAR10DVS(root=str(stage), data_type="frame", frames_number=16, split_by="number")
            if dataset.classes != CLASSES or len(dataset) != 10000:
                raise RuntimeError("Incomplete frame dataset")
            for frames, label in dataset:
                if frames.shape != (16, 2, 128, 128) or not np.isfinite(frames).all():
                    raise ValueError(f"Invalid frames in class {label}")
            (stage / "READY-official-t16").write_text("10000 recordings\n")
            os.rename(stage, root)
        print("Official frame dataset ready:", root)


if __name__ == "__main__":
    main()
