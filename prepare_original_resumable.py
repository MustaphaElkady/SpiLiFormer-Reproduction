import argparse
import csv
import fcntl
import os
from pathlib import Path
from collections import Counter

import numpy as np
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

CLASSES = "airplane automobile bird cat deer dog frog horse ship truck".split()
EXPECTED = 1000


def valid_npz(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            if not z.files:
                return False

            for key in z.files:
                arr = z[key]
                if arr.size == 0:
                    return False
                if not np.issubdtype(arr.dtype, np.number):
                    return False
                if not np.isfinite(arr).all():
                    return False

        return True
    except Exception:
        return False


def create_or_read_manifest(raw, root):
    manifest = root / "conversion_manifest.csv"

    if manifest.exists():
        with manifest.open() as f:
            rows = list(csv.DictReader(f))
    else:
        rows = []

        for cls in CLASSES:
            files = sorted((raw / cls).glob("*.aedat"))

            if len(files) != EXPECTED:
                raise RuntimeError(
                    f"{cls}: expected 1000 files, found {len(files)}"
                )

            for src in files:
                rows.append({
                    "class": cls,
                    "filename": src.name,
                    "source_bytes": src.stat().st_size
                })

        tmp = root / "conversion_manifest.tmp"

        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class", "filename", "source_bytes"]
            )
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp, manifest)

    counts = Counter(row["class"] for row in rows)

    if len(rows) != 10000 or any(counts[c] != 1000 for c in CLASSES):
        raise RuntimeError("Invalid conversion manifest")

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=Path)
    parser.add_argument("root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--delete-source-after-verify",
        action="store_true"
    )
    args = parser.parse_args()

    raw = args.raw.resolve()
    root = args.root.resolve()

    root.parent.mkdir(parents=True, exist_ok=True)

    with (root.parent / (root.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        root.mkdir(parents=True, exist_ok=True)

        ready = root / "READY-official-t16"

        if ready.exists():
            print("Dataset already ready:", root)
            return

        rows = create_or_read_manifest(raw, root)

        if args.dry_run:
            print("PREFLIGHT PASSED")
            print("Manifest entries:", len(rows))
            print("Classes:", len(CLASSES))
            print("Samples per class:", EXPECTED)
            print("No conversion or source deletion performed.")
            return

        if not args.delete_source_after_verify:
            raise RuntimeError(
                "Full conversion requires explicit source deletion approval"
            )

        events_root = root / "events_np"

        for cls in CLASSES:
            (events_root / cls).mkdir(parents=True, exist_ok=True)

        total = len(rows)

        for index, row in enumerate(rows, start=1):
            cls = row["class"]
            filename = row["filename"]

            src = raw / cls / filename
            dst = events_root / cls / (Path(filename).stem + ".npz")
            tmp = dst.with_name(dst.stem + ".partial.npz")

            if src.exists():
                if src.stat().st_size != int(row["source_bytes"]):
                    raise RuntimeError(
                        f"Original file size changed: {src}"
                    )

            if dst.exists() and valid_npz(dst):
                if tmp.exists():
                    tmp.unlink()

                if src.exists():
                    src.unlink()
                    print("SOURCE REMOVED:", src, flush=True)

                print(f"[{index}/{total}] VERIFIED:", dst, flush=True)
                continue

            if not src.exists():
                raise RuntimeError(
                    f"Missing source and valid NPZ: {src}"
                )

            if dst.exists():
                dst.unlink()

            if tmp.exists():
                tmp.unlink()

            print(f"[{index}/{total}] CONVERTING:", src, flush=True)

            try:
                CIFAR10DVS.read_aedat_save_to_np(
                    str(src), str(tmp)
                )

                if not valid_npz(tmp):
                    raise RuntimeError(
                        f"NPZ integrity validation failed: {tmp}"
                    )

                os.replace(tmp, dst)

            except Exception:
                if tmp.exists():
                    tmp.unlink()
                raise

            # Delete source only after NPZ has been validated
            src.unlink()

            print(
                f"[{index}/{total}] VERIFIED AND SAVED:",
                dst,
                flush=True
            )

        # Full dataset validation using official loader
        print("===== FINAL DATASET VALIDATION =====", flush=True)

        dataset = CIFAR10DVS(
            root=str(root),
            data_type="frame",
            frames_number=16,
            split_by="number"
        )

        if dataset.classes != CLASSES or len(dataset) != 10000:
            raise RuntimeError("Incomplete official frame dataset")

        for i in range(len(dataset)):
            frames, label = dataset[i]

            if frames.shape != (16, 2, 128, 128):
                raise RuntimeError(
                    f"Invalid frame shape at sample {i}"
                )

            if not np.isfinite(frames).all():
                raise RuntimeError(
                    f"Invalid frame values at sample {i}"
                )

            if (i + 1) % 500 == 0:
                print(
                    f"Validated {i + 1}/10000 samples",
                    flush=True
                )

        ready_tmp = root / "READY-official-t16.tmp"
        ready_tmp.write_text("10000 recordings\n")
        os.replace(ready_tmp, ready)

        print("===================================")
        print("OFFICIAL DATASET READY")
        print("Samples:", len(dataset))
        print("Classes:", len(dataset.classes))
        print("Frames: 16")
        print("Split: number")
        print("===================================")


if __name__ == "__main__":
    main()
