"""Prep step 3: pack the cropped images into WebDataset tar shards for the GPU box.

Each sample is <key>.jpg + <key>.json ({"label", "make_model", "make", "generation", "source",
"label_idx", "make_model_idx", "make_idx"}). Class indices are fixed in classes.json so every
run (and the app) agrees on them.

Output (default ~/Downloads/car_shards):
    {train,val,test}-000000.tar ...   ~--shard-size images each, shuffled within split
    classes.json                      label / make_model / make -> index, plus label -> make_model, make
    manifest.csv                      the rows that were packed (with crop box and detector score)

Upload the whole folder to the GPU box (see TRAINING.md).

Usage:
    python3.11 prep_pack.py
"""

import argparse
import csv
import io
import json
import random
import tarfile
from collections import Counter
from pathlib import Path

from prep_crop import CROPS_CSV
from prep_manifest import MANIFEST_PATH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "car_shards")
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--min-train", type=int, default=10, help="drop Car Models 3887 labels with fewer train images left")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_bytes(tar, name, data):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def main():
    args = parse_args()
    crops = {r["path"]: r for r in csv.DictReader(open(CROPS_CSV))}
    rows = []
    for r in csv.DictReader(open(MANIFEST_PATH)):
        c = crops.get(r["path"])
        if c and c["crop_path"]:
            rows.append({**r, **{k: c[k] for k in ("crop_path", "detected", "score", "x0", "y0", "x1", "y1")}})
    print(f"{len(rows):,} cropped images to pack")

    # Car Models 3887 labels left with too few train images after cropping (press galleries that were
    # mostly interior shots) are dropped entirely. VMMRdb labels are kept as-is, matching the README split.
    n_train = Counter(r["label"] for r in rows if r["split"] == "train")
    lost = {r["label"] for r in rows if r["source"] != "vmmrdb" and n_train[r["label"]] < args.min_train}
    if lost:
        print(f"Dropping {len(lost)} labels with < {args.min_train} train images: {sorted(lost)[:10]}")
        rows = [r for r in rows if r["label"] not in lost]

    labels = sorted({r["label"] for r in rows})
    make_models = sorted({r["make_model"] for r in rows})
    makes = sorted({r["make"] for r in rows})
    idx = {"label": {l: i for i, l in enumerate(labels)},
           "make_model": {m: i for i, m in enumerate(make_models)},
           "make": {m: i for i, m in enumerate(makes)}}
    parents = {r["label"]: (r["make_model"], r["make"]) for r in rows}
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump({**idx, "label_parents": {l: {"make_model": mm, "make": mk} for l, (mm, mk) in parents.items()}},
              open(args.out / "classes.json", "w"), indent=1)

    rng = random.Random(args.seed)
    for split in ("train", "val", "test"):
        split_rows = [r for r in rows if r["split"] == split]
        rng.shuffle(split_rows)
        for s in range(0, len(split_rows), args.shard_size):
            path = args.out / f"{split}-{s // args.shard_size:06d}.tar"
            with tarfile.open(path, "w") as tar:
                for k, r in enumerate(split_rows[s:s + args.shard_size], start=s):
                    key = f"{split}{k:07d}"
                    meta = {f: r[f] for f in ("label", "make_model", "make", "generation", "source")}
                    meta.update(label_idx=idx["label"][r["label"]], make_model_idx=idx["make_model"][r["make_model"]],
                                make_idx=idx["make"][r["make"]], orig_path=r["path"])
                    add_bytes(tar, f"{key}.jpg", Path(r["crop_path"]).read_bytes())
                    add_bytes(tar, f"{key}.json", json.dumps(meta).encode())
        print(f"  {split:<5} {len(split_rows):>7,} images -> {-(-len(split_rows) // args.shard_size)} shards")

    with open(args.out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    src = Counter(r["source"] for r in rows)
    print(f"\n{len(labels):,} labels, {len(make_models):,} make_models, {len(makes)} makes; "
          f"by source: {dict(src)}\nWrote {args.out}")


if __name__ == "__main__":
    main()
