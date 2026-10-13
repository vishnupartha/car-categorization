"""Prep step 1: merge VMMRdb + exotic cars from Car Models 3887 into one manifest CSV.

VMMRdb rows reuse the existing make_model split from probe.py (same train/val/test images),
labelled with their year bucket from year_buckets.json (e.g. "honda_civic_2001-2005").

Car Models 3887 (huggingface.co/datasets/Unit293/car_models_3887, scraped from autoevolution)
only contributes cars VMMRdb is missing or barely has: exotic makes (Pagani, Koenigsegg, Ferrari,
...) plus halo models from mainstream makes (Porsche 918, Lexus LFA, ...). Their generation comes
from the gallery folder (".../PAGANI_Huayra/2012Present" -> "2012-present"). Its train/test zips
are used as the split, with 10% of train held out as val.

VMMRdb images of the exotic makes, and of the halo models listed as replaced, are dropped
(VMMRdb's Ferraris are a handful of mislabel-prone photos).

Output: data/manifest.csv  (path, source, split, make, model, generation, make_model, label)

Usage:
    python3.11 prep_manifest.py --cm-root ~/Downloads/car_models_3887
    # expects <cm-root>/data_full.csv and the zips extracted to <cm-root>/train and <cm-root>/test
"""

import argparse
import ast
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import BUCKETS_PATH, DATA_DIR, load_labels, make_model, split_path

MANIFEST_PATH = DATA_DIR / "manifest.csv"

# Car Models 3887 brand (upper-cased) -> our make name. Every model of these brands is taken.
EXOTIC_BRANDS = {
    "PAGANI": "pagani", "KOENIGSEGG": "koenigsegg", "BUGATTI": "bugatti", "MCLAREN": "mclaren",
    "FERRARI": "ferrari", "LAMBORGHINI": "lamborghini", "RIMAC": "rimac", "ZENVO": "zenvo",
    "SPYKER": "spyker", "PININFARINA": "pininfarina",
}

# (brand, regex on the model name, our make, VMMRdb make_model it replaces). Halo cars from
# mainstream makes. Labelled by variant like the exotics ("audi_r8 v10 spyder"), since one family
# label across coupe/spyder/GT galleries gives overlapping generations.
HALO_MODELS = [
    ("PORSCHE", r"^PORSCHE 918", "porsche", None),
    ("PORSCHE", r"^PORSCHE Carrera GT", "porsche", None),
    ("FORD", r"^FORD GT$", "ford", None),
    ("LEXUS", r"^LEXUS LFA", "lexus", None),
    ("MERCEDESBENZ", r"^MERCEDES BENZ SLR", "mercedes benz", "mercedes benz_slr"),
    ("MERCEDESBENZ", r"^MERCEDES BENZ SLS", "mercedes benz", None),
    ("ACURA", r"^ACURA NSX", "acura", "acura_nsx"),
    ("HONDA", r"^HONDA NSX", "honda", None),
    ("AUDI", r"^AUDI R8", "audi", "audi_r8"),
    ("BMW", r"^BMW i8", "bmw", None),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cm-root", type=Path, default=Path.home() / "Downloads" / "car_models_3887")
    p.add_argument("--max-year", type=int, default=2016,
                   help="only take Car Models 3887 generations (gallery start year) that started by this year")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def clean(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def generation(dir_path, from_year, to_year):
    """'.../2012Present' -> '2012-present', '.../20082011' -> '2008-2011'; else the CSV years."""
    leaf = dir_path.rstrip("/").split("/")[-1]
    if m := re.fullmatch(r"(\d{4})(\d{4})", leaf):
        return f"{m[1]}-{m[2]}"
    if m := re.fullmatch(r"(\d{4})Present", leaf, re.I):
        return f"{m[1]}-present"
    return f"{from_year}-{to_year}"


def index_images(cm_root):
    """basename -> (split, path) for every extracted Car Models 3887 image."""
    index = {}
    for split in ("train", "test"):
        for dirpath, _, files in os.walk(cm_root / split):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    index[f] = (split, os.path.join(dirpath, f))
    return index


def car_models_rows(args):
    index = index_images(args.cm_root)
    print(f"Car Models 3887: {len(index):,} extracted images found")
    rows, missing = [], 0
    with open(args.cm_root / "data_full.csv", encoding="utf-8", errors="ignore") as f:
        for r in csv.DictReader(f):
            brand = r["brand"].upper()
            if brand in EXOTIC_BRANDS:
                make = EXOTIC_BRANDS[brand]
            else:
                make = next((mk for b, pat, mk, _ in HALO_MODELS if brand == b and re.search(pat, r["model"])), None)
                if make is None:
                    continue
            model = clean(re.sub(rf"^{re.escape(make)}\s+", "", r["model"], flags=re.I))
            mm = f"{make}_{model}"
            gen = generation(r["dir_path"], r["from_year"], r["to_year"])
            if not gen[:4].isdigit() or int(gen[:4]) > args.max_year:
                continue
            for name in ast.literal_eval(r["image_file_names"]):
                if name not in index:
                    missing += 1
                    continue
                split, path = index[name]
                rows.append(dict(path=path, source="car_models_3887", split=split, make=make, model=model,
                                 generation=gen, make_model=mm, label=f"{mm}_{gen}"))
    print(f"Car Models 3887: {len(rows):,} exotic/halo images kept, {missing:,} listed in CSV but not extracted")

    # Hold out val from train, per label, so every label keeps at least one train image.
    rng = random.Random(args.seed)
    by_label = defaultdict(list)
    for r in rows:
        if r["split"] == "train":
            by_label[r["label"]].append(r)
    for group in by_label.values():
        rng.shuffle(group)
        for r in group[:int(len(group) * args.val_frac)]:
            r["split"] = "val"
    return rows


def vmmrdb_rows(drop_make_models):
    labels, paths = load_labels()
    split = np.load(split_path("make_model", 10, 42))
    label_to_bucket = json.load(open(BUCKETS_PATH))["label_to_bucket"]
    rows = []
    for name in ("train", "val", "test"):
        for i in split[name]:
            mm = make_model(labels[i])
            if mm in drop_make_models:
                continue
            bucket = label_to_bucket[labels[i]]
            make, model = mm.split("_", 1)
            rows.append(dict(path=paths[i], source="vmmrdb", split=name, make=make, model=model,
                             generation=bucket.rsplit("_", 1)[1], make_model=mm, label=bucket))
    return rows


def main():
    args = parse_args()
    cm = car_models_rows(args)
    cm_makes = {r["make"] for r in cm if r["make"] in EXOTIC_BRANDS.values()}
    replaced = {rep for *_, rep in HALO_MODELS if rep}
    labels, _ = load_labels()
    drop = {make_model(l) for l in set(labels) if make_model(l) in replaced or make_model(l).split("_")[0] in cm_makes}
    vm = vmmrdb_rows(drop)
    print(f"VMMRdb: {len(vm):,} images kept, dropped make_models now covered by Car Models 3887: {sorted(drop)}")

    rows = vm + cm
    with open(MANIFEST_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    splits = Counter((r["source"], r["split"]) for r in rows)
    print(f"\nWrote {MANIFEST_PATH}: {len(rows):,} images, {len({r['label'] for r in rows}):,} labels, "
          f"{len({r['make_model'] for r in rows}):,} make_models")
    for k in sorted(splits):
        print(f"  {k[0]:<16} {k[1]:<5} {splits[k]:>7,}")
    no_test = sorted({r["label"] for r in cm} - {r["label"] for r in cm if r["split"] == "test"})
    if no_test:
        print(f"\nWarning: {len(no_test)} Car Models 3887 labels have no test images: {no_test[:10]}")


if __name__ == "__main__":
    main()
