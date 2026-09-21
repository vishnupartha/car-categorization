"""Group model years into buckets where the car looks the same (roughly: generations/facelifts).

For each make+model, walk its years in order and compare each year's embeddings to the
previous year's. A drop in similarity means the car was redesigned, so a new bucket starts.

Similarity is sample-size corrected: the average cross-year pairwise similarity divided by the
average within-year pairwise similarity. Two years of an unchanged car score ~1.0 no matter how
many photos each has; a redesign scores well below (e.g. Camry 2001->2002 is 0.47).

Only train images (from the make_model split made by probe.py) are used to build buckets,
so the test set stays untouched.

Output: year_buckets.json mapping every label (e.g. "honda_civic_2003") to a bucket
(e.g. "honda_civic_2001-2005"). Then run:  python3.11 probe.py --target bucket

Usage:
    python3.11 year_buckets.py
    python3.11 year_buckets.py --threshold 0.93     # fewer, wider buckets
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from common import BUCKETS_PATH, ROOT, load_embeddings, load_labels, split_path
SHOW = ["honda_civic", "toyota_camry", "honda_accord", "ford_explorer", "ford_mustang",
        "chevrolet_silverado", "nissan_altima"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.95,
                   help="start a new bucket when corrected similarity to the previous year is below this")
    p.add_argument("--min-year-images", type=int, default=5,
                   help="years with fewer train images can't start a bucket; they join the nearest one")
    p.add_argument("--min-images", type=int, default=10, help="which make_model split to use (see probe.py)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    path = split_path("make_model", args.min_images, args.seed)
    if not path.exists():
        raise SystemExit(f"Need {path.relative_to(ROOT)}: run `probe.py --skip-knn --skip-probe` first")
    labels, _ = load_labels()
    embeddings = load_embeddings()
    train = np.load(path, allow_pickle=True)["train"]

    # Per (make_model, year): sum of train embeddings and count
    sums, counts = {}, defaultdict(int)
    for i in train:
        key = tuple(labels[i].rsplit("_", 1))
        sums[key] = sums.get(key, 0) + embeddings[i]
        counts[key] += 1

    def within(key):
        n = counts[key]
        return (sums[key] @ sums[key] - n) / (n * (n - 1))

    def similarity(a, b):
        between = sums[a] @ sums[b] / (counts[a] * counts[b])
        return between / np.sqrt(within(a) * within(b))

    # Every year that appears anywhere in the dataset, so val/test-only years get a bucket too
    years_by_model = defaultdict(set)
    for label in np.unique(labels):
        mm, year = label.rsplit("_", 1)
        years_by_model[mm].add(year)

    mapping, summary = {}, {}
    for mm, years in sorted(years_by_model.items()):
        years = sorted(years, key=int)
        strong = [y for y in years if counts[(mm, y)] >= args.min_year_images]

        if not strong:
            groups = [years]
        else:
            # Split strong years where similarity to the previous strong year drops
            groups = [[strong[0]]]
            for prev, year in zip(strong, strong[1:]):
                if similarity((mm, prev), (mm, year)) < args.threshold:
                    groups.append([])
                groups[-1].append(year)

            # Attach weak years to the bucket of the nearest strong year (ties -> earlier)
            owner = {y: g for g, ys in enumerate(groups) for y in ys}
            for year in years:
                if year not in owner:
                    nearest = min(strong, key=lambda s: (abs(int(s) - int(year)), int(s)))
                    groups[owner[nearest]].append(year)
            groups = [sorted(g, key=int) for g in groups]

        summary[mm] = []
        for g in groups:
            name = f"{mm}_{g[0]}-{g[-1]}" if len(g) > 1 else f"{mm}_{g[0]}"
            summary[mm].append(name)
            for year in g:
                mapping[f"{mm}_{year}"] = name

    with open(BUCKETS_PATH, "w") as f:
        json.dump({"threshold": args.threshold, "min_year_images": args.min_year_images,
                   "label_to_bucket": mapping, "buckets_by_model": summary}, f, indent=1)

    n_buckets = sum(len(v) for v in summary.values())
    print(f"{len(mapping):,} year-labels -> {n_buckets:,} buckets across {len(summary):,} models "
          f"(threshold {args.threshold})")
    for mm in SHOW:
        if mm in summary:
            print(f"  {mm:<22} " + ", ".join(b.rsplit("_", 1)[1] for b in summary[mm]))
    print(f"Saved to {BUCKETS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
