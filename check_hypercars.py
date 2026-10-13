"""Check whether VMMRdb contains hypercars / exotic supercars.

Matches whole exotic makes (pagani, koenigsegg, ferrari, ...) plus halo models
from mainstream makes (porsche 918, ford gt, lexus lfa, ...).

    python check_hypercars.py
"""

from collections import Counter

from common import load_labels, make_model

EXOTIC_MAKES = [
    "pagani", "koenigsegg", "ferrari", "lamborghini", "bugatti", "mclaren",
    "rimac", "ssc", "hennessey", "saleen", "spyker", "noble", "zenvo",
    "aston martin", "maserati", "lotus", "w motors", "czinger", "gumpert",
]

# (make, substring of model) for hypercars sold under mainstream badges
HALO_MODELS = [
    ("porsche", "918"), ("porsche", "carrera gt"), ("porsche", "959"),
    ("ford", "gt"), ("mercedes benz", "slr"), ("mercedes benz", "sls"),
    ("mercedes benz", "amg one"), ("lexus", "lfa"), ("acura", "nsx"),
    ("honda", "nsx"), ("dodge", "viper"), ("nissan", "gt-r"), ("nissan", "gtr"),
    ("jaguar", "xj220"), ("aston martin", "valkyrie"),
]


def main():
    labels, _ = load_labels()
    image_counts = Counter(labels)
    makes = {l.split("_")[0] for l in image_counts}

    print(f"{len(image_counts)} classes, {len(makes)} makes, {len(labels)} images\n")

    print("Exotic makes:")
    for make in EXOTIC_MAKES:
        classes = sorted(l for l in image_counts if l.split("_")[0] == make)
        if not classes:
            print(f"  {make:<14} -- not present")
            continue
        n = sum(image_counts[c] for c in classes)
        models = Counter(make_model(c) for c in classes)
        print(f"  {make:<14} {len(classes)} classes, {n} images")
        for mm in sorted(models):
            print(f"      {mm}")

    print("\nHalo models from other makes:")
    found = False
    for make, needle in HALO_MODELS:
        classes = sorted(
            l for l in image_counts
            if l.split("_")[0] == make and make_model(l).split("_", 1)[-1].startswith(needle)
        )
        if classes:
            found = True
            n = sum(image_counts[c] for c in classes)
            print(f"  {make} *{needle}*: {n} images  ({', '.join(classes)})")
    if not found:
        print("  none")


if __name__ == "__main__":
    main()
