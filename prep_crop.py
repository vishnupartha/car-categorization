"""Prep step 2: crop every manifest image to the car, so training matches what the app will see.

App photos won't be framed like listing photos, so the app will run the same detector and crop
first. Training on crops keeps the two consistent.

Detector: torchvision Faster R-CNN (MobileNetV3 320px, COCO). BSD-licensed, unlike YOLO/ultralytics
(AGPL). Picks the highest score x area box among cars/trucks, pads it by 10%, and saves the crop
with its long side at most --max-side px.

No car found:
    vmmrdb           keep the full image (listing photos are almost always the car)
    car_models_3887  drop it (press galleries include interior, engine and wheel shots)

Resumable: images already in data/crops.csv are skipped.

Output: <out>/<source>/<parent folder>/<file>.jpg and data/crops.csv
        (path, crop_path, detected, score, x0, y0, x1, y1)

Usage:
    python3.11 prep_crop.py                       # all of data/manifest.csv
    python3.11 prep_crop.py --limit 200           # quick check
"""

import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
                                          fasterrcnn_mobilenet_v3_large_320_fpn)
from torchvision.transforms.functional import pil_to_tensor

from common import DATA_DIR
from prep_manifest import MANIFEST_PATH

CROPS_CSV = DATA_DIR / "crops.csv"
CAR_CLASSES = {3, 8}  # COCO: car, truck (pickups and SUVs often come out as truck)
FIELDS = ["path", "crop_path", "detected", "score", "x0", "y0", "x1", "y1"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path.home() / "Downloads" / "car_crops")
    p.add_argument("--max-side", type=int, default=512)
    p.add_argument("--pad", type=float, default=0.10)
    p.add_argument("--min-score", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


class Images(Dataset):
    """Decodes in the DataLoader workers; yields uint8 CHW tensors (or None if unreadable)."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        try:
            return i, pil_to_tensor(Image.open(self.rows[i]["path"]).convert("RGB"))
        except Exception:
            return i, None


def collate(batch):
    return batch


def crop_path(out, row):
    p = Path(row["path"])
    return out / row["source"] / p.parent.name / (p.stem + ".jpg")


def pick_box(det, w, h, min_score):
    best, best_val = None, 0.0
    for box, label, score in zip(det["boxes"].tolist(), det["labels"].tolist(), det["scores"].tolist()):
        if label not in CAR_CLASSES or score < min_score:
            continue
        x0, y0, x1, y1 = box
        val = score * (x1 - x0) * (y1 - y0) / (w * h)
        if val > best_val:
            best, best_val = (box, score), val
    return best


def save_crop(img, box, out, max_side):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = Image.fromarray(img[:, y0:y1, x0:x1].permute(1, 2, 0).numpy())
    crop.thumbnail((max_side, max_side))
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, quality=90)


def main():
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    # The 320px variant is ~2.5x faster than the full one on MPS and plenty for one large object.
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
                                                  box_score_thresh=args.min_score).eval().to(device)

    rows = list(csv.DictReader(open(MANIFEST_PATH)))[:args.limit]
    done = {}
    if CROPS_CSV.exists():
        done = {r["path"]: r for r in csv.DictReader(open(CROPS_CSV))}
    todo = [r for r in rows if r["path"] not in done]
    print(f"{len(rows):,} manifest images, {len(rows) - len(todo):,} already done, {len(todo):,} to crop on {device}")

    results = dict(done)
    empty = dict(crop_path="", detected=0, score=0, x0=0, y0=0, x1=0, y1=0)
    loader = DataLoader(Images(todo), batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    t0, n = time.time(), 0
    with torch.no_grad(), ThreadPoolExecutor(4) as pool:
        for batch in loader:
            n += len(batch)
            ok = [(i, img) for i, img in batch if img is not None]
            for i, img in batch:
                if img is None:
                    results[todo[i]["path"]] = dict(path=todo[i]["path"], **empty)
            dets = model([img.to(device).float() / 255 for _, img in ok]) if ok else []
            for (i, img), det in zip(ok, dets):
                row = todo[i]
                h, w = img.shape[1:]
                best = pick_box({k: v.cpu() for k, v in det.items()}, w, h, args.min_score)
                if best is None and row["source"] != "vmmrdb":
                    results[row["path"]] = dict(path=row["path"], **empty)
                    continue
                if best is None:
                    box, score = (0, 0, w, h), 0.0
                else:
                    (x0, y0, x1, y1), score = best
                    px, py = (x1 - x0) * args.pad, (y1 - y0) * args.pad
                    box = (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))
                out = crop_path(args.out, row)
                pool.submit(save_crop, img, box, out, args.max_side)
                results[row["path"]] = dict(path=row["path"], crop_path=str(out), detected=int(best is not None),
                                            score=round(score, 3), **dict(zip(["x0", "y0", "x1", "y1"],
                                                                              (round(v) for v in box))))
            if n % (args.batch_size * 100) < args.batch_size:
                rate = n / (time.time() - t0)
                print(f"  {n:,}/{len(todo):,}  {rate:.0f} img/s  ~{(len(todo) - n) / rate / 60:.0f} min left", flush=True)
                write(results)
    write(results)

    by_source = {}
    for m in rows:
        r = results[m["path"]]
        s = by_source.setdefault(m["source"], [0, 0, 0])
        s[0] += 1
        s[1] += int(r["detected"])
        s[2] += int(not r["crop_path"])
    for src, (total, det, dropped) in by_source.items():
        print(f"{src:<16} {total:,} images, car detected in {det / total:.1%}, dropped {dropped:,}")


def write(results):
    tmp = CROPS_CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(results.values())
    tmp.replace(CROPS_CSV)


if __name__ == "__main__":
    main()
