"""Steps 1 + 2: honest train/test evaluation and a classifier on frozen DINOv3 embeddings.

Uses the embeddings from embed.py (data/).

Step 1: stratified train/val/test split, then kNN retrieval of test images against the
        train gallery only (no test image can match itself or another test image).
Step 2: train a linear (or small MLP) classifier on the frozen train embeddings, evaluate it
        on the same test set, and write an error analysis to analysis/.

The split is saved to splits/ so later experiments reuse the exact same test set.

Usage:
    python3.11 probe.py                          # make+model, classes with >= 10 images
    python3.11 probe.py --target label           # make+model+year (much harder)
    python3.11 probe.py --target bucket --head mlp   # make+model+year bucket (run year_buckets.py first)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from common import (ANALYSIS_DIR, BUCKETS_PATH, CHECKPOINTS_DIR, ROOT, load_embeddings,
                    load_labels, make_model, make_only, split_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["make_model", "label", "bucket"], default="make_model",
                   help="make_model = e.g. 'honda_civic'; label = 'honda_civic_2002'; "
                        "bucket = 'honda_civic_2001-2005' (run year_buckets.py first)")
    p.add_argument("--min-images", type=int, default=10,
                   help="drop classes with fewer images than this")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1, help="fraction of train held out for model selection")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10, help="neighbors for kNN vote")
    p.add_argument("--dup-threshold", type=float, default=0.95,
                   help="cosine sim above which a test image counts as a near-duplicate of a train image")
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--top-confusions", type=int, default=25, help="how many confused pairs to list")
    p.add_argument("--skip-knn", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------- data + split

def load_data():
    labels, paths = load_labels()
    embeddings = load_embeddings()
    assert embeddings.shape[0] == len(labels), "embeddings and labels are out of sync"
    return embeddings, labels, paths


def make_split(labels, args):
    if args.target == "bucket":
        return bucket_split(labels, args)

    targets = np.array([make_model(l) for l in labels]) if args.target == "make_model" else labels

    classes, counts = np.unique(targets, return_counts=True)
    keep_classes = classes[counts >= args.min_images]
    keep = np.flatnonzero(np.isin(targets, keep_classes))

    class_to_idx = {c: i for i, c in enumerate(keep_classes)}
    y = np.array([class_to_idx[t] for t in targets[keep]])

    train_val, test, y_train_val, _ = train_test_split(
        keep, y, test_size=args.test_size, stratify=y, random_state=args.seed)
    train, val = train_test_split(
        train_val, test_size=args.val_size, stratify=y_train_val, random_state=args.seed)

    y_all = np.full(len(labels), -1)
    y_all[keep] = y

    print(f"Target: {args.target}, classes with >= {args.min_images} images: "
          f"{len(keep_classes):,} classes, {len(keep):,} images "
          f"({len(keep) / len(labels):.1%} of dataset)")
    print(f"Split: train {len(train):,} / val {len(val):,} / test {len(test):,}")

    path = split_path(args.target, args.min_images, args.seed)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, train=train, val=val, test=test, classes=keep_classes, y=y_all)
    print(f"Saved split to {path.relative_to(ROOT)}")

    return train, val, test, keep_classes, y_all


def bucket_split(labels, args):
    """Reuse the make_model split (the one year_buckets.py was built from) with bucket labels."""
    path = split_path("make_model", args.min_images, args.seed)
    if not path.exists() or not BUCKETS_PATH.exists():
        raise SystemExit(f"Need {path.name} and {BUCKETS_PATH.name}: run "
                         f"`probe.py --target make_model --min-images {args.min_images}` "
                         f"then `year_buckets.py` first")

    split = np.load(path, allow_pickle=True)
    train, val, test = split["train"], split["val"], split["test"]
    label_to_bucket = json.load(open(BUCKETS_PATH))["label_to_bucket"]

    keep = np.concatenate([train, val, test])
    targets = np.array([label_to_bucket[l] for l in labels[keep]])
    classes = np.unique(targets)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = np.full(len(labels), -1)
    y_all[keep] = [class_to_idx[t] for t in targets]

    years = [c.rsplit("_", 1)[1].split("-") for c in classes]   # "2001-2005" or "2012"
    spans = np.array([int(y[-1]) - int(y[0]) + 1 for y in years])
    print(f"Target: bucket, {len(classes):,} buckets over {len(keep):,} images "
          f"(median span {np.median(spans):.0f} years, mean {spans.mean():.1f})")
    print(f"Split (reused from {path.name}): "
          f"train {len(train):,} / val {len(val):,} / test {len(test):,}")
    return train, val, test, classes, y_all


# ---------------------------------------------------------------- step 1: kNN

def knn_eval(embeddings, labels, train, test, y_all, classes, args):
    print("\n=== Step 1: kNN retrieval (test queries vs. train gallery) ===")
    t0 = time.time()

    # Gallery = train + val; the probe needs val held out, kNN doesn't.
    gallery = np.setdiff1d(np.arange(len(labels)), test)
    gallery = gallery[y_all[gallery] >= 0]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings[gallery]))

    sims, nbrs = index.search(np.ascontiguousarray(embeddings[test]), args.k)
    nbrs = gallery[nbrs]  # back to dataset row indices
    print(f"Searched {len(test):,} queries against {len(gallery):,} images in {time.time() - t0:.0f}s")

    y_test = y_all[test]
    y_nbrs = y_all[nbrs]

    # Similarity-weighted vote over the k neighbors
    votes = np.zeros((len(test), len(classes)), dtype=np.float32)
    np.add.at(votes, (np.arange(len(test))[:, None], y_nbrs), sims)
    knn_pred = votes.argmax(axis=1)

    top1 = y_nbrs[:, 0] == y_test
    near_dup = sims[:, 0] >= args.dup_threshold

    makes = np.array([make_only(l) for l in labels])
    make_models = np.array([make_model(l) for l in labels])
    report = {
        "1-NN accuracy": top1.mean(),
        f"{args.k}-NN weighted vote accuracy": (knn_pred == y_test).mean(),
        "1-NN make accuracy": (makes[nbrs[:, 0]] == makes[test]).mean(),
        "1-NN make+model accuracy": (make_models[nbrs[:, 0]] == make_models[test]).mean(),
        "1-NN exact label (incl. year) accuracy": (labels[nbrs[:, 0]] == labels[test]).mean(),
    }
    for name, value in report.items():
        print(f"  {name:<42} {value:.2%}")

    print(f"\n  Near-duplicates (top-1 sim >= {args.dup_threshold}): "
          f"{near_dup.sum():,} of {len(test):,} test images ({near_dup.mean():.1%})")
    print(f"  1-NN accuracy on near-duplicates:          {top1[near_dup].mean():.2%}" if near_dup.any() else "")
    print(f"  1-NN accuracy excluding near-duplicates:   {top1[~near_dup].mean():.2%}")
    pct = np.percentile(sims[:, 0], [5, 25, 50, 75, 95])
    print("  Top-1 similarity percentiles (5/25/50/75/95): " + " / ".join(f"{p:.3f}" for p in pct))

    return near_dup


# ---------------------------------------------------------------- step 2: probe

class Probe(nn.Module):
    def __init__(self, dim, n_classes, head, mean, std):
        super().__init__()
        # Standardize features using train statistics; helps the linear probe converge.
        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))
        if head == "linear":
            self.net = nn.Linear(dim, n_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(dim, 2048), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(2048, n_classes),
            )

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


@torch.no_grad()
def predict(model, x, batch_size=8192):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def train_probe(embeddings, labels, train, val, test, y_all, classes, near_dup, args):
    print(f"\n=== Step 2: {args.head} probe on frozen embeddings ===")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    x_train = torch.from_numpy(embeddings[train]).to(device)
    x_val = torch.from_numpy(embeddings[val]).to(device)
    x_test = torch.from_numpy(embeddings[test]).to(device)
    y_train = torch.from_numpy(y_all[train]).to(device)
    y_val = torch.from_numpy(y_all[val]).to(device)
    y_test = torch.from_numpy(y_all[test]).to(device)

    mean = x_train.mean(0).cpu()
    std = x_train.std(0).cpu() + 1e-6
    model = Probe(embeddings.shape[1], len(classes), args.head, mean, std).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)

    best_val, best_state, best_epoch = -1.0, None, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train), device=device)
        total_loss = 0.0
        for i in range(0, len(train), args.batch_size):
            batch = perm[i:i + args.batch_size]
            loss = F.cross_entropy(model(x_train[batch]), y_train[batch],
                                   label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(batch)

        val_acc = (predict(model, x_val).argmax(1) == y_val).float().mean().item()
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:3d}  loss {total_loss / len(train):.4f}  val acc {val_acc:.2%}"
              f"  ({time.time() - t0:.0f}s)")

    model.load_state_dict(best_state)
    logits = predict(model, x_test)
    pred = logits.argmax(1)
    top1 = (pred == y_test).cpu().numpy()
    top5 = (logits.topk(5, dim=1).indices == y_test[:, None]).any(1).float().mean().item()

    class_makes = np.array([make_only(c) for c in classes])
    make_acc = (class_makes[pred.cpu().numpy()] == class_makes[y_all[test]]).mean()

    print(f"\n  Best epoch {best_epoch} (val acc {best_val:.2%})")
    print(f"  {'Test top-1 accuracy':<42} {top1.mean():.2%}")
    print(f"  {'Test top-5 accuracy':<42} {top5:.2%}")
    print(f"  {'Test make accuracy':<42} {make_acc:.2%}")
    if args.target != "make_model":
        class_mms = np.array([make_model(c) for c in classes])
        mm_acc = (class_mms[pred.cpu().numpy()] == class_mms[y_all[test]]).mean()
        print(f"  {'Test make+model accuracy':<42} {mm_acc:.2%}")
    if near_dup is not None:
        print(f"  {'Test top-1 excluding near-duplicates':<42} {top1[~near_dup].mean():.2%}")

    ckpt_path = CHECKPOINTS_DIR / f"probe_{args.head}_{args.target}_min{args.min_images}.pt"
    ckpt_path.parent.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(classes),
                "head": args.head, "args": vars(args)}, ckpt_path)
    print(f"  Saved probe to {ckpt_path.relative_to(ROOT)}")

    probs = logits.softmax(1).max(1).values.cpu().numpy()
    return pred.cpu().numpy(), probs


# ---------------------------------------------------------------- error analysis

def confusion_report(paths, test, y_all, pred, confidence, classes, args):
    print("\n=== Error analysis (test set) ===")
    true = y_all[test]
    wrong = pred != true

    class_mm = np.array([c if args.target == "make_model" else make_model(c) for c in classes])
    class_make = np.array([make_only(c) for c in classes])
    true_mm, pred_mm = class_mm[true], class_mm[pred]

    wrong_make = wrong & (class_make[true] != class_make[pred])
    wrong_model = wrong & ~wrong_make & (true_mm != pred_mm)
    wrong_years = wrong & (true_mm == pred_mm)
    n_wrong = wrong.sum()
    print(f"  {n_wrong:,} errors out of {len(test):,} test images:")
    print(f"    wrong make                    {wrong_make.sum():6,}  ({wrong_make.sum() / n_wrong:.0%} of errors)")
    print(f"    right make, wrong model       {wrong_model.sum():6,}  ({wrong_model.sum() / n_wrong:.0%})")
    if args.target != "make_model":
        print(f"    right model, wrong years      {wrong_years.sum():6,}  ({wrong_years.sum() / n_wrong:.0%})")

    # Most-confused make+model pairs, counting both directions together
    model_errors = wrong & (true_mm != pred_mm)
    pair_counts = {}
    for t, p in zip(true_mm[model_errors], pred_mm[model_errors]):
        a, b = sorted((t, p))
        fwd, back = pair_counts.get((a, b), (0, 0))
        pair_counts[(a, b)] = (fwd + 1, back) if t == a else (fwd, back + 1)
    ranked = sorted(pair_counts.items(), key=lambda kv: -sum(kv[1]))

    test_count = dict(zip(*np.unique(true_mm, return_counts=True)))
    top = ranked[:args.top_confusions]
    covered = sum(sum(c) for _, c in top)
    print(f"\n  Top {len(top)} confused make+model pairs "
          f"(cover {covered / model_errors.sum():.0%} of {model_errors.sum():,} wrong-model errors):")
    print(f"    {'A':<28} {'B':<28} {'A->B':>5} {'B->A':>5}   % of A's / B's test images")
    for (a, b), (ab, ba) in top:
        print(f"    {a:<28} {b:<28} {ab:5d} {ba:5d}   {ab / test_count.get(a, 1):5.1%} / {ba / test_count.get(b, 1):5.1%}")

    out_dir = ANALYSIS_DIR
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.target}_{args.head}"

    csv_path = out_dir / f"errors_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "true", "predicted", "confidence", "error_type"])
        for i in np.flatnonzero(wrong):
            kind = "make" if wrong_make[i] else "model" if wrong_model[i] else "years"
            image = Path(paths[test[i]])
            writer.writerow([f"{image.parent.name}/{image.name}", classes[true[i]], classes[pred[i]],
                             f"{confidence[i]:.3f}", kind])
    print(f"\n  Saved all errors to {csv_path.relative_to(ROOT)}")

    save_gallery(paths[test], true_mm, [pair for pair, _ in top[:12]], out_dir / f"confused_pairs_{tag}.png")


def save_gallery(test_paths, true_mm, pairs, out_path, per_side=3):
    """One row per confused pair: a few test photos of A, then a few of B."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(pairs), 2 * per_side, figsize=(2.6 * 2 * per_side, 2.1 * len(pairs)))
    for row, (a, b) in enumerate(pairs):
        for side, mm in enumerate((a, b)):
            idx = np.flatnonzero(true_mm == mm)
            for j, i in enumerate(rng.choice(idx, size=min(per_side, len(idx)), replace=False)):
                ax = axes[row, side * per_side + j]
                ax.imshow(Image.open(test_paths[i]).convert("RGB"))
                ax.set_title(mm, fontsize=8, color="tab:blue" if side == 0 else "tab:red")
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"  Saved photos of the top {len(pairs)} confused pairs to {out_path.relative_to(ROOT)}")


def main():
    args = parse_args()
    embeddings, labels, paths = load_data()
    train, val, test, classes, y_all = make_split(labels, args)

    near_dup = None
    if not args.skip_knn:
        near_dup = knn_eval(embeddings, labels, train, test, y_all, classes, args)
    if not args.skip_probe:
        pred, confidence = train_probe(embeddings, labels, train, val, test, y_all, classes, near_dup, args)
        confusion_report(paths, test, y_all, pred, confidence, classes, args)


if __name__ == "__main__":
    main()
