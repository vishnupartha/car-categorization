"""Step 0: compute DINOv3 embeddings for every VMMRdb image.

Runs the pretrained DINOv3 ViT-L/16 backbone (frozen, no training) on each image and saves
its 1024-dim CLS embedding. Takes ~2h40m for all 281,752 images on an Apple M-series GPU.

Writes:
    data/dinov3_vmmrdb_full.npz            embeddings (raw), labels, paths
    data/vmmrdb_embeddings_normalized.npy  L2-normalized embeddings (used by everything else)

Requires a local clone of https://github.com/facebookresearch/dinov3 and the
dinov3_vitl16_pretrain_lvd1689m weights (request access via the DINOv3 repo).

Usage:
    python3.11 embed.py --dataset ~/Downloads/VMMRdb \
        --repo ./dinov3 --weights ~/Downloads/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
    python3.11 embed.py ... --limit 1000      # quick test; writes separate *_1000 files
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from tqdm import tqdm

from common import DATA_DIR, EMBEDDINGS_PATH, METADATA_PATH


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path.home() / "Downloads" / "VMMRdb",
                   help="VMMRdb root: one folder per make_model_year")
    p.add_argument("--repo", type=Path, default=Path("dinov3"), help="local clone of facebookresearch/dinov3")
    p.add_argument("--weights", type=Path,
                   default=Path.home() / "Downloads" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, help="embed a random sample of this many images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="replace existing embedding files")
    return p.parse_args()


def main():
    args = parse_args()
    # A --limit sample goes to its own files so it never clobbers the full run
    meta_path, emb_path = METADATA_PATH, EMBEDDINGS_PATH
    if args.limit:
        meta_path = DATA_DIR / f"dinov3_vmmrdb_{args.limit}.npz"
        emb_path = DATA_DIR / f"vmmrdb_embeddings_normalized_{args.limit}.npy"
    if not args.overwrite and (meta_path.exists() or emb_path.exists()):
        raise SystemExit(f"{meta_path.name} already exists; pass --overwrite to replace it")

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(args.dataset.rglob("*.jpg"))
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:args.limit]]
    labels = [p.parent.name for p in paths]
    print(f"{len(paths):,} images from {args.dataset}, device {device}")

    model = torch.hub.load(str(args.repo), "dinov3_vitl16", source="local", weights=str(args.weights))
    model = model.to(device).eval()

    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    embeddings = []
    for start in tqdm(range(0, len(paths), args.batch_size)):
        batch = [transform(Image.open(p).convert("RGB")) for p in paths[start:start + args.batch_size]]
        with torch.inference_mode():
            embeddings.append(model(torch.stack(batch).to(device)).cpu().numpy())
    embeddings = np.concatenate(embeddings).astype(np.float32)

    DATA_DIR.mkdir(exist_ok=True)
    np.savez(meta_path, embeddings=embeddings, labels=np.array(labels), paths=np.array([str(p) for p in paths]))
    np.save(emb_path, embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True))
    print(f"Saved {meta_path.relative_to(DATA_DIR.parent)} and {emb_path.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
