"""Shared paths and label helpers.

VMMRdb folder names look like "honda_civic_2003" or "mercedes benz_e430_1999":
make and model are separated by "_", and the year is always last.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = ROOT / "splits"
CHECKPOINTS_DIR = ROOT / "checkpoints"
ANALYSIS_DIR = ROOT / "analysis"

METADATA_PATH = DATA_DIR / "dinov3_vmmrdb_full.npz"               # embeddings (raw), labels, paths
EMBEDDINGS_PATH = DATA_DIR / "vmmrdb_embeddings_normalized.npy"   # L2-normalized, same row order
BUCKETS_PATH = ROOT / "year_buckets.json"


def make_model(label):
    """'honda_civic_2003' -> 'honda_civic'"""
    return label.rsplit("_", 1)[0]


def make_only(label):
    """'honda_civic_2003' -> 'honda'"""
    return label.split("_")[0]


def load_labels():
    meta = np.load(METADATA_PATH, allow_pickle=True)
    return meta["labels"].astype(str), meta["paths"].astype(str)


def load_embeddings():
    return np.load(EMBEDDINGS_PATH)


def split_path(target, min_images, seed):
    return SPLITS_DIR / f"{target}_min{min_images}_seed{seed}.npz"
