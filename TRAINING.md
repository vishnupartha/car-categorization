# Fine-tuning brief: DINOv3 car identification

A handoff spec for setting up and running the fine-tuning on a rented GPU. The data prep is done
locally (`prep_manifest.py` → `prep_crop.py` → `prep_pack.py`). This document covers everything after that.

## Goal

Turn a car photo into an embedding where photos of the same **make + model + generation** sit close
together. The app identifies a car by nearest-neighbour search over per-class reference embeddings
(Supabase pgvector). This setup also lets the app:
- add a new car later by embedding about 30 photos, with no retraining
- answer "unknown" when the best match is below a similarity cutoff
- report make → model → generation, so a low-confidence result can still be "Ferrari, probably a 458"

## Where we are

- Backbone: **DINOv3 ViT-L/16** (`dinov3_vitl16_pretrain_lvd1689m`, 303M params, 1024-d CLS output), **frozen**.
- A linear or MLP probe on frozen embeddings (VMMRdb only, uncropped) gets **91.9% make+model** and
  **89.1% make+model+generation** top-1 (see README.md).
- 59% of errors are "right make, wrong model". Some of those are label overlap in VMMRdb
  (`honda_civic` vs `honda_civic_coupe`, Silverado vs Sierra), which no model can fix.
  See `analysis/errors_bucket_mlp.csv`.

## Inputs (copy these to the GPU box)

| What | Where locally | Notes |
|---|---|---|
| Shards | `~/Downloads/car_shards/` | `{train,val,test}-NNNNNN.tar`, about 12 GB total |
| Class index | `car_shards/classes.json` | `label`, `make_model`, `make` → index, plus `label_parents` |
| Packed manifest | `car_shards/manifest.csv` | one row per image, with source and detector box |
| DINOv3 weights | `~/Downloads/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth` | gated by Meta, so bring this copy |
| DINOv3 code | `git clone https://github.com/facebookresearch/dinov3` | load with `torch.hub.load(repo, "dinov3_vitl16", source="local", weights=...)` |

**Shard format (WebDataset).** Each sample is `<key>.jpg` (car crop, long side ≤512 px) and `<key>.json`:
```json
{"label": "pagani_huayra_2012-present", "make_model": "pagani_huayra", "make": "pagani",
 "generation": "2012-present", "source": "car_models_3887",
 "label_idx": 1234, "make_model_idx": 567, "make_idx": 60, "orig_path": "..."}
```
- **Sources.**
  - `vmmrdb`: about 280k used-car listing photos, 834 make_models, about 2.1k labels.
  - `car_models_3887`: about 6–7k press/studio photos of exotic and halo cars, including Pagani,
    Koenigsegg, Bugatti, McLaren, Ferrari, Lamborghini and Porsche 918.
- **Splits.**
  - VMMRdb keeps the same train/val/test split as the README results (seed 42).
  - Car Models 3887 uses its own train/test, with 10% of train held out as val.

The exact counts are printed by `prep_pack.py`.

## Hardware and cost

- 1× **A100 80GB** or **H100** (RunPod, Lambda or Vast). Use bf16. Apple MPS is far too slow for this.
- At least 50 GB of persistent disk for shards, weights and checkpoints.
- **Before any full run:** time 200 steps, then report img/s and the projected cost per epoch and per
  run. Shut the instance down whenever nothing is running.

## Model

```
image ─► detector crop (already done in shards) ─► resize 256×256 ─► ImageNet normalize
      ─► DINOv3 ViT-L/16 (CLS token, 1024-d)
      ─► head: Linear(1024→512) ─► BatchNorm1d ─► L2-normalize      ← the embedding the app stores
      ─► sub-center ArcFace over `label` (training only)
```

- **Unfreezing (run E1).** Train the last 4 of 24 transformer blocks plus the final norm; freeze the rest.
  Run the frozen part under `torch.no_grad()` to save memory.
- **Target.** `label_idx` (make_model + generation). `make_model` and `make` come from `label_parents` at
  eval time, so no extra heads are needed.
- **Loss.** Sub-center ArcFace with k=3, s=30, m=0.30. Sub-centers tolerate noisy labels and
  multi-modal classes such as front vs rear views. Optional: an auxiliary CE on `make_idx` (weight 0.1)
  if make accuracy drops.
- **Class balance.** The largest class has thousands of images and an exotic has about 30. Sample with
  probability ∝ `n_class^0.5` per image (square-root sampling), or use a per-batch class-balanced sampler.
  Either way, report results for both sources separately.

### Augmentation

This bridges studio press photos and phone photos.

| Transform | Setting |
|---|---|
| RandomResizedCrop to 256 | scale 0.5–1.0, ratio 3/4–4/3 |
| Horizontal flip | p=0.5 |
| ColorJitter | 0.3 / 0.3 / 0.3 / 0.05 |
| RandomGrayscale | p=0.05 |
| GaussianBlur | p=0.2 |
| JPEG re-encode, quality 30–95 | p=0.5 |
| RandomPerspective | distortion 0.2, p=0.3 |
| Downscale ×0.25–0.5 then upscale | p=0.2 |

Eval uses only resize to 256×256 plus normalize, the same as `embed.py`.

### Optimisation (starting point)

| Setting | Value |
|---|---|
| Optimizer | AdamW, weight decay 0.05 |
| LR, unfrozen backbone blocks | 2e-5 |
| LR, head + ArcFace | 1e-3 |
| Schedule | 1 epoch linear warmup, then cosine |
| Epochs | 10–15 (an epoch = one pass over train) |
| Batch size | 256 (use gradient accumulation if needed) |
| Precision | bf16 autocast, grad clip 1.0 |
| Drop path, unfrozen blocks | 0.1 |
| Checkpoints | every epoch and resumable; keep best by val label top-1 |

Log to Weights & Biases.

## Evaluation

The evaluation must match how the app serves results:

1. Embed all train images with the fine-tuned model. The **prototype** for each label is the mean of
   its L2-normalized train embeddings, re-normalized.
2. Classify val/test images by cosine similarity to the prototypes.
3. Report:
   - top-1 / top-5 on `label`
   - accuracy on `make_model` and `make` (derived through `label_parents`)
   - all of the above **separately for `vmmrdb` and `car_models_3887`**, plus the overall figure

Also report a **linear probe** on the fine-tuned 512-d embeddings, for comparison with the README numbers.

A **real-world test set** is coming: `realworld/<label>/*.jpg`, phone photos to be added later. The eval
script should accept any folder in that layout, run the same detector crop as `prep_crop.py`
(torchvision `fasterrcnn_mobilenet_v3_large_320_fpn`, COCO car/truck, 10% padding), and report the
same metrics. Once it exists, this is the number that decides which model ships.

## Experiments, in order

| ID | What | Why |
|---|---|---|
| **E0** | Frozen backbone. Prototype eval and a linear probe on the new shards. | Baseline on the new data (crops + exotics). Everything else is compared to this. Cheap: one embedding pass. |
| **E1** | Last 4 blocks + head, ArcFace, settings above | Main candidate |
| **E2** | Last 8 blocks, layer-wise LR decay 0.8 | Check whether more unfreezing helps |
| **E3** | Optional: LoRA rank 16 on all blocks' qkv/proj, rest frozen | Cheaper alternative if E2 overfits |
| **E4** | Open set: retrain the best config with ~10% of make_models held out entirely (seeded list, include ~15 exotic ones) | Set the "unknown" threshold. Report AUROC (known vs unknown, using max cosine similarity) and the threshold that keeps 95% of known-class images |

Stop and report after E0 and E1 before spending on E2–E4.

## Deliverables

Put the code in this repo under `train/` (dataset, model, train, eval, export scripts), following the
style of `probe.py`. Each finished run produces:

- `checkpoint.pt`: the trainable parameters (unfrozen blocks, head, ArcFace) plus the config used.
  The frozen weights come from the original `.pth`.
- `prototypes.npy` (float16, `[n_labels, 512]`) and `prototype_labels.json`. Built from train + val for
  the final model.
- `classes.json`, copied from the shards.
- `metrics.json`, the W&B run link, and a short results table in the style of README.md.
- **ONNX export** of backbone + head (input 1×3×256×256 → 512-d normalized embedding), with a check
  that it matches PyTorch within cosine 0.999. Also export the detector to ONNX for the FastAPI server.

## Gotchas

- **VMMRdb label overlap caps accuracy.** Don't tune against errors like Civic vs Civic Coupe or
  F-250 vs F-350.
- **Year ranges.** Some Car Models 3887 ranges are wrong (LaFerrari, Enzo, F40, F50 are all
  "1984-2015"). Treat them as IDs, not facts.
- **Near-duplicates.** About 20% of VMMRdb test images have a near-copy in train (same listing).
  Report test accuracy with and without them, as the README does (cosine ≥ 0.95 to any train image).
- **Licensing.**
  - VMMRdb and Car Models 3887 (autoevolution press photos) are fine for research. Check before shipping
    a commercial product.
  - DINOv3 weights are under Meta's DINOv3 license.
