# CleavageEmbryo v1.1 — clean & train

Dataset: **CleavageEmbryov1.1** (COCO JSON + `images/train`, `images/val`).

- **Annotations:** instance masks for **cell** (category 1) and **fragment** (category 2).
- **Cleaning:** checks files exist, drops images with no annotations, optional blur filter.
- **Label:** `label_high_fragmentation` = 1 if **fragment_ratio** ≥ **median fragment_ratio on the train split** (so ~half of train images are “high fragmentation” for a learnable binary task).

This is **not** live birth; it learns **morphology-related fragmentation** from the official cell/fragment counts.

## Setup

Use the same venv as `blastocyst_ml`:

```bash
cd ../blastocyst_ml && source .venv/bin/activate && cd ../cleavage_ml
```

## 1. Clean

```bash
python3 clean_cleavage_data.py --data "$HOME/Downloads/CleavageEmbryov1.1"
```

Optional: `--blur-threshold 50` (requires opencv), `--blur-dropped dropped.txt`.

Output: `$HOME/Downloads/CleavageEmbryov1.1/cleavage_manifest_cleaned.csv`

## 2. Train

```bash
python3 train_cleavage.py --manifest "$HOME/Downloads/CleavageEmbryov1.1/cleavage_manifest_cleaned.csv" --epochs 30 --save cleavage_fragmentation_model.pt
```

Model is saved under `cleavage_ml/cleavage_fragmentation_model.pt` (image-only CNN).
