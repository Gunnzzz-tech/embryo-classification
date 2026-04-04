# Embryo live birth prediction (image + CSV)

Train a model to predict **live birth (LB)** from blastocyst images and clinical CSV (EXP, ICM, TE, Age, etc.).

## Dataset

- **Blastocyst_Dataset** with:
  - `Images/` – embryo images (e.g. `0001_01.png`)
  - `Clincial_annotations.csv` – columns: Image, Fond, d, EXP_silver, ICM_silver, TE_silver, Age, …, **LB** (0/1)

## Setup

On **macOS with Homebrew Python** you’ll see `externally-managed-environment` if you use `pip` directly. Use a **virtual environment** instead:

```bash
cd blastocyst_ml
./setup_venv.sh
source .venv/bin/activate
```

Or manually:

```bash
cd blastocyst_ml
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then **always activate the venv** before running scripts: `source .venv/bin/activate`. After that, `python3` and `pip` use the venv.

## 1. Clean the data (run first)

```bash
source .venv/bin/activate   # if using venv
python3 clean_data.py --data "$HOME/Downloads/Blastocyst_Dataset"
```

This removes invalid LB, out-of-range Gardner scores (EXP/ICM/TE), invalid day, age, duplicates, rows whose image file is missing, and **blurry images** (and their CSV rows). Blur is detected via Laplacian variance (opencv); low variance = blurry. Install `opencv-python-headless` for blur check.

- `--blur-threshold 50` (default): drop images with Laplacian variance &lt; 50. Use 0 to disable. Typical range 50–200.
- `--blur-dropped blurry_dropped.txt`: save list of dropped image filenames to a file (optional; useful to delete those files from `Images/` later).
- `--skip-blur`: do not run blur check (e.g. if opencv not installed).

Output: `Clincial_annotations_cleaned.csv` in the dataset folder. CSV and image set stay in sync: when an image is dropped, its row is removed from the CSV.

## 2. Train

```bash
source .venv/bin/activate   # if using venv
# Use cleaned CSV (recommended)
python3 train_live_birth.py --data "$HOME/Downloads/Blastocyst_Dataset" --csv Clincial_annotations_cleaned.csv --epochs 30 --save embryo_lb_model.pt

# Or with default (original) CSV
python3 train_live_birth.py --data "$HOME/Downloads/Blastocyst_Dataset" --epochs 30 --save embryo_lb_model.pt
```

Options: `--no-tabular` (image only), `--val-frac`, `--lr`, `--size` (image size).

## Output

- `embryo_lb_model.pt` – saved model state and (if used) `tab_cols` for inference.
