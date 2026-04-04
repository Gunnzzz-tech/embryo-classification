#!/usr/bin/env python3
"""
HTTP API for embryo live-birth prediction (image ± tabular, same as training).

  cd blastocyst_ml && source .venv/bin/activate  # or your venv
  pip install -r requirements.txt
  export EMBRYO_LB_CKPT=./embryo_lb_model.pt   # optional

Older checkpoints may omit tab_mean/tab_std. Then either re-train with train_live_birth.py, or set one of:
  EMBRYO_LB_TAB_STATS_JSON=/path/to/stats.json   # {"tab_mean":[...],"tab_std":[...]} matching ckpt tab order
  EMBRYO_LB_ANNOTATIONS_CSV=/path/to/Clincial_annotations*.csv
  BLASTOCYST_DATA=/path/to/Blastocyst_Dataset     # uses Clincial_annotations_cleaned.csv or Clincial_annotations.csv

  uvicorn serve_predict:app --host 127.0.0.1 --port 8000

Frontend: set VITE_ML_API_URL=http://127.0.0.1:8000
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model import EmbryoLBModel

CKPT_PATH = os.environ.get("EMBRYO_LB_CKPT", str(Path(__file__).resolve().parent / "embryo_lb_model.pt"))

app = FastAPI(title="Embryo live-birth predict", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime: dict[str, Any] = {}


def _load_ckpt(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_num_tabular(ckpt: dict[str, Any]) -> int:
    if "num_tabular" in ckpt:
        return int(ckpt["num_tabular"])
    st = ckpt.get("model") or {}
    w = st.get("fc_tab.weight")
    if w is not None:
        return int(w.shape[1])
    tab_cols = ckpt.get("tab_cols") or []
    return len(tab_cols)


def _tab_stats_from_json(path: str, num_tabular: int) -> tuple[list[float], list[float]]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    mean = list(obj.get("tab_mean") or [])
    std = list(obj.get("tab_std") or [])
    if len(mean) != num_tabular or len(std) != num_tabular:
        raise ValueError(
            f"{path}: expected tab_mean/tab_std length {num_tabular}, got {len(mean)}/{len(std)}"
        )
    return mean, std


def _tab_stats_from_annotations_csv(
    csv_path: str,
    tab_cols: list[str],
    data_root: Optional[str],
    image_subdir: str,
) -> tuple[list[float], list[float]]:
    from dataset import load_annotations

    df = load_annotations(csv_path)
    if data_root and "Image" in df.columns:
        img_dir = os.path.join(data_root, image_subdir)
        if os.path.isdir(img_dir):
            n0 = len(df)
            df = df[df["Image"].map(lambda n: os.path.isfile(os.path.join(img_dir, str(n))))].copy()
            if len(df) == 0:
                print(
                    f"[serve_predict] No rows with images under {img_dir}; "
                    "using all annotation rows for mean/std (may differ from training split)."
                )
                df = load_annotations(csv_path)
            elif len(df) < n0:
                print(f"[serve_predict] tab stats: {len(df)}/{n0} rows with images in {img_dir}")

    missing = [c for c in tab_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns {missing} (checkpoint expects {tab_cols})")

    work = df[tab_cols].copy()
    if "Age" in work.columns:
        work["Age"] = work["Age"].fillna(work["Age"].median())
    work = work.fillna(work.median())
    tab_mean = work.mean().astype(np.float32)
    tab_std = work.std().replace(0, 1).astype(np.float32)
    return [float(tab_mean[c]) for c in tab_cols], [float(tab_std[c]) for c in tab_cols]


def _resolve_tab_mean_std(
    ckpt: dict[str, Any], num_tabular: int, tab_cols: list[str]
) -> tuple[Optional[list[float]], Optional[list[float]], Optional[str]]:
    """Return (mean, std, source_label) or (None, None, None) if unresolved."""
    tab_mean: list[float] = list(ckpt.get("tab_mean") or [])
    tab_std: list[float] = list(ckpt.get("tab_std") or [])
    if num_tabular == 0:
        return tab_mean, tab_std, "checkpoint" if tab_mean else None
    if len(tab_mean) == num_tabular and len(tab_std) == num_tabular:
        return tab_mean, tab_std, "checkpoint"

    jpath = os.environ.get("EMBRYO_LB_TAB_STATS_JSON", "").strip()
    if jpath and os.path.isfile(jpath):
        m, s = _tab_stats_from_json(jpath, num_tabular)
        return m, s, jpath

    csv_path = os.environ.get("EMBRYO_LB_ANNOTATIONS_CSV", "").strip()
    data_root: Optional[str] = None
    image_subdir = os.environ.get("EMBRYO_LB_IMAGE_DIR", "Images")
    if not csv_path:
        dr = os.path.expanduser(os.environ.get("BLASTOCYST_DATA", "").strip())
        if dr and os.path.isdir(dr):
            data_root = dr
            for name in ("Clincial_annotations_cleaned.csv", "Clincial_annotations.csv"):
                cand = os.path.join(dr, name)
                if os.path.isfile(cand):
                    csv_path = cand
                    break
    else:
        csv_path = os.path.expanduser(csv_path)
        if os.path.isfile(csv_path):
            parent = os.path.dirname(csv_path)
            if os.path.isdir(os.path.join(parent, image_subdir)):
                data_root = parent

    if csv_path and os.path.isfile(csv_path):
        if data_root is None:
            data_root = os.path.dirname(csv_path)
        m, s = _tab_stats_from_annotations_csv(csv_path, tab_cols, data_root, image_subdir)
        return m, s, csv_path

    return None, None, None


@app.on_event("startup")
def _startup() -> None:
    _runtime.clear()
    try:
        ckpt = _load_ckpt(CKPT_PATH)
    except FileNotFoundError as e:
        print(f"[serve_predict] {e}; /predict will be unavailable until EMBRYO_LB_CKPT is valid.")
        _runtime["load_error"] = str(e)
        return
    except Exception as e:
        print(f"[serve_predict] Failed to load checkpoint: {e}")
        _runtime["load_error"] = str(e)
        return

    state = ckpt["model"]
    tab_cols: list[str] = list(ckpt.get("tab_cols") or [])
    image_size = int(ckpt.get("image_size") or 224)
    num_tabular = _infer_num_tabular(ckpt)
    if num_tabular > 0 and len(tab_cols) != num_tabular:
        print(
            "[serve_predict] tab_cols length does not match num_tabular; "
            "fix checkpoint or re-train so tab_cols matches the tabular branch."
        )
        _runtime["load_error"] = "checkpoint tab_cols mismatch"
        return
    try:
        tab_mean_l, tab_std_l, stats_src = _resolve_tab_mean_std(ckpt, num_tabular, tab_cols)
    except Exception as e:
        print(f"[serve_predict] Could not resolve tab_mean/tab_std: {e}")
        _runtime["load_error"] = f"missing tab_mean/tab_std in checkpoint ({e})"
        return
    if num_tabular > 0 and (not tab_mean_l or not tab_std_l or len(tab_mean_l) != num_tabular):
        print(
            "[serve_predict] Checkpoint missing tab_mean/tab_std. "
            "Set EMBRYO_LB_ANNOTATIONS_CSV or BLASTOCYST_DATA, or EMBRYO_LB_TAB_STATS_JSON, "
            "or re-train with train_live_birth.py."
        )
        _runtime["load_error"] = "missing tab_mean/tab_std in checkpoint"
        return
    tab_mean: list[float] = list(tab_mean_l or [])
    tab_std: list[float] = list(tab_std_l or [])
    if num_tabular > 0 and stats_src and stats_src != "checkpoint":
        print(f"[serve_predict] Using tab_mean/tab_std from {stats_src}")
    model = EmbryoLBModel(image_size=image_size, num_tabular=num_tabular, num_classes=2)
    model.load_state_dict(state)
    model.eval()
    _runtime.update(
        {
            "model": model,
            "tab_cols": tab_cols,
            "tab_mean": torch.tensor(tab_mean, dtype=torch.float32) if tab_mean else None,
            "tab_std": torch.tensor(tab_std, dtype=torch.float32) if tab_std else None,
            "image_size": image_size,
            "num_tabular": num_tabular,
        }
    )


def _preprocess_image(raw: bytes, size: int) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [size, size])
    return t.unsqueeze(0)


def _build_tab_vector(
    tab_cols: list[str],
    tab_mean: torch.Tensor,
    tab_std: torch.Tensor,
    exp: Optional[float],
    icm: Optional[float],
    te: Optional[float],
    d: Optional[float],
    age: Optional[float],
) -> torch.Tensor:
    raw_map = {
        "EXP_silver": exp,
        "ICM_silver": icm,
        "TE_silver": te,
        "d": d,
        "Age": age,
    }
    vals = []
    for i, col in enumerate(tab_cols):
        v = raw_map.get(col)
        if v is None:
            v = float(tab_mean[i].item())
        vals.append(float(v))
    x = torch.tensor(vals, dtype=torch.float32)
    x = (x - tab_mean) / tab_std
    return x.unsqueeze(0)


@app.get("/health")
def health() -> dict[str, str]:
    if "model" not in _runtime:
        return {"status": "degraded", "detail": _runtime.get("load_error", "no model")}
    return {"status": "ok"}


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    exp: Optional[float] = Form(None),
    icm: Optional[float] = Form(None),
    te: Optional[float] = Form(None),
    d: Optional[float] = Form(None),
    age: Optional[float] = Form(None),
) -> dict[str, Any]:
    if "model" not in _runtime:
        raise HTTPException(
            503,
            _runtime.get("load_error", "Model not loaded. Set EMBRYO_LB_CKPT and restart."),
        )
    model: EmbryoLBModel = _runtime["model"]
    tab_cols: list[str] = _runtime["tab_cols"]
    tab_mean = _runtime["tab_mean"]
    tab_std = _runtime["tab_std"]
    image_size: int = _runtime["image_size"]
    num_tabular: int = _runtime["num_tabular"]

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = _preprocess_image(raw, image_size)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    tab = None
    if num_tabular > 0 and tab_mean is not None and tab_std is not None:
        tab = _build_tab_vector(tab_cols, tab_mean, tab_std, exp, icm, te, d, age)

    with torch.no_grad():
        logits = model(im, tab)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    p0 = float(probs[0].item())
    p1 = float(probs[1].item())
    pred = int(probs.argmax().item())
    return {
        "predicted_lb_class": pred,
        "probability_lb_1": p1,
        "probabilities": [p0, p1],
        "model_type": "live_birth_binary",
        "note": "Research / demo only. Predicts live-birth class, not Gardner grade or developmental stage.",
    }
