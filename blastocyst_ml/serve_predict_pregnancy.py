#!/usr/bin/env python3
"""
HTTP API for pregnancy-outcome (fertile/non-fertile) prediction from a
blastocyst image, optionally combined with clinical tabular features.

  cd blastocyst_ml && source .venv/bin/activate  # or your venv
  pip install -r requirements.txt
  export EMBRYO_PREGNANCY_CKPT=./pregnancy_resnet18.pt   # optional, this is the default

  uvicorn serve_predict_pregnancy:app --host 127.0.0.1 --port 8001

Frontend: set VITE_PREGNANCY_API_URL=http://127.0.0.1:8001

Tabular features are optional per request: pass a JSON object as the
`tabular` form field (any subset of the 36 clinical columns -- see
dataset_pregnancy.TAB_COLS for the exact names/spelling) and whichever
columns are omitted fall back to the training-set mean for that column
(same graceful-degradation approach the original live-birth model used for
missing EXP/ICM/TE/day/age). This means the endpoint works with an image
alone -- accuracy is naturally best when the real clinical values are sent.

NOTE: unlike serve_predict_gardner.py, this runs on its own port (8001) as a
separate process/service -- it's a different model with a different input
shape (image + tabular), not a drop-in replacement for the Gardner grading
API on port 8000. Run both if you want both sets of predictions.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from gradcam import overlay_to_png_bytes  # shared helper, model-agnostic
from gradcam_pregnancy import PregnancyGradCAM, build_tab_vector, load_model

CKPT_PATH = os.environ.get(
    "EMBRYO_PREGNANCY_CKPT", str(Path(__file__).resolve().parent / "pregnancy_resnet18.pt")
)

app = FastAPI(title="Embryo pregnancy-outcome predict", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime: dict[str, Any] = {}
LABELS = {0: "non_pregnant", 1: "pregnant"}


@app.on_event("startup")
def _startup() -> None:
    _runtime.clear()
    if not os.path.isfile(CKPT_PATH):
        _runtime["load_error"] = f"Checkpoint not found: {CKPT_PATH}"
        print(f"[serve_predict_pregnancy] {_runtime['load_error']}; /predict will be unavailable.")
        return
    try:
        model, image_size, preprocess, tab_cols, tab_mean, tab_std = load_model(CKPT_PATH)
    except Exception as e:
        _runtime["load_error"] = f"Failed to load checkpoint: {e}"
        print(f"[serve_predict_pregnancy] {_runtime['load_error']}")
        return

    print(f"[serve_predict_pregnancy] Loaded {CKPT_PATH} (backbone={model.backbone_name}, image_size={image_size})")
    gradcam = PregnancyGradCAM(model)
    _runtime.update(
        {
            "model": model,
            "image_size": image_size,
            "preprocess": preprocess,
            "tab_cols": tab_cols,
            "tab_mean": tab_mean,
            "tab_std": tab_std,
            "gradcam": gradcam,
            "backbone": model.backbone_name,
        }
    )


def _parse_tabular(tabular_json: Optional[str]) -> dict:
    if not tabular_json:
        return {}
    try:
        obj = json.loads(tabular_json)
        if not isinstance(obj, dict):
            raise ValueError("`tabular` must be a JSON object")
        return obj
    except Exception as e:
        raise HTTPException(400, f"Invalid `tabular` JSON: {e}") from e


@app.get("/health")
def health() -> dict[str, str]:
    if "model" not in _runtime:
        return {"status": "degraded", "detail": _runtime.get("load_error", "no model")}
    return {"status": "ok", "backbone": _runtime.get("backbone", "unknown")}


@app.get("/tabular_schema")
def tabular_schema() -> dict[str, Any]:
    """Column names the `tabular` JSON field accepts, so a form can be built against them."""
    if "model" not in _runtime:
        raise HTTPException(503, _runtime.get("load_error", "Model not loaded."))
    return {"columns": _runtime["tab_cols"]}


@app.post("/predict")
def predict(file: UploadFile = File(...), tabular: Optional[str] = Form(None)) -> dict[str, Any]:
    if "model" not in _runtime:
        raise HTTPException(
            503, _runtime.get("load_error", "Model not loaded. Set EMBRYO_PREGNANCY_CKPT and restart.")
        )
    model = _runtime["model"]
    image_size: int = _runtime["image_size"]
    preprocess = _runtime["preprocess"]
    tab_cols = _runtime["tab_cols"]
    tab_mean = _runtime["tab_mean"]
    tab_std = _runtime["tab_std"]

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = preprocess(raw, image_size)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    values = _parse_tabular(tabular)
    tab = build_tab_vector(tab_cols, tab_mean, tab_std, values)
    used_defaults = [c for c in tab_cols if c not in values]

    with torch.no_grad():
        logits = model(im, tab)
        probs = torch.softmax(logits, dim=1).squeeze(0)
    predicted_class = int(probs.argmax().item())

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": [float(x) for x in probs.tolist()],
        "model_type": "pregnancy_outcome_binary",
        "backbone": _runtime.get("backbone", "unknown"),
        "tabular_fields_defaulted": used_defaults,
        "note": "Research / demo only. Predicts pregnancy outcome class, not a clinical diagnosis.",
    }


@app.post("/predict_heatmap")
def predict_heatmap(
    file: UploadFile = File(...),
    tabular: Optional[str] = Form(None),
    alpha: float = Query(0.45, ge=0.0, le=1.0, description="Heatmap overlay opacity"),
) -> dict[str, Any]:
    """Same as /predict, plus a Grad-CAM heatmap (base64 PNG) showing which region of the
    image most drove the pregnant/non-pregnant prediction."""
    if "model" not in _runtime:
        raise HTTPException(
            503, _runtime.get("load_error", "Model not loaded. Set EMBRYO_PREGNANCY_CKPT and restart.")
        )
    image_size: int = _runtime["image_size"]
    preprocess = _runtime["preprocess"]
    tab_cols = _runtime["tab_cols"]
    tab_mean = _runtime["tab_mean"]
    tab_std = _runtime["tab_std"]
    gradcam: PregnancyGradCAM = _runtime["gradcam"]

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = preprocess(raw, image_size)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    values = _parse_tabular(tabular)
    tab = build_tab_vector(tab_cols, tab_mean, tab_std, values)
    used_defaults = [c for c in tab_cols if c not in values]

    # Min-max rescaled to [0,1] before overlay: this tensor is ImageNet-normalized
    # (can be negative), unlike a raw [0,1] tensor, so a naive *255 would clip/garble it.
    display = im.squeeze(0).permute(1, 2, 0).numpy()
    display = display - display.min()
    if display.max() > 1e-8:
        display = display / display.max()
    original_rgb = (display * 255).astype(np.uint8)

    cam, predicted_class, probs = gradcam(im, tab)
    png_bytes = overlay_to_png_bytes(original_rgb, cam, alpha=alpha)

    return {
        "predicted_class": predicted_class,
        "predicted_label": LABELS[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": [float(x) for x in probs.tolist()],
        "heatmap_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        "model_type": "pregnancy_outcome_binary_heatmap",
        "backbone": _runtime.get("backbone", "unknown"),
        "tabular_fields_defaulted": used_defaults,
        "note": "Research / demo only. Predicts pregnancy outcome class, not a clinical diagnosis.",
    }
