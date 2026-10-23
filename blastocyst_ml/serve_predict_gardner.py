#!/usr/bin/env python3
"""
HTTP API for Gardner-grade prediction (EXP/ICM/TE) from a blastocyst image.

  cd blastocyst_ml && source .venv/bin/activate  # or your venv
  pip install -r requirements.txt
  export EMBRYO_GARDNER_CKPT=./embryo_gardner_model.pt   # optional, this is the default
                                                          # (point this at gardner_resnet18.pt /
                                                          # gardner_efficientnet_b0.pt to serve
                                                          # the retrained transfer-learning model)

  uvicorn serve_predict_gardner:app --host 127.0.0.1 --port 8000

Frontend: set VITE_ML_API_URL=http://127.0.0.1:8000

Model loading is checkpoint-driven (see gradcam.load_model): whichever
checkpoint EMBRYO_GARDNER_CKPT points at -- the original from-scratch CNN, or
a retrained ResNet18/EfficientNet-B0 transfer-learning model -- is detected
automatically, and its own matching image preprocessing is used, so this file
does not need to change when you swap checkpoints.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from gradcam import GardnerGradCAM, load_model, overlay_to_png_bytes

CKPT_PATH = os.environ.get(
    "EMBRYO_GARDNER_CKPT", str(Path(__file__).resolve().parent / "embryo_gardner_model.pt")
)

app = FastAPI(title="Embryo Gardner-grade predict", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_runtime: dict[str, Any] = {}


@app.on_event("startup")
def _startup() -> None:
    _runtime.clear()
    if not os.path.isfile(CKPT_PATH):
        _runtime["load_error"] = f"Checkpoint not found: {CKPT_PATH}"
        print(f"[serve_predict_gardner] {_runtime['load_error']}; /predict will be unavailable.")
        return
    try:
        model, image_size, preprocess_fn = load_model(CKPT_PATH)
    except Exception as e:
        _runtime["load_error"] = f"Failed to load checkpoint: {e}"
        print(f"[serve_predict_gardner] {_runtime['load_error']}")
        return

    backbone = getattr(model, "backbone_name", "legacy_cnn")
    print(f"[serve_predict_gardner] Loaded {CKPT_PATH} (backbone={backbone}, image_size={image_size})")

    # One GardnerGradCAM per process, hooks stay registered on the model's last
    # conv block for the life of the app. Each request's forward+backward pass
    # overwrites the captured activations/gradients, so this is fine for the
    # single dev-server-process, one-request-at-a-time usage this API is built
    # for; it is NOT safe if you later run multiple uvicorn workers/threads
    # handling requests concurrently against the same model instance.
    gradcam = GardnerGradCAM(model)
    _runtime.update(
        {
            "model": model,
            "image_size": image_size,
            "preprocess": preprocess_fn,
            "gradcam": gradcam,
            "backbone": backbone,
        }
    )


@app.get("/health")
def health() -> dict[str, str]:
    if "model" not in _runtime:
        return {"status": "degraded", "detail": _runtime.get("load_error", "no model")}
    return {"status": "ok", "backbone": _runtime.get("backbone", "unknown")}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if "model" not in _runtime:
        raise HTTPException(
            503, _runtime.get("load_error", "Model not loaded. Set EMBRYO_GARDNER_CKPT and restart.")
        )
    model = _runtime["model"]
    image_size: int = _runtime["image_size"]
    preprocess = _runtime["preprocess"]

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = preprocess(raw, image_size)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    with torch.no_grad():
        logits_exp, logits_icm, logits_te = model(im)
        probs_exp = torch.softmax(logits_exp, dim=1).squeeze(0)
        probs_icm = torch.softmax(logits_icm, dim=1).squeeze(0)
        probs_te = torch.softmax(logits_te, dim=1).squeeze(0)

    return {
        "predicted_exp": int(probs_exp.argmax().item()),
        "predicted_icm": int(probs_icm.argmax().item()),
        "predicted_te": int(probs_te.argmax().item()),
        "probabilities_exp": [float(x) for x in probs_exp.tolist()],
        "probabilities_icm": [float(x) for x in probs_icm.tolist()],
        "probabilities_te": [float(x) for x in probs_te.tolist()],
        "model_type": "gardner_grading",
        "backbone": _runtime.get("backbone", "unknown"),
        "note": "Research / demo only. Predicts Gardner EXP/ICM/TE grades, not live-birth outcome.",
    }


@app.post("/predict_heatmap")
def predict_heatmap(
    file: UploadFile = File(...),
    heads: str = Query("exp,icm,te", description="Comma-separated subset of exp,icm,te"),
    alpha: float = Query(0.45, ge=0.0, le=1.0, description="Heatmap overlay opacity"),
) -> dict[str, Any]:
    """
    Same predictions as /predict, plus a Grad-CAM heatmap (base64 PNG, jet
    colormap over the input image) per requested head, showing which region
    of the image most drove that head's predicted grade.
    """
    if "model" not in _runtime:
        raise HTTPException(
            503, _runtime.get("load_error", "Model not loaded. Set EMBRYO_GARDNER_CKPT and restart.")
        )
    image_size: int = _runtime["image_size"]
    preprocess = _runtime["preprocess"]
    gradcam: GardnerGradCAM = _runtime["gradcam"]

    requested_heads = [h.strip() for h in heads.split(",") if h.strip()]
    for h in requested_heads:
        if h not in ("exp", "icm", "te"):
            raise HTTPException(400, f"Unknown head '{h}'; must be one of exp, icm, te")

    raw = file.file.read()
    if not raw:
        raise HTTPException(400, "Empty file")
    try:
        im = preprocess(raw, image_size)
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}") from e

    # Heatmap is overlaid onto the model's own (possibly normalized) input
    # tensor rendered back to a displayable 0-255 image, at image_size x
    # image_size, so it lines up pixel-for-pixel with what the model saw.
    display = im.squeeze(0).permute(1, 2, 0).numpy()
    display = display - display.min()
    if display.max() > 1e-8:
        display = display / display.max()
    original_rgb = (display * 255).astype(np.uint8)

    result: dict[str, Any] = {"model_type": "gardner_grading_heatmap", "backbone": _runtime.get("backbone", "unknown")}
    for head in requested_heads:
        cam, predicted_class, probs = gradcam(im, head)
        png_bytes = overlay_to_png_bytes(original_rgb, cam, alpha=alpha)
        result[head] = {
            "predicted_class": predicted_class,
            "confidence": float(probs[predicted_class]),
            "probabilities": [float(x) for x in probs.tolist()],
            "heatmap_png_base64": base64.b64encode(png_bytes).decode("ascii"),
        }
    return result
