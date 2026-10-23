"""
Grad-CAM for the pregnancy-outcome (fertile/non-fertile) model
(model_pregnancy_transfer.PregnancyTransferModel), hooked on the image
backbone's last conv block -- same technique as gradcam.py uses for the
Gardner grading model, adapted for a single binary head and an image+tabular
input instead of image-only with three heads.
"""
from __future__ import annotations

import io
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from model_pregnancy_transfer import PregnancyTransferModel


def load_model(ckpt_path: str):
    """
    Returns (model, image_size, preprocess_fn, tab_cols, tab_mean, tab_std).
    preprocess_fn(raw_bytes) -> 1xCxHxW tensor, normalized exactly as this
    checkpoint was trained (ImageNet mean/std).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_size = int(ckpt.get("image_size") or 224)
    tab_cols = list(ckpt["tab_cols"])
    tab_mean = torch.tensor(ckpt["tab_mean"], dtype=torch.float32)
    tab_std = torch.tensor(ckpt["tab_std"], dtype=torch.float32)
    mean = ckpt.get("image_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("image_std", [0.229, 0.224, 0.225])

    model = PregnancyTransferModel(ckpt["backbone"], num_tabular=ckpt["num_tabular"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    def preprocess(raw: bytes, size: int = image_size) -> torch.Tensor:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [size, size])
        t = TF.normalize(t, mean=mean, std=std)
        return t.unsqueeze(0)

    return model, image_size, preprocess, tab_cols, tab_mean, tab_std


def build_tab_vector(tab_cols: list, tab_mean: torch.Tensor, tab_std: torch.Tensor, values: dict) -> torch.Tensor:
    """values: dict of {col_name: float}, any subset. Missing columns fall back to the
    training-set mean (normalizes to 0 for that feature), exactly like the old LB service did
    for exp/icm/te/d/age -- so this works even when the caller only has the image."""
    raw = []
    for i, col in enumerate(tab_cols):
        v = values.get(col)
        raw.append(float(v) if v is not None else float(tab_mean[i].item()))
    x = torch.tensor(raw, dtype=torch.float32)
    x = (x - tab_mean) / tab_std
    return x.unsqueeze(0)


class PregnancyGradCAM:
    """Grad-CAM hooked on the model's last spatial conv block (pre-global-pool)."""

    def __init__(self, model: PregnancyTransferModel):
        self.model = model
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._target_layer = model.gradcam_target_module()
        self._fwd_handle = self._target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self._target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image: torch.Tensor, tabular: torch.Tensor, class_idx: Optional[int] = None):
        """
        Returns (cam: HxW float32 in [0,1], predicted_class: int (0=non-pregnant,
        1=pregnant), probs: 1D tensor of length 2).
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image, tabular)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        target_class = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy().astype("float32")
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, target_class, probs.detach()
