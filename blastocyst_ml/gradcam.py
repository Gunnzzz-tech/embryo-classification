"""
Grad-CAM for the Gardner grading model (EXP / ICM / TE), supporting both:
  - the original from-scratch CNN (model_gardner.EmbryoGardnerModel), hooked on
    its last batch-norm (bn4), preprocessing = raw [0,1], no normalization.
  - the new transfer-learning models (model_gardner_transfer.GardnerTransferModel,
    backbone="resnet18" or "efficientnet_b0"), hooked on the backbone's last
    conv block, preprocessing = ImageNet-normalized.

Which one loads is auto-detected from the checkpoint: a "backbone" key means
the new transfer model; its absence means the legacy CNN. This keeps
serve_predict_gardner.py and visualize_heatmap.py working unchanged whichever
checkpoint EMBRYO_GARDNER_CKPT points at.
"""
from __future__ import annotations

import io
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

Head = Literal["exp", "icm", "te"]
HEAD_INDEX = {"exp": 0, "icm": 1, "te": 2}


def load_model(ckpt_path: str):
    """
    Load a Gardner checkpoint (legacy CNN or transfer-learning backbone).
    Returns (model, image_size, preprocess_fn) where preprocess_fn(raw_bytes) ->
    a 1xCxHxW tensor using exactly the normalization that checkpoint was trained with.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_size = int(ckpt.get("image_size") or 224)

    if "backbone" in ckpt:
        from model_gardner_transfer import GardnerTransferModel

        model = GardnerTransferModel(
            ckpt["backbone"],
            int(ckpt["num_exp"]),
            int(ckpt["num_icm"]),
            int(ckpt["num_te"]),
            pretrained=False,
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        mean = ckpt.get("image_mean", [0.485, 0.456, 0.406])
        std = ckpt.get("image_std", [0.229, 0.224, 0.225])

        def preprocess(raw: bytes, size: int = image_size) -> torch.Tensor:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            t = TF.resize(t, [size, size])
            t = TF.normalize(t, mean=mean, std=std)
            return t.unsqueeze(0)

        return model, image_size, preprocess

    # Legacy from-scratch CNN.
    from model_gardner import EmbryoGardnerModel

    model = EmbryoGardnerModel(int(ckpt["num_exp"]), int(ckpt["num_icm"]), int(ckpt["num_te"]))
    model.load_state_dict(ckpt["model"])
    model.eval()

    def preprocess_legacy(raw: bytes, size: int = image_size) -> torch.Tensor:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        t = TF.resize(t, [size, size])
        return t.unsqueeze(0)

    return model, image_size, preprocess_legacy


def preprocess_image(raw: bytes, size: int) -> torch.Tensor:
    """Kept for backwards compatibility (visualize_heatmap.py's old call sites
    that don't have a per-checkpoint preprocess_fn). Assumes the legacy,
    un-normalized preprocessing -- prefer the `preprocess_fn` returned by
    `load_model` for a checkpoint-correct result."""
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    t = TF.resize(t, [size, size])
    return t.unsqueeze(0)


class GardnerGradCAM:
    """
    Grad-CAM hooked on the model's last spatial conv block (right before
    global-average-pooling). Works with either the legacy CNN (hooks bn4) or a
    transfer-learning backbone (hooks model.gradcam_target_module()).
    """

    def __init__(self, model):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        if hasattr(model, "gradcam_target_module"):
            self._target_layer = model.gradcam_target_module()
        else:
            self._target_layer = model.bn4  # legacy EmbryoGardnerModel
        self._fwd_handle = self._target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self._target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self._activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    def remove_hooks(self):
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, image: torch.Tensor, head: Head, class_idx: int | None = None):
        """
        Run one forward pass and compute a Grad-CAM heatmap for the given
        head ('exp' | 'icm' | 'te'). If class_idx is None, uses the head's
        own predicted (argmax) class -- i.e. "why did the model give THIS
        score", which is what you almost always want for a single case.

        Returns (cam: HxW float32 in [0,1], predicted_class: int, probs: 1D tensor).
        """
        self.model.zero_grad(set_to_none=True)
        logits_exp, logits_icm, logits_te = self.model(image)
        logits = {"exp": logits_exp, "icm": logits_icm, "te": logits_te}[head]
        probs = torch.softmax(logits, dim=1).squeeze(0)

        target_class = int(logits.argmax(dim=1).item()) if class_idx is None else class_idx
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Grad-CAM: channel weights = global-average-pooled gradients.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy().astype("float32")
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam, target_class, probs.detach()

    def all_heads(self, image: torch.Tensor):
        """Convenience: Grad-CAM + prediction for exp, icm, and te in one call."""
        results = {}
        for head in ("exp", "icm", "te"):
            cam, cls, probs = self(image, head)
            results[head] = {"cam": cam, "predicted_class": cls, "probabilities": probs.tolist()}
        return results


def overlay_heatmap(original_rgb: "np.ndarray", cam: "np.ndarray", alpha: float = 0.45) -> "np.ndarray":
    """original_rgb: HxWx3 uint8 RGB (any resolution). cam: hxw float32 in [0,1]
    at the model's internal feature-map resolution. Returns an HxWx3 uint8 RGB
    image at original_rgb's resolution, with a jet-colormap heatmap blended in."""
    import cv2

    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)
    heatmap_u8 = np.uint8(255 * cam_resized)
    heatmap_color_bgr = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_color_rgb + (1 - alpha) * original_rgb).astype(np.uint8)
    return blended


def overlay_to_png_bytes(original_rgb: "np.ndarray", cam: "np.ndarray", alpha: float = 0.45) -> bytes:
    """Same as overlay_heatmap, but returns encoded PNG bytes ready to return
    from an API endpoint (e.g. base64-encoded in a JSON response)."""
    blended = overlay_heatmap(original_rgb, cam, alpha=alpha)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return buf.getvalue()
