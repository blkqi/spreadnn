from __future__ import annotations

from importlib.resources import files as _pkg_files
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

_STRIP_H = 256
_STRIP_W = 64

__all__ = ("SpreadModel",)


class SpreadModel:
    """MobileNetV3-small binary classifier for manga spread detection.

    Usage::

        model = SpreadModel()
        prob = model.score_pair(right_page, left_page)
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._net: nn.Module | None = None
        self._model_path = model_path

    def score_pair(self, img_e: np.ndarray, img_o: np.ndarray) -> float:
        """Return spread probability [0, 1] for a pair of decoded pages.

        Parameters
        ----------
        img_e : numpy.ndarray
            The RIGHT page (its left edge is the inner gutter).
            BGR ordering (OpenCV convention).
        img_o : numpy.ndarray
            The LEFT page (its right edge is the inner gutter).

        Returns
        -------
        float
            Sigmoid probability that this pair is a true two-page spread.
        """
        h_e, w_e = img_e.shape[:2]
        h_o, w_o = img_o.shape[:2]
        target_h = max(h_e, h_o)

        if h_e != target_h:
            img_e = cv2.resize(img_e, (w_e, target_h), interpolation=cv2.INTER_LINEAR)
        if h_o != target_h:
            img_o = cv2.resize(img_o, (w_o, target_h), interpolation=cv2.INTER_LINEAR)

        half_w = _STRIP_W // 2
        strip_e = cv2.resize(img_e[:, :half_w], (half_w, _STRIP_H))
        strip_o = cv2.resize(img_o[:, w_o - half_w:], (half_w, _STRIP_H))

        canvas = cv2.cvtColor(np.hstack((strip_o, strip_e)), cv2.COLOR_BGR2RGB)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = (tensor - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...].astype(np.float32)

        if self._net is None:
            self._load_model()

        with torch.no_grad():
            outputs = self._net(torch.from_numpy(tensor))
            logit = float(outputs.numpy().flatten()[0])
            prob = 1.0 / (1.0 + np.exp(-logit))

        return prob

    def _load_model(self) -> None:
        model_path = self._resolve_model_path()
        net = models.mobilenet_v3_small()
        num_features = net.classifier[0].in_features
        net.classifier = nn.Sequential(nn.Linear(num_features, 1))
        net.load_state_dict(torch.load(str(model_path), map_location=torch.device("cpu")))
        net.eval()
        self._net = net

    def _resolve_model_path(self) -> Path:
        mp = self._model_path
        if mp is None:
            return _pkg_files("spreadnn") / "models" / "manga-digital.pth"
        p = Path(mp)
        if p.suffix == ".pth":
            return p
        return _pkg_files("spreadnn") / "models" / f"{mp}.pth"
