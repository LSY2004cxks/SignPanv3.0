from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class ResizeTransform:
    src_size: Tuple[int, int]  # (h, w)
    dst_size: Tuple[int, int]  # (h, w)

    @property
    def sx(self) -> float:
        return self.dst_size[1] / float(self.src_size[1])

    @property
    def sy(self) -> float:
        return self.dst_size[0] / float(self.src_size[0])

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.dst_size[1], self.dst_size[0]), interpolation=cv2.INTER_LINEAR)

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        out = pts.copy()
        out[..., 0] = out[..., 0] * self.sx
        out[..., 1] = out[..., 1] * self.sy
        return out

    def invert_points(self, pts: np.ndarray) -> np.ndarray:
        out = pts.copy()
        out[..., 0] = out[..., 0] / self.sx
        out[..., 1] = out[..., 1] / self.sy
        return out
