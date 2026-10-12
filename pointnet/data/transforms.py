"""Point cloud augmentations operating on ``(N, C)`` float arrays.

Geometric transforms only touch the XYZ channels (first three columns);
other channels such as colour are left as they are.
"""
from __future__ import annotations

import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            pts = t(pts)
        return pts


class RandomRotateZ:
    """Random rotation around the vertical (gravity) axis."""

    def __init__(self, max_angle: float = 2 * np.pi):
        self.max_angle = max_angle

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        theta = np.random.uniform(0, self.max_angle)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=pts.dtype)
        pts = pts.copy()
        pts[:, :3] = pts[:, :3] @ rot.T
        return pts


class RandomScale:
    def __init__(self, low: float = 0.8, high: float = 1.25):
        self.low, self.high = low, high

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        pts[:, :3] *= np.random.uniform(self.low, self.high)
        return pts


class RandomJitter:
    """Clipped Gaussian noise on XYZ (sigma=0.01, clip=0.05 as in the paper)."""

    def __init__(self, sigma: float = 0.01, clip: float = 0.05):
        self.sigma, self.clip = sigma, clip

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        pts = pts.copy()
        noise = np.clip(self.sigma * np.random.randn(len(pts), 3), -self.clip, self.clip)
        pts[:, :3] += noise.astype(pts.dtype)
        return pts


class RandomShuffle:
    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return pts[np.random.permutation(len(pts))]


def cls_train_transform() -> Compose:
    return Compose([RandomRotateZ(), RandomScale(), RandomJitter(), RandomShuffle()])


def seg_train_transform() -> Compose:
    # Shuffling is unnecessary for segmentation and would desync labels.
    return Compose([RandomJitter(sigma=0.005, clip=0.02)])
