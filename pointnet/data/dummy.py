"""Generate a tiny synthetic dataset in the raw S3DIS folder layout.

Useful to smoke-test the whole pipeline (preprocess -> train -> test)
without downloading the ~4 GB S3DIS release.
"""
from __future__ import annotations

import os

import numpy as np

from .s3dis import CLASSES


def _box(rng, lo, hi, n):
    return rng.uniform(lo, hi, size=(n, 3))


def make_dummy_s3dis(root: str, areas=(1, 5), rooms_per_area: int = 2, points_per_object: int = 600,
                     seed: int = 0) -> str:
    """Write rooms with floor, ceiling, walls and a few furniture boxes.

    Returns ``root`` which mimics ``Stanford3dDataset_v1.2_Aligned_Version``.
    """
    rng = np.random.default_rng(seed)
    palette = rng.integers(0, 256, size=(len(CLASSES), 3))
    for area in areas:
        for r in range(rooms_per_area):
            ann_dir = os.path.join(root, f"Area_{area}", f"office_{r + 1}", "Annotations")
            os.makedirs(ann_dir, exist_ok=True)
            w, d, h = rng.uniform(3, 5), rng.uniform(3, 5), 3.0
            objects = {
                "floor_1": _box(rng, [0, 0, 0], [w, d, 0.02], points_per_object),
                "ceiling_1": _box(rng, [0, 0, h - 0.02], [w, d, h], points_per_object),
                "wall_1": _box(rng, [0, 0, 0], [0.02, d, h], points_per_object),
                "wall_2": _box(rng, [w - 0.02, 0, 0], [w, d, h], points_per_object),
            }
            for k, cls in enumerate(["table", "chair", "bookcase", "clutter"]):
                x, y = rng.uniform(0.5, w - 1.0), rng.uniform(0.5, d - 1.0)
                height = {"table": 0.75, "chair": 1.0, "bookcase": 2.0, "clutter": 0.3}[cls]
                objects[f"{cls}_{k + 1}"] = _box(rng, [x, y, 0.02], [x + 0.6, y + 0.6, height], points_per_object)
            for name, xyz in objects.items():
                label = CLASSES.index(name.split("_")[0])
                rgb = np.clip(palette[label] + rng.normal(0, 10, size=(len(xyz), 3)), 0, 255)
                np.savetxt(os.path.join(ann_dir, f"{name}.txt"), np.hstack([xyz, rgb]), fmt="%.4f")
    return root
