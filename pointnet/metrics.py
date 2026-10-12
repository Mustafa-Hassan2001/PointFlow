"""Confusion-matrix based metrics: overall accuracy, mean class accuracy, IoU."""
from __future__ import annotations

import numpy as np


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.matrix.fill(0)

    def update(self, pred, target) -> None:
        """Accumulate predictions. ``pred``/``target`` are integer arrays of any (equal) shape."""
        pred = np.asarray(pred).reshape(-1).astype(np.int64)
        target = np.asarray(target).reshape(-1).astype(np.int64)
        valid = (target >= 0) & (target < self.num_classes)
        idx = target[valid] * self.num_classes + pred[valid]
        self.matrix += np.bincount(idx, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    # rows = ground truth, cols = prediction
    @property
    def overall_accuracy(self) -> float:
        total = self.matrix.sum()
        return float(np.trace(self.matrix) / total) if total else 0.0

    @property
    def class_accuracy(self) -> np.ndarray:
        gt = self.matrix.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gt > 0, np.diag(self.matrix) / np.maximum(gt, 1), np.nan)

    @property
    def mean_accuracy(self) -> float:
        acc = self.class_accuracy
        return float(np.nanmean(acc)) if np.any(~np.isnan(acc)) else 0.0

    @property
    def iou(self) -> np.ndarray:
        tp = np.diag(self.matrix)
        union = self.matrix.sum(axis=0) + self.matrix.sum(axis=1) - tp
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, tp / np.maximum(union, 1), np.nan)

    @property
    def mean_iou(self) -> float:
        iou = self.iou
        return float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0

    def summary(self, class_names=None) -> str:
        names = class_names or [str(i) for i in range(self.num_classes)]
        acc, iou = self.class_accuracy, self.iou
        lines = [f"{'class':<12}{'acc':>8}{'IoU':>8}{'#gt':>12}"]
        for i, name in enumerate(names):
            lines.append(f"{name:<12}{acc[i]:>8.4f}{iou[i]:>8.4f}{self.matrix[i].sum():>12d}")
        lines.append(f"OA={self.overall_accuracy:.4f}  mAcc={self.mean_accuracy:.4f}  mIoU={self.mean_iou:.4f}")
        return "\n".join(lines)
