"""Train / evaluate loops shared by the classification and segmentation scripts.

Batches are ``(points (B, N, C), labels)`` where labels are ``(B,)`` for
classification and ``(B, N)`` for segmentation. Logits are ``(B, K)`` or
``(B, K, N)`` respectively, so ``argmax(dim=1)`` works for both.
"""
from __future__ import annotations

import torch
from tqdm import tqdm

from .metrics import ConfusionMatrix


def train_one_epoch(model, loader, criterion, optimizer, device, num_classes: int, desc: str = "train",
                    grad_clip: float | None = None):
    model.train()
    cm = ConfusionMatrix(num_classes)
    total_loss, n = 0.0, 0
    for points, labels in tqdm(loader, desc=desc, leave=False):
        points = points.to(device, non_blocking=True).float().transpose(1, 2)
        labels = labels.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        logits, trans_feat = model(points)
        loss = criterion(logits, labels, trans_feat)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * points.shape[0]
        n += points.shape[0]
        cm.update(logits.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
    return total_loss / max(n, 1), cm


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes: int, desc: str = "eval"):
    model.eval()
    cm = ConfusionMatrix(num_classes)
    total_loss, n = 0.0, 0
    for points, labels in tqdm(loader, desc=desc, leave=False):
        points = points.to(device, non_blocking=True).float().transpose(1, 2)
        labels = labels.to(device, non_blocking=True).long()
        logits, trans_feat = model(points)
        total_loss += criterion(logits, labels, trans_feat).item() * points.shape[0]
        n += points.shape[0]
        cm.update(logits.argmax(dim=1).cpu().numpy(), labels.cpu().numpy())
    return total_loss / max(n, 1), cm
