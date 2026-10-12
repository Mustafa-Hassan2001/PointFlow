"""Training helpers: seeding, logging, checkpoints, BN momentum schedule, PLY export."""
from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own NumPy seed (random block sampling relies on it)."""
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def get_logger(log_dir: str, name: str = "pointnet") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"{name}:{log_dir}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(os.path.join(log_dir, "log.txt"))):
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


def bn_momentum_adjust(model: torch.nn.Module, momentum: float) -> None:
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.momentum = momentum


def bn_momentum_for_epoch(epoch: int, init: float = 0.1, decay: float = 0.5, step: int = 20,
                          clip: float = 0.01) -> float:
    """BN momentum schedule from the paper (0.5 -> 0.99 decay, in PyTorch's convention)."""
    return max(init * decay ** (epoch // step), clip)


def save_checkpoint(path: str, model, optimizer, epoch: int, metrics: dict, args: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "epoch": epoch, "metrics": metrics, "args": args}, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY with per-vertex colour (viewable in MeshLab/CloudCompare)."""
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)
    vertex = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex["x"], vertex["y"], vertex["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {len(xyz)}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex.tobytes())
