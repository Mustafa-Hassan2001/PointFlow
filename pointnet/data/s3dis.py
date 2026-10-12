"""S3DIS datasets for semantic segmentation and object classification.

Both datasets read the per-room ``.npy`` files produced by
:mod:`pointnet.data.preprocess` (columns ``x y z r g b label instance``).
The standard protocol holds out one area (Area 5 by default) for testing.
"""
from __future__ import annotations

import glob
import os

import numpy as np
from torch.utils.data import Dataset

CLASSES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter",
]
CLASS2LABEL = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

# RGB colours used when exporting predictions (PLY files and the HTML viewer).
# Chosen to stay distinguishable on both dark and light backgrounds.
CLASS_COLORS = np.array([
    [201, 178, 143],  # ceiling  - sand
    [138, 111, 78],   # floor    - brown
    [127, 157, 185],  # wall     - steel blue
    [224, 160, 58],   # beam     - amber
    [181, 101, 29],   # column   - rust
    [79, 195, 217],   # window   - cyan
    [155, 107, 214],  # door     - violet
    [229, 83, 61],    # table    - red
    [242, 194, 48],   # chair    - yellow
    [217, 79, 138],   # sofa     - pink
    [63, 174, 106],   # bookcase - green
    [91, 124, 250],   # board    - blue
    [154, 163, 173],  # clutter  - grey
], dtype=np.uint8)


def list_rooms(data_root: str, split: str, test_area: int) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_root, "Area_*.npy")))
    if not files:
        raise FileNotFoundError(f"No Area_*.npy files in {data_root}; run prepare_s3dis.py first")
    tag = f"Area_{test_area}_"
    if split == "train":
        rooms = [f for f in files if not os.path.basename(f).startswith(tag)]
    elif split == "test":
        rooms = [f for f in files if os.path.basename(f).startswith(tag)]
    elif split == "all":
        rooms = files
    else:
        raise ValueError(f"Unknown split {split!r}")
    if not rooms:
        raise RuntimeError(f"Split {split!r} with test_area={test_area} is empty in {data_root}")
    return rooms


def label_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Inverse-frequency class weights ``(max_freq / freq) ** (1/3)`` (as in common S3DIS setups)."""
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes).astype(np.float64)
    freq = counts / max(counts.sum(), 1)
    freq[freq == 0] = freq[freq > 0].min() if np.any(freq > 0) else 1.0
    return np.power(freq.max() / freq, 1.0 / 3.0).astype(np.float32)


def make_block_features(points: np.ndarray, block_center_xy: np.ndarray, room_max: np.ndarray,
                        use_color: bool = True) -> np.ndarray:
    """Build the 9-d PointNet S3DIS input for points inside one block.

    Channels: block-centred ``x y`` + absolute ``z``, ``r g b`` in [0, 1],
    and ``x y z`` normalized by the room extent. Without colour the RGB
    channels are dropped (6-d input).
    """
    xyz = points[:, :3]
    local = xyz.copy()
    local[:, :2] -= block_center_xy
    normalized = xyz / np.maximum(room_max, 1e-6)
    feats = [local]
    if use_color:
        feats.append(points[:, 3:6] / 255.0)
    feats.append(normalized)
    return np.concatenate(feats, axis=1).astype(np.float32)


class S3DISSegDataset(Dataset):
    """Randomly sampled ``block_size x block_size`` (in XY) blocks for training.

    Rooms are sampled proportionally to their point count and each item is a
    block of ``num_point`` points, following the original PointNet setup.
    """

    def __init__(self, data_root: str, split: str = "train", test_area: int = 5, num_point: int = 4096,
                 block_size: float = 1.0, sample_rate: float = 1.0, use_color: bool = True,
                 transform=None, min_block_points: int = 1024):
        self.num_point = num_point
        self.block_size = block_size
        self.use_color = use_color
        self.transform = transform
        self.min_block_points = min_block_points
        self.room_files = list_rooms(data_root, split, test_area)

        self.room_points, self.room_labels, self.room_max = [], [], []
        counts = []
        for f in self.room_files:
            data = np.load(f)
            self.room_points.append(data[:, :6].astype(np.float32))
            self.room_labels.append(data[:, 6].astype(np.int64))
            self.room_max.append(data[:, :3].max(axis=0))
            counts.append(len(data))
        counts = np.asarray(counts, dtype=np.float64)
        all_labels = np.concatenate(self.room_labels)
        self.class_weights = label_weights(all_labels)

        num_iter = max(int(counts.sum() * sample_rate / num_point), 1)
        prob = counts / counts.sum()
        self.room_idxs = np.repeat(np.arange(len(counts)), np.round(prob * num_iter).astype(np.int64))
        if len(self.room_idxs) == 0:
            self.room_idxs = np.arange(len(counts))

    @property
    def in_channels(self) -> int:
        return 9 if self.use_color else 6

    def __len__(self) -> int:
        return len(self.room_idxs)

    def __getitem__(self, idx: int):
        r = self.room_idxs[idx]
        points, labels = self.room_points[r], self.room_labels[r]
        half = self.block_size / 2.0
        point_idxs = None
        for _ in range(10):
            center = points[np.random.randint(len(points)), :2]
            lo, hi = center - half, center + half
            mask = np.all((points[:, :2] >= lo) & (points[:, :2] <= hi), axis=1)
            point_idxs = np.nonzero(mask)[0]
            if len(point_idxs) >= self.min_block_points:
                break
        replace = len(point_idxs) < self.num_point
        selected = np.random.choice(point_idxs, self.num_point, replace=replace)
        block = points[selected]
        feats = make_block_features(block, center, self.room_max[r], self.use_color)
        if self.transform is not None:
            feats = self.transform(feats)
        return feats, labels[selected]


def scene_to_blocks(points: np.ndarray, room_max: np.ndarray, num_point: int = 4096, block_size: float = 1.0,
                    stride: float = 0.5, use_color: bool = True, padding: float = 1e-3):
    """Cover a whole room with overlapping blocks for evaluation.

    Every point of the room ends up in at least one block. Blocks with more
    than ``num_point`` points are split into several samples; the last sample
    is padded by re-using random points from the block.

    Returns:
        feats:   ``(B, num_point, C)`` float32 block features.
        indices: ``(B, num_point)`` int64 indices into ``points``.
    """
    coord_min, coord_max = points[:, :3].min(axis=0), points[:, :3].max(axis=0)
    grid_x = int(np.ceil(max(coord_max[0] - coord_min[0] - block_size, 0) / stride)) + 1
    grid_y = int(np.ceil(max(coord_max[1] - coord_min[1] - block_size, 0) / stride)) + 1
    all_feats, all_idx = [], []
    for ix in range(grid_x):
        for iy in range(grid_y):
            s_x = min(coord_min[0] + ix * stride, max(coord_max[0] - block_size, coord_min[0]))
            s_y = min(coord_min[1] + iy * stride, max(coord_max[1] - block_size, coord_min[1]))
            e_x, e_y = s_x + block_size, s_y + block_size
            mask = ((points[:, 0] >= s_x - padding) & (points[:, 0] <= e_x + padding)
                    & (points[:, 1] >= s_y - padding) & (points[:, 1] <= e_y + padding))
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                continue
            n_batch = int(np.ceil(len(idx) / num_point))
            extra = n_batch * num_point - len(idx)
            if extra:
                idx = np.concatenate([idx, np.random.choice(idx, extra, replace=extra > len(idx))])
            np.random.shuffle(idx)
            center = np.array([s_x + block_size / 2.0, s_y + block_size / 2.0], dtype=np.float32)
            feats = make_block_features(points[idx], center, room_max, use_color)
            all_feats.append(feats.reshape(n_batch, num_point, -1))
            all_idx.append(idx.reshape(n_batch, num_point))
    return np.concatenate(all_feats, axis=0), np.concatenate(all_idx, axis=0)


class S3DISSceneBlocks(Dataset):
    """Deterministic block sampling of a fixed set of rooms (fast validation).

    Each item is one block of ``num_point`` points produced by
    :func:`scene_to_blocks` with stride == block size (no overlap).
    """

    def __init__(self, data_root: str, split: str = "test", test_area: int = 5, num_point: int = 4096,
                 block_size: float = 1.0, use_color: bool = True, seed: int = 0):
        rng_state = np.random.get_state()
        np.random.seed(seed)
        feats, labels = [], []
        try:
            for f in list_rooms(data_root, split, test_area):
                data = np.load(f)
                pts = data[:, :6].astype(np.float32)
                block_feats, idx = scene_to_blocks(pts, data[:, :3].max(axis=0), num_point, block_size,
                                                   block_size, use_color)
                feats.append(block_feats)
                labels.append(data[:, 6].astype(np.int64)[idx])
        finally:
            np.random.set_state(rng_state)
        self.feats = np.concatenate(feats)
        self.labels = np.concatenate(labels)

    def __len__(self) -> int:
        return len(self.feats)

    def __getitem__(self, idx: int):
        return self.feats[idx], self.labels[idx]


def normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xyz, axis=1))
    return xyz / max(scale, 1e-6)


class S3DISObjectDataset(Dataset):
    """Object classification on S3DIS: every annotated instance is one sample.

    Each object is centred and scaled to the unit sphere, then ``num_point``
    points are sampled from it. To keep memory bounded, objects are
    sub-sampled to at most ``max_cached_points`` points when loaded.
    """

    def __init__(self, data_root: str, split: str = "train", test_area: int = 5, num_point: int = 1024,
                 use_color: bool = False, transform=None, min_points: int = 50,
                 max_cached_points: int = 8192, seed: int = 0):
        self.num_point = num_point
        self.use_color = use_color
        self.transform = transform
        rng = np.random.default_rng(seed)
        self.objects, labels = [], []
        for f in list_rooms(data_root, split, test_area):
            data = np.load(f)
            inst = data[:, 7].astype(np.int64)
            order = np.argsort(inst, kind="stable")
            inst_sorted = inst[order]
            boundaries = np.flatnonzero(np.diff(inst_sorted)) + 1
            for group in np.split(order, boundaries):
                if len(group) < min_points:
                    continue
                if len(group) > max_cached_points:
                    group = rng.choice(group, max_cached_points, replace=False)
                self.objects.append(data[group, :6].astype(np.float32))
                labels.append(int(data[group[0], 6]))
        if not self.objects:
            raise RuntimeError("No objects found; check data_root / min_points")
        self.labels = np.asarray(labels, dtype=np.int64)
        self.class_weights = label_weights(self.labels)

    @property
    def in_channels(self) -> int:
        return 6 if self.use_color else 3

    def __len__(self) -> int:
        return len(self.objects)

    def __getitem__(self, idx: int):
        obj = self.objects[idx]
        choice = np.random.choice(len(obj), self.num_point, replace=len(obj) < self.num_point)
        pts = obj[choice]
        feats = [normalize_unit_sphere(pts[:, :3])]
        if self.use_color:
            feats.append(pts[:, 3:6] / 255.0)
        feats = np.concatenate(feats, axis=1).astype(np.float32)
        if self.transform is not None:
            feats = self.transform(feats)
        return feats, self.labels[idx]
