"""Convert the raw S3DIS release into one ``.npy`` file per room.

Raw layout (``Stanford3dDataset_v1.2_Aligned_Version``)::

    Area_1/
      conferenceRoom_1/
        conferenceRoom_1.txt          # whole room (ignored)
        Annotations/
          chair_1.txt                 # "x y z r g b" per line, one object instance
          wall_3.txt
          ...

Output: ``<out_dir>/Area_1_conferenceRoom_1.npy`` - float32 array ``(N, 8)`` with
columns ``x y z r g b label instance``. Coordinates are shifted so the room's
minimum corner is at the origin.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from .s3dis import CLASS2LABEL


def load_annotation_txt(path: str) -> np.ndarray:
    """Load an ``x y z r g b`` file, skipping malformed lines.

    The official release contains at least one corrupted line (e.g. in
    ``Area_5/hallway_6/Annotations/ceiling_1.txt``), so fall back to a tolerant
    line-by-line parser if ``np.loadtxt`` fails.
    """
    try:
        data = np.loadtxt(path, dtype=np.float32, ndmin=2)
        if data.shape[1] >= 6:
            return data[:, :6]
    except ValueError:
        pass
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                rows.append([float(p) for p in parts[:6]])
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def class_from_filename(filename: str) -> int:
    """``chair_12.txt`` -> label of ``chair``; unknown classes (e.g. ``stairs``) -> clutter."""
    name = os.path.basename(filename).split("_")[0]
    return CLASS2LABEL.get(name, CLASS2LABEL["clutter"])


def room_to_array(room_dir: str) -> np.ndarray:
    parts = []
    files = sorted(glob.glob(os.path.join(room_dir, "Annotations", "*.txt")))
    for inst_id, path in enumerate(files):
        pts = load_annotation_txt(path)
        if len(pts) == 0:
            continue
        label = np.full((len(pts), 1), class_from_filename(path), dtype=np.float32)
        inst = np.full((len(pts), 1), inst_id, dtype=np.float32)
        parts.append(np.concatenate([pts, label, inst], axis=1))
    if not parts:
        raise RuntimeError(f"No annotations found in {room_dir}")
    data = np.concatenate(parts, axis=0)
    data[:, :3] -= data[:, :3].min(axis=0)
    return data


def find_rooms(raw_root: str) -> list[str]:
    return sorted(d for d in glob.glob(os.path.join(raw_root, "Area_*", "*")) if os.path.isdir(os.path.join(d, "Annotations")))


def preprocess_s3dis(raw_root: str, out_dir: str, overwrite: bool = False, verbose: bool = True) -> list[str]:
    """Convert every room under ``raw_root``; returns the list of written files."""
    os.makedirs(out_dir, exist_ok=True)
    rooms = find_rooms(raw_root)
    if not rooms:
        raise FileNotFoundError(f"No Area_*/<room>/Annotations folders found under {raw_root}")
    written = []
    for i, room_dir in enumerate(rooms):
        area = os.path.basename(os.path.dirname(room_dir))
        room = os.path.basename(room_dir)
        out_path = os.path.join(out_dir, f"{area}_{room}.npy")
        if os.path.exists(out_path) and not overwrite:
            written.append(out_path)
            continue
        data = room_to_array(room_dir)
        np.save(out_path, data)
        written.append(out_path)
        if verbose:
            print(f"[{i + 1}/{len(rooms)}] {area}/{room}: {len(data):,} points -> {out_path}")
    return written
