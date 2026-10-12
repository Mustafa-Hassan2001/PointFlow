"""Run a trained segmentation model on test rooms and write an interactive 3D HTML viewer.

Open the resulting file in a browser to orbit through each room and compare
prediction, ground truth, an error map and the original RGB colours.

Example:
    python visualize.py --data-root data/s3dis_npy --checkpoint runs/seg/best.pth --out runs/seg/viewer.html
"""
import argparse
import os

import numpy as np
import torch

from pointnet import PointNetSeg
from pointnet.data import NUM_CLASSES
from pointnet.data.s3dis import list_rooms
from pointnet.utils import get_device, load_checkpoint, set_seed
from pointnet.viewer import build_viewer_html, encode_room
from test_seg import predict_room


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/s3dis_npy")
    p.add_argument("--checkpoint", default="runs/seg/best.pth")
    p.add_argument("--out", default=None, help="defaults to <checkpoint dir>/viewer.html")
    p.add_argument("--test-area", type=int, default=None, help="defaults to the area used in training")
    p.add_argument("--rooms", nargs="*", default=None, help="room names to include (e.g. office_1); default all")
    p.add_argument("--max-rooms", type=int, default=6, help="cap on rooms embedded (keeps the file small)")
    p.add_argument("--max-points", type=int, default=150_000, help="display points per room (metrics use all)")
    p.add_argument("--stride", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--title", default="PointNet S3DIS Viewer")
    p.add_argument("--notice", default="", help="optional HTML note shown under the header")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    train_args = ckpt["args"]
    test_area = args.test_area if args.test_area is not None else train_args["test_area"]

    in_channels = 6 if train_args.get("no_color", False) else 9
    model = PointNetSeg(NUM_CLASSES, in_channels, not train_args.get("no_input_transform", False),
                        not train_args.get("no_feature_transform", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    files = list_rooms(args.data_root, "test", test_area)
    if args.rooms:
        files = [f for f in files if any(os.path.basename(f) == f"Area_{test_area}_{r}.npy" for r in args.rooms)]
    files = files[:args.max_rooms]

    rooms = []
    for f in files:
        data = np.load(f)
        points, labels = data[:, :6].astype(np.float32), data[:, 6].astype(np.int64)
        pred = predict_room(model, points, data[:, :3].max(axis=0), train_args, args.stride, 1, args.batch_size,
                            device)
        name = os.path.splitext(os.path.basename(f))[0].replace(f"Area_{test_area}_", "")
        rooms.append(encode_room(name, points[:, :3], points[:, 3:6], labels, pred, args.max_points))
        print(f"{name}: {len(points):,} points, mIoU {rooms[-1]['metrics']['miou']:.4f}")

    subtitle = (f"PointNet semantic segmentation · Area {test_area} · checkpoint epoch {ckpt['epoch'] + 1} · "
                f"{'XYZ+RGB' if in_channels == 9 else 'XYZ'} input")
    html = build_viewer_html(rooms, args.title, subtitle, args.notice)
    out = args.out or os.path.join(os.path.dirname(args.checkpoint), "viewer.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"Wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
