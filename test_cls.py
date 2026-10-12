"""Evaluate a trained PointNet classifier on S3DIS objects with test-time voting.

Each object is classified ``--num-votes`` times with different random point
samples (and random Z rotations when votes > 1); class scores are averaged.

Example:
    python test_cls.py --data-root data/s3dis_npy --checkpoint runs/cls/best.pth --num-votes 5
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from pointnet import PointNetCls
from pointnet.data import CLASSES, NUM_CLASSES, S3DISObjectDataset
from pointnet.data.transforms import RandomRotateZ
from pointnet.metrics import ConfusionMatrix
from pointnet.utils import get_device, get_logger, load_checkpoint, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/s3dis_npy")
    p.add_argument("--checkpoint", default="runs/cls/best.pth")
    p.add_argument("--test-area", type=int, default=None, help="defaults to the area used in training")
    p.add_argument("--num-votes", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    train_args = ckpt["args"]
    test_area = args.test_area if args.test_area is not None else train_args["test_area"]
    logger = get_logger(os.path.join(os.path.dirname(args.checkpoint), "eval"), "test_cls")

    transform = RandomRotateZ() if args.num_votes > 1 else None
    dataset = S3DISObjectDataset(args.data_root, "test", test_area, train_args["num_point"],
                                 train_args.get("use_color", False), transform=transform,
                                 min_points=train_args.get("min_points", 50))
    loader = DataLoader(dataset, args.batch_size, shuffle=False)

    model = PointNetCls(NUM_CLASSES, dataset.in_channels, not train_args.get("no_input_transform", False),
                        not train_args.get("no_feature_transform", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scores = np.zeros((len(dataset), NUM_CLASSES), dtype=np.float32)
    for _ in range(args.num_votes):
        offset = 0
        for points, _ in loader:
            logits, _ = model(points.to(device).float().transpose(1, 2))
            scores[offset:offset + len(points)] += torch.softmax(logits, dim=1).cpu().numpy()
            offset += len(points)

    cm = ConfusionMatrix(NUM_CLASSES)
    cm.update(scores.argmax(axis=1), dataset.labels)
    logger.info(f"Area {test_area} object classification ({len(dataset)} objects, {args.num_votes} votes)\n"
                f"{cm.summary(CLASSES)}")
    return cm


if __name__ == "__main__":
    main()
