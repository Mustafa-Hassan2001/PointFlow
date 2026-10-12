"""Whole-room semantic segmentation evaluation with sliding-window voting.

Every room of the test area is covered with overlapping blocks (``--stride``);
per-point logits from all blocks (and ``--num-votes`` random re-samplings) are
summed before taking the argmax, so every original point gets a prediction.

Example:
    python test_seg.py --data-root data/s3dis_npy --checkpoint runs/seg/best.pth --visualize
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from pointnet import PointNetSeg
from pointnet.data import CLASS_COLORS, CLASSES, NUM_CLASSES, scene_to_blocks
from pointnet.data.s3dis import list_rooms
from pointnet.metrics import ConfusionMatrix
from pointnet.utils import get_device, get_logger, load_checkpoint, set_seed, write_ply


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/s3dis_npy")
    p.add_argument("--checkpoint", default="runs/seg/best.pth")
    p.add_argument("--out-dir", default=None, help="defaults to <checkpoint dir>/eval")
    p.add_argument("--test-area", type=int, default=None, help="defaults to the area used in training")
    p.add_argument("--stride", type=float, default=0.5)
    p.add_argument("--num-votes", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--visualize", action="store_true", help="write prediction / ground-truth PLY files")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def predict_room(model, points, room_max, train_args, stride, num_votes, batch_size, device):
    use_color = not train_args.get("no_color", False)
    scores = np.zeros((len(points), NUM_CLASSES), dtype=np.float32)
    for _ in range(num_votes):
        feats, idx = scene_to_blocks(points, room_max, train_args["num_point"], train_args["block_size"], stride,
                                     use_color)
        for s in range(0, len(feats), batch_size):
            batch = torch.from_numpy(feats[s:s + batch_size]).to(device).transpose(1, 2)
            logits, _ = model(batch)
            prob = torch.softmax(logits, dim=1).transpose(1, 2).reshape(-1, NUM_CLASSES).cpu().numpy()
            np.add.at(scores, idx[s:s + batch_size].reshape(-1), prob)
    return scores.argmax(axis=1)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    ckpt = load_checkpoint(args.checkpoint, device)
    train_args = ckpt["args"]
    test_area = args.test_area if args.test_area is not None else train_args["test_area"]
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    logger = get_logger(out_dir, "test_seg")

    in_channels = 6 if train_args.get("no_color", False) else 9
    model = PointNetSeg(NUM_CLASSES, in_channels, not train_args.get("no_input_transform", False),
                        not train_args.get("no_feature_transform", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    total = ConfusionMatrix(NUM_CLASSES)
    for f in tqdm(list_rooms(args.data_root, "test", test_area), desc="rooms"):
        data = np.load(f)
        points, labels = data[:, :6].astype(np.float32), data[:, 6].astype(np.int64)
        pred = predict_room(model, points, data[:, :3].max(axis=0), train_args, args.stride, args.num_votes,
                            args.batch_size, device)
        room_cm = ConfusionMatrix(NUM_CLASSES)
        room_cm.update(pred, labels)
        total.update(pred, labels)
        name = os.path.splitext(os.path.basename(f))[0]
        logger.info(f"{name}: OA={room_cm.overall_accuracy:.4f} mIoU={room_cm.mean_iou:.4f}")
        if args.visualize:
            write_ply(os.path.join(out_dir, f"{name}_pred.ply"), points[:, :3], CLASS_COLORS[pred])
            write_ply(os.path.join(out_dir, f"{name}_gt.ply"), points[:, :3], CLASS_COLORS[labels])
            np.savetxt(os.path.join(out_dir, f"{name}_pred.txt"), pred, fmt="%d")

    logger.info(f"Area {test_area} whole-scene results\n{total.summary(CLASSES)}")
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), total.matrix)
    return total


if __name__ == "__main__":
    main()
