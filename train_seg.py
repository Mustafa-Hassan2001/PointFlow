"""Train PointNet for semantic segmentation on S3DIS (default: test on Area 5).

Example:
    python train_seg.py --data-root data/s3dis_npy --test-area 5 --epochs 100 --batch-size 16
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from pointnet import PointNetLoss, PointNetSeg
from pointnet.data import CLASSES, NUM_CLASSES, S3DISSceneBlocks, S3DISSegDataset
from pointnet.data.transforms import seg_train_transform
from pointnet.engine import evaluate, train_one_epoch
from pointnet.utils import (bn_momentum_adjust, bn_momentum_for_epoch, get_device, get_logger, save_checkpoint,
                            set_seed, worker_init_fn)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/s3dis_npy")
    p.add_argument("--log-dir", default="runs/seg")
    p.add_argument("--test-area", type=int, default=5)
    p.add_argument("--num-point", type=int, default=4096)
    p.add_argument("--block-size", type=float, default=1.0)
    p.add_argument("--sample-rate", type=float, default=1.0, help="fraction of points used to size an epoch")
    p.add_argument("--no-color", action="store_true", help="use XYZ-only features (6-d input)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=20)
    p.add_argument("--lr-decay", type=float, default=0.5)
    p.add_argument("--reg-weight", type=float, default=1e-3, help="feature transform regularizer weight")
    p.add_argument("--class-weights", action="store_true", help="weight the loss by inverse class frequency")
    p.add_argument("--no-input-transform", action="store_true")
    p.add_argument("--no-feature-transform", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    device = get_device(args.device)
    logger = get_logger(args.log_dir, "seg")
    logger.info(f"args: {json.dumps(vars(args))}")
    logger.info(f"device: {device}")

    use_color = not args.no_color
    train_set = S3DISSegDataset(args.data_root, "train", args.test_area, args.num_point, args.block_size,
                                args.sample_rate, use_color, transform=seg_train_transform())
    val_set = S3DISSceneBlocks(args.data_root, "test", args.test_area, args.num_point, args.block_size, use_color)
    logger.info(f"train rooms: {len(train_set.room_files)}  train blocks/epoch: {len(train_set)}  "
                f"val blocks: {len(val_set)}")

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size,
                              **loader_kw)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, **loader_kw)

    model = PointNetSeg(NUM_CLASSES, train_set.in_channels, not args.no_input_transform,
                        not args.no_feature_transform).to(device)
    weights = torch.from_numpy(train_set.class_weights) if args.class_weights else None
    criterion = PointNetLoss(args.reg_weight, weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_decay)

    best_miou = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        bn_momentum_adjust(model, bn_momentum_for_epoch(epoch, step=args.lr_step))
        train_loss, train_cm = train_one_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES,
                                               desc=f"epoch {epoch + 1}")
        scheduler.step()
        val_loss, val_cm = evaluate(model, val_loader, criterion, device, NUM_CLASSES)
        metrics = {"train_loss": train_loss, "train_oa": train_cm.overall_accuracy, "val_loss": val_loss,
                   "val_oa": val_cm.overall_accuracy, "val_macc": val_cm.mean_accuracy, "val_miou": val_cm.mean_iou}
        logger.info(f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s) "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        save_checkpoint(os.path.join(args.log_dir, "last.pth"), model, optimizer, epoch, metrics, vars(args))
        if val_cm.mean_iou > best_miou:
            best_miou = val_cm.mean_iou
            save_checkpoint(os.path.join(args.log_dir, "best.pth"), model, optimizer, epoch, metrics, vars(args))
            logger.info(f"new best mIoU {best_miou:.4f}\n{val_cm.summary(CLASSES)}")
    logger.info(f"done. best val mIoU (block-level) = {best_miou:.4f}")
    return best_miou


if __name__ == "__main__":
    main()
