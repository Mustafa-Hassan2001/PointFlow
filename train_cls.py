"""Train PointNet for object classification on S3DIS instances (default: test on Area 5).

Every annotated object instance (chair_1, table_3, ...) becomes one sample
labelled with its semantic class.

Example:
    python train_cls.py --data-root data/s3dis_npy --epochs 200 --batch-size 32
"""
import argparse
import json
import os
import time

import torch
from torch.utils.data import DataLoader

from pointnet import PointNetCls, PointNetLoss
from pointnet.data import CLASSES, NUM_CLASSES, S3DISObjectDataset
from pointnet.data.transforms import cls_train_transform
from pointnet.engine import evaluate, train_one_epoch
from pointnet.utils import (bn_momentum_adjust, bn_momentum_for_epoch, get_device, get_logger, save_checkpoint,
                            set_seed, worker_init_fn)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="data/s3dis_npy")
    p.add_argument("--log-dir", default="runs/cls")
    p.add_argument("--test-area", type=int, default=5)
    p.add_argument("--num-point", type=int, default=1024)
    p.add_argument("--use-color", action="store_true", help="add RGB to the XYZ input (6-d)")
    p.add_argument("--min-points", type=int, default=50, help="skip objects with fewer points")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lr-step", type=int, default=20)
    p.add_argument("--lr-decay", type=float, default=0.7)
    p.add_argument("--reg-weight", type=float, default=1e-3)
    p.add_argument("--class-weights", action="store_true", help="weight the loss by inverse class frequency")
    p.add_argument("--label-smoothing", type=float, default=0.0)
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
    logger = get_logger(args.log_dir, "cls")
    logger.info(f"args: {json.dumps(vars(args))}")
    logger.info(f"device: {device}")

    train_set = S3DISObjectDataset(args.data_root, "train", args.test_area, args.num_point, args.use_color,
                                   transform=cls_train_transform(), min_points=args.min_points)
    val_set = S3DISObjectDataset(args.data_root, "test", args.test_area, args.num_point, args.use_color,
                                 min_points=args.min_points)
    logger.info(f"train objects: {len(train_set)}  test objects: {len(val_set)}")

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda", worker_init_fn=worker_init_fn)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, drop_last=len(train_set) > args.batch_size,
                              **loader_kw)
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, **loader_kw)

    model = PointNetCls(NUM_CLASSES, train_set.in_channels, not args.no_input_transform,
                        not args.no_feature_transform).to(device)
    weights = torch.from_numpy(train_set.class_weights) if args.class_weights else None
    criterion = PointNetLoss(args.reg_weight, weights, args.label_smoothing).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_decay)

    best_acc = -1.0
    for epoch in range(args.epochs):
        t0 = time.time()
        bn_momentum_adjust(model, bn_momentum_for_epoch(epoch, step=args.lr_step))
        train_loss, train_cm = train_one_epoch(model, train_loader, criterion, optimizer, device, NUM_CLASSES,
                                               desc=f"epoch {epoch + 1}")
        scheduler.step()
        val_loss, val_cm = evaluate(model, val_loader, criterion, device, NUM_CLASSES)
        metrics = {"train_loss": train_loss, "train_acc": train_cm.overall_accuracy, "val_loss": val_loss,
                   "val_acc": val_cm.overall_accuracy, "val_macc": val_cm.mean_accuracy}
        logger.info(f"epoch {epoch + 1}/{args.epochs} ({time.time() - t0:.0f}s) "
                    + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        save_checkpoint(os.path.join(args.log_dir, "last.pth"), model, optimizer, epoch, metrics, vars(args))
        if val_cm.overall_accuracy > best_acc:
            best_acc = val_cm.overall_accuracy
            save_checkpoint(os.path.join(args.log_dir, "best.pth"), model, optimizer, epoch, metrics, vars(args))
            logger.info(f"new best accuracy {best_acc:.4f}\n{val_cm.summary(CLASSES)}")
    logger.info(f"done. best val accuracy = {best_acc:.4f}")
    return best_acc


if __name__ == "__main__":
    main()
