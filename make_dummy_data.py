"""Create a small synthetic dataset in the raw S3DIS layout and preprocess it.

Lets you run the full pipeline in a couple of minutes on a CPU:
    python make_dummy_data.py --raw-root data/dummy_raw --out data/dummy_npy
    python train_seg.py --data-root data/dummy_npy --epochs 2 --num-point 1024 --batch-size 8
"""
import argparse

from pointnet.data.dummy import make_dummy_s3dis
from pointnet.data.preprocess import preprocess_s3dis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", default="data/dummy_raw")
    parser.add_argument("--out", default="data/dummy_npy")
    parser.add_argument("--rooms-per-area", type=int, default=3)
    parser.add_argument("--points-per-object", type=int, default=1500)
    args = parser.parse_args()
    make_dummy_s3dis(args.raw_root, rooms_per_area=args.rooms_per_area, points_per_object=args.points_per_object)
    preprocess_s3dis(args.raw_root, args.out, overwrite=True)


if __name__ == "__main__":
    main()
