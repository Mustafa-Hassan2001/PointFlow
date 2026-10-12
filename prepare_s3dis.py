"""Convert raw S3DIS (Stanford3dDataset_v1.2_Aligned_Version) into per-room .npy files.

Example:
    python prepare_s3dis.py --raw-root /path/to/Stanford3dDataset_v1.2_Aligned_Version --out data/s3dis_npy
"""
import argparse

from pointnet.data.preprocess import preprocess_s3dis


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", required=True, help="folder containing Area_1 ... Area_6")
    parser.add_argument("--out", default="data/s3dis_npy", help="output folder for .npy rooms")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    files = preprocess_s3dis(args.raw_root, args.out, overwrite=args.overwrite)
    print(f"Done: {len(files)} rooms in {args.out}")


if __name__ == "__main__":
    main()
