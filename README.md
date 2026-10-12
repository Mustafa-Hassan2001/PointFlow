# PointFlow: 3D Point Cloud Segmentation and Classification with PointNet

This is a PyTorch re-implementation of **PointNet** ([Qi et al., CVPR 2017](https://arxiv.org/abs/1612.00593)). It works directly on raw 3D point clouds and needs no voxel grids, meshes or multi-view images in between. It is trained and evaluated on the **S3DIS** indoor dataset (Stanford Large-Scale 3D Indoor Spaces) for two tasks:

* **Per-point semantic segmentation** labels every point with one of 13 classes.
* **Object classification** gives one label to each annotated object instance, such as a chair, table or wall.

## Architecture

```
            input points (B, C, N)   C = xyz [+ rgb] [+ normalized xyz]
                     │
          ┌──── T-Net 3x3 (input transform, applied to xyz) ────┐
                     │
              shared MLP (64, 64)
                     │
          ┌──── T-Net 64x64 (feature transform) ────┐   ──►  64-d local point features ─────┐
                     │                                                                      │
              shared MLP (64, 128, 1024)                                                    │
                     │                                                                      │
              max-pool (symmetric) ──► 1024-d global feature                                │
                     │                          │                                           │
     Classification: │                          └── Segmentation: repeat N times, concat ◄──┘
     MLP (512, 256, K) → class scores                 (1088-d) → shared MLP (512, 256, 128, K)
                                                       → per-point class scores
```

* **T-Nets** are small PointNets that predict alignment matrices. They are initialized to the identity.
* **Feature-transform regularization** adds `L_reg = ||I − A Aᵀ||²_F` to the loss (weight `1e-3`), which keeps the 64×64 transform close to orthogonal.
* **Max-pooling** is the symmetric function that makes the network invariant to the order of the input points. A unit test checks this.

## Project layout

```
pointnet/
  model.py            TNet, PointNetEncoder, PointNetCls, PointNetSeg, loss + regularizer
  metrics.py          confusion matrix → OA, mean class accuracy, per-class IoU / mIoU
  engine.py           shared train / eval loops
  utils.py            seeding, logging, checkpoints, BN-momentum schedule, PLY export
  data/
    preprocess.py     raw S3DIS → per-room .npy (x y z r g b label instance)
    s3dis.py          segmentation block datasets, whole-scene blocking, object dataset
    transforms.py     rotation / scaling / jitter / shuffle augmentations
    dummy.py          synthetic rooms in the raw S3DIS layout (for smoke tests)
  viewer.py           builds the self-contained three.js viewer page
prepare_s3dis.py      preprocessing CLI
train_seg.py          train semantic segmentation
test_seg.py           whole-room evaluation with sliding-window voting + PLY export
train_cls.py          train object classification
test_cls.py           evaluate classification with multi-vote testing
visualize.py          interactive 3D HTML viewer of predictions vs. ground truth
make_dummy_data.py    build a tiny synthetic dataset to try the pipeline on CPU
tests/                pytest suite (model, metrics, datasets, end-to-end scripts)
```

## Setup

```bash
pip install -r requirements.txt      # torch, numpy, tqdm
pip install pytest                   # optional, for tests
```

## Data

1. Request and download S3DIS from the [official site](http://buildingparser.stanford.edu/dataset.html). You need `Stanford3dDataset_v1.2_Aligned_Version.zip`.
2. Unzip it and convert each room to a compact `.npy` file:

```bash
python prepare_s3dis.py --raw-root /path/to/Stanford3dDataset_v1.2_Aligned_Version --out data/s3dis_npy
```

Each room becomes `Area_<k>_<room>.npy` with columns `x y z r g b label instance`. Coordinates are shifted so that the room's minimum corner sits at the origin. Objects of unknown classes (for example `stairs`) are mapped to `clutter`. The loader also skips the malformed line in the official `Area_5/hallway_6` file.

**Classes (13):** ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board, clutter.

**Split:** the standard protocol trains on Areas 1, 2, 3, 4 and 6 and tests on **Area 5** (`--test-area 5`). Use a different `--test-area` to run 6-fold cross-validation.

## Semantic segmentation

```bash
python train_seg.py --data-root data/s3dis_npy --test-area 5 --epochs 100 --batch-size 16
python test_seg.py  --data-root data/s3dis_npy --checkpoint runs/seg/best.pth --visualize
```

* **Training input.** The model trains on random 1 m × 1 m blocks in the XY plane, with 4096 points sampled from each block. Every point has 9 features: XY centred on the block plus absolute Z, RGB scaled to [0, 1], and XYZ normalized by the room's size. This follows the original PointNet S3DIS setup. Pass `--no-color` to train on geometry only (6 features).
* **Validation during training.** After each epoch the model is scored on non-overlapping blocks from the test area. The checkpoint with the best block-level mIoU is saved to `runs/seg/best.pth`.
* **Final evaluation.** `test_seg.py` covers each test room with overlapping blocks (`--stride 0.5`). It sums the softmax scores for every original point across all the blocks that contain it (and across `--num-votes` re-samplings), then takes the argmax. It reports OA, mAcc, per-class IoU and mIoU over **all points** of the test area.
* **Visualization.** `--visualize` writes `*_pred.ply` and `*_gt.ply` files with class colours. Open them in MeshLab or CloudCompare.

### Interactive 3D viewer

```bash
python visualize.py --data-root data/s3dis_npy --checkpoint runs/seg/best.pth --out runs/seg/viewer.html
```

This runs the model on the test rooms and writes one HTML file. Open it in a browser to:

* orbit, pan and zoom through each room (drag, shift-drag or right-drag, scroll, pinch on touch screens)
* colour the points by prediction, ground truth, an error map (misclassified points in red) or the original RGB
* read OA and mIoU for the room and the whole test area, plus per-class IoU
* click a class in the legend to hide or show it

The page loads three.js from a CDN, so it needs an internet connection. Use `--max-rooms` and `--max-points` to keep the file small; metrics always use every point.

Useful flags: `--class-weights` (inverse-frequency weighted loss), `--no-input-transform` and `--no-feature-transform` (ablations), and `--sample-rate` (sets how many blocks make up one epoch).

## Object classification

```bash
python train_cls.py --data-root data/s3dis_npy --test-area 5 --epochs 200 --batch-size 32
python test_cls.py  --data-root data/s3dis_npy --checkpoint runs/cls/best.pth --num-votes 5
```

Each annotated S3DIS instance (`chair_3`, `table_1`, …) becomes one sample:

1. The script samples 1024 points from the object.
2. It centres them and scales them to the unit sphere.
3. During training it augments them with a random Z rotation, scaling, jitter and shuffling.

Add `--use-color` to append RGB to the input. With `--num-votes`, `test_cls.py` averages the predictions over several random re-samplings and rotations of each object.

## Quick start without S3DIS

You can check the whole pipeline on synthetic rooms in a few minutes on a CPU:

```bash
python make_dummy_data.py --raw-root data/dummy_raw --out data/dummy_npy
python train_seg.py --data-root data/dummy_npy --log-dir runs/dummy_seg --epochs 15 --num-point 1024 --batch-size 8 --sample-rate 5
python test_seg.py  --data-root data/dummy_npy --checkpoint runs/dummy_seg/best.pth
python visualize.py --data-root data/dummy_npy --checkpoint runs/dummy_seg/best.pth   # writes runs/dummy_seg/viewer.html
python train_cls.py --data-root data/dummy_npy --log-dir runs/dummy_cls --epochs 15 --num-point 256 --batch-size 8
```

## Tests

```bash
python -m pytest -q
```

The tests cover:

* output shapes and gradients
* T-Net identity initialization
* permutation invariance
* the regularizer
* metric arithmetic
* parsing of malformed lines
* the datasets
* a full train → test run of all four scripts on synthetic data

## Reference results

The PointNet paper reports these numbers for S3DIS semantic segmentation (6-fold cross-validation, XYZ + RGB + normalized location input):

| Method | Overall accuracy | mIoU |
|---|---|---|
| PointNet (paper) | 78.6 | 47.6 |

Results on Area 5 alone are usually reported lower than the 6-fold average, at roughly 41–43 mIoU for PointNet. Run `test_seg.py` on your own checkpoint to get your numbers.

## Training defaults (from the paper)

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3, weight decay 1e-4 |
| Learning-rate schedule | Step decay, ×0.5 every 20 epochs (segmentation) or ×0.7 (classification) |
| BatchNorm momentum | Decays from 0.1 to 0.01 (PyTorch convention) |
| Dropout | 0.3 drop rate, i.e. keep probability 0.7 |
| Feature-transform regularizer weight | 1e-3 |

## Citation

```bibtex
@inproceedings{qi2017pointnet,
  title     = {PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
  author    = {Qi, Charles R. and Su, Hao and Mo, Kaichun and Guibas, Leonidas J.},
  booktitle = {CVPR},
  year      = {2017}
}
@inproceedings{armeni2016s3dis,
  title     = {3D Semantic Parsing of Large-Scale Indoor Spaces},
  author    = {Armeni, Iro and Sener, Ozan and Zamir, Amir R. and Jiang, Helen and Brilakis, Ioannis and Fischer, Martin and Savarese, Silvio},
  booktitle = {CVPR},
  year      = {2016}
}
```
