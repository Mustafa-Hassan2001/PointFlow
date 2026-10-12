import numpy as np
import pytest
import torch

from pointnet import PointNetCls, PointNetLoss, PointNetSeg, TNet, feature_transform_regularizer
from pointnet.data import NUM_CLASSES, S3DISObjectDataset, S3DISSceneBlocks, S3DISSegDataset, scene_to_blocks
from pointnet.data.dummy import make_dummy_s3dis
from pointnet.data.preprocess import class_from_filename, load_annotation_txt, preprocess_s3dis
from pointnet.metrics import ConfusionMatrix


@pytest.fixture(scope="module")
def npy_root(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    out = tmp_path_factory.mktemp("npy")
    make_dummy_s3dis(str(raw), areas=(1, 5), rooms_per_area=1, points_per_object=300)
    preprocess_s3dis(str(raw), str(out), verbose=False)
    return str(out)


def test_tnet_starts_as_identity():
    tnet = TNet(3).eval()
    out = tnet(torch.randn(2, 3, 50))
    assert torch.allclose(out, torch.eye(3).expand(2, 3, 3))


def test_cls_shapes():
    model = PointNetCls(NUM_CLASSES, in_channels=3)
    logits, trans_feat = model(torch.randn(4, 3, 128))
    assert logits.shape == (4, NUM_CLASSES)
    assert trans_feat.shape == (4, 64, 64)


def test_seg_shapes_and_backward():
    model = PointNetSeg(NUM_CLASSES, in_channels=9)
    x = torch.randn(2, 9, 256)
    logits, trans_feat = model(x)
    assert logits.shape == (2, NUM_CLASSES, 256)
    loss = PointNetLoss()(logits, torch.randint(0, NUM_CLASSES, (2, 256)), trans_feat)
    loss.backward()
    assert torch.isfinite(loss)


def test_permutation_invariance_of_global_feature():
    model = PointNetCls(5, in_channels=3).eval()
    x = torch.randn(1, 3, 64)
    perm = torch.randperm(64)
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x[:, :, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_regularizer_zero_for_orthogonal():
    q, _ = torch.linalg.qr(torch.randn(3, 8, 8))
    assert feature_transform_regularizer(q).item() < 1e-8
    assert feature_transform_regularizer(2 * torch.eye(8).unsqueeze(0)).item() > 1


def test_confusion_matrix():
    cm = ConfusionMatrix(3)
    cm.update([0, 1, 1, 2], [0, 1, 2, 2])
    assert cm.overall_accuracy == pytest.approx(0.75)
    np.testing.assert_allclose(cm.iou, [1.0, 0.5, 0.5])
    assert cm.mean_accuracy == pytest.approx((1 + 1 + 0.5) / 3)


def test_preprocess_helpers(tmp_path):
    f = tmp_path / "chair_1.txt"
    f.write_text("1 2 3 4 5 6\nbad line\n7 8 9 10 11 12\n1 2 3 4 5 \x10\n")
    assert load_annotation_txt(str(f)).shape == (2, 6)
    assert class_from_filename("stairs_1.txt") == class_from_filename("clutter_2.txt")


def test_seg_dataset(npy_root):
    ds = S3DISSegDataset(npy_root, "train", test_area=5, num_point=512, min_block_points=10)
    pts, labels = ds[0]
    assert pts.shape == (512, 9) and labels.shape == (512,)
    assert pts[:, 3:6].max() <= 1.0 and labels.max() < NUM_CLASSES
    val = S3DISSceneBlocks(npy_root, "test", test_area=5, num_point=512)
    assert val[0][0].shape == (512, 9)


def test_scene_blocks_cover_every_point(npy_root):
    data = np.load(f"{npy_root}/Area_5_office_1.npy")
    feats, idx = scene_to_blocks(data[:, :6], data[:, :3].max(0), num_point=256, stride=0.5)
    assert feats.shape[1:] == (256, 9)
    assert set(np.unique(idx)) == set(range(len(data)))


def test_object_dataset(npy_root):
    ds = S3DISObjectDataset(npy_root, "train", test_area=5, num_point=128)
    pts, label = ds[0]
    assert pts.shape == (128, 3)
    assert np.linalg.norm(pts, axis=1).max() <= 1.0 + 1e-5
    assert len(ds) == 8  # 4 structural + 4 furniture objects in one dummy room


def test_end_to_end_scripts(npy_root, tmp_path):
    import test_cls
    import test_seg
    import train_cls
    import train_seg

    seg_dir = str(tmp_path / "seg")
    train_seg.main(["--data-root", npy_root, "--log-dir", seg_dir, "--epochs", "1", "--num-point", "256",
                    "--batch-size", "4", "--workers", "0", "--device", "cpu"])
    cm = test_seg.main(["--data-root", npy_root, "--checkpoint", f"{seg_dir}/best.pth", "--device", "cpu",
                        "--visualize"])
    assert cm.matrix.sum() > 0

    cls_dir = str(tmp_path / "cls")
    train_cls.main(["--data-root", npy_root, "--log-dir", cls_dir, "--epochs", "1", "--num-point", "128",
                    "--batch-size", "4", "--workers", "0", "--device", "cpu"])
    cm = test_cls.main(["--data-root", npy_root, "--checkpoint", f"{cls_dir}/best.pth", "--device", "cpu",
                        "--num-votes", "2"])
    assert cm.matrix.sum() == 8


def test_viewer_html(npy_root):
    from pointnet.viewer import build_viewer_html, encode_room

    data = np.load(f"{npy_root}/Area_5_office_1.npy")
    labels = data[:, 6].astype(np.int64)
    room = encode_room("office_1", data[:, :3], data[:, 3:6], labels, labels, max_points=500)
    assert room["n"] == 500 and room["metrics"]["oa"] == pytest.approx(1.0)
    html = build_viewer_html([room], title="Test Viewer")
    assert html.startswith("<!doctype html>") and "<title>Test Viewer</title>" in html
    assert "__DATA__" not in html
