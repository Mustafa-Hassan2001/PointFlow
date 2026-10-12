"""PointNet (Qi et al., CVPR 2017) in PyTorch.

The network consumes raw point clouds of shape ``(B, C, N)`` where the first
three channels are always XYZ coordinates and any remaining channels are extra
per-point features (RGB, normalized room coordinates, ...).

Components
----------
* :class:`TNet`            - the "mini-PointNet" that regresses a k x k alignment matrix.
* :class:`PointNetEncoder` - shared MLPs + input/feature transforms + symmetric max-pool.
* :class:`PointNetCls`     - global feature -> class scores (object classification).
* :class:`PointNetSeg`     - local + global features -> per-point class scores (semantic segmentation).
* :func:`feature_transform_regularizer` - ``||I - A A^T||_F^2`` orthogonality loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn(in_ch: int, out_ch: int) -> nn.Sequential:
    """Shared per-point MLP layer (1x1 conv) + BN + ReLU."""
    return nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))


def _fc_bn(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_ch, out_ch, bias=False), nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True))


class TNet(nn.Module):
    """Predicts a ``k x k`` transformation matrix from a ``(B, k, N)`` input.

    The last layer is initialised to zero and the identity is added to its
    output, so the network starts out as an identity transform.
    """

    def __init__(self, k: int = 3):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(_conv_bn(k, 64), _conv_bn(64, 128), _conv_bn(128, 1024))
        self.fc = nn.Sequential(_fc_bn(1024, 512), _fc_bn(512, 256))
        self.out = nn.Linear(256, k * k)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = torch.max(x, dim=2)[0]
        x = self.out(self.fc(x))
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype).flatten()
        return (x + identity).view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """PointNet backbone.

    Args:
        in_channels: number of input channels (>= 3, first three are XYZ).
        input_transform: use the 3x3 input T-Net (applied to XYZ only).
        feature_transform: use the 64x64 feature T-Net.

    Returns from ``forward``:
        global_feat: ``(B, 1024)`` max-pooled global descriptor.
        point_feat:  ``(B, 64, N)`` per-point features after the feature transform.
        trans:       ``(B, 3, 3)`` input transform or ``None``.
        trans_feat:  ``(B, 64, 64)`` feature transform or ``None``.
    """

    def __init__(self, in_channels: int = 3, input_transform: bool = True, feature_transform: bool = True):
        super().__init__()
        if in_channels < 3:
            raise ValueError("in_channels must be >= 3 (XYZ first)")
        self.in_channels = in_channels
        self.stn = TNet(3) if input_transform else None
        self.fstn = TNet(64) if feature_transform else None
        self.mlp1 = nn.Sequential(_conv_bn(in_channels, 64), _conv_bn(64, 64))
        self.mlp2 = nn.Sequential(
            _conv_bn(64, 64),
            _conv_bn(64, 128),
            nn.Conv1d(128, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
        )

    def forward(self, x: torch.Tensor):
        trans = trans_feat = None
        if self.stn is not None:
            trans = self.stn(x[:, :3])
            xyz = torch.bmm(x[:, :3].transpose(1, 2), trans).transpose(1, 2)
            x = torch.cat([xyz, x[:, 3:]], dim=1)

        x = self.mlp1(x)
        if self.fstn is not None:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(1, 2), trans_feat).transpose(1, 2)
        point_feat = x

        x = self.mlp2(x)
        global_feat = torch.max(x, dim=2)[0]
        return global_feat, point_feat, trans, trans_feat


class PointNetCls(nn.Module):
    """Classification network: global feature -> MLP(512, 256, k)."""

    def __init__(self, num_classes: int, in_channels: int = 3, input_transform: bool = True,
                 feature_transform: bool = True, dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, input_transform, feature_transform)
        self.head = nn.Sequential(
            _fc_bn(1024, 512),
            nn.Dropout(dropout),
            _fc_bn(512, 256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """``x``: ``(B, C, N)`` -> logits ``(B, num_classes)``, trans_feat."""
        global_feat, _, _, trans_feat = self.encoder(x)
        return self.head(global_feat), trans_feat


class PointNetSeg(nn.Module):
    """Semantic segmentation network.

    Each point's 64-d local feature is concatenated with the 1024-d global
    feature (1088-d) and passed through a shared MLP(512, 256, 128, k).
    """

    def __init__(self, num_classes: int, in_channels: int = 9, input_transform: bool = True,
                 feature_transform: bool = True, dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, input_transform, feature_transform)
        self.head = nn.Sequential(
            _conv_bn(1088, 512),
            _conv_bn(512, 256),
            _conv_bn(256, 128),
            nn.Dropout(dropout),
            nn.Conv1d(128, num_classes, 1),
        )

    def forward(self, x: torch.Tensor):
        """``x``: ``(B, C, N)`` -> logits ``(B, num_classes, N)``, trans_feat."""
        n = x.shape[2]
        global_feat, point_feat, _, trans_feat = self.encoder(x)
        global_feat = global_feat.unsqueeze(2).expand(-1, -1, n)
        feat = torch.cat([point_feat, global_feat], dim=1)
        return self.head(feat), trans_feat


def feature_transform_regularizer(trans: torch.Tensor) -> torch.Tensor:
    """Orthogonality loss ``mean_b ||I - A A^T||_F^2`` (Eq. 2 in the paper)."""
    k = trans.shape[1]
    identity = torch.eye(k, device=trans.device, dtype=trans.dtype).unsqueeze(0)
    diff = identity - torch.bmm(trans, trans.transpose(1, 2))
    return diff.pow(2).sum(dim=(1, 2)).mean()


class PointNetLoss(nn.Module):
    """Cross entropy + weighted feature-transform regularization.

    Works for both classification logits ``(B, K)`` and segmentation logits
    ``(B, K, N)`` (``F.cross_entropy`` handles the extra dimension).
    """

    def __init__(self, reg_weight: float = 1e-3, class_weights: torch.Tensor | None = None,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.reg_weight = reg_weight
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, trans_feat: torch.Tensor | None = None):
        loss = F.cross_entropy(logits, target, weight=self.class_weights, label_smoothing=self.label_smoothing)
        if trans_feat is not None and self.reg_weight > 0:
            loss = loss + self.reg_weight * feature_transform_regularizer(trans_feat)
        return loss
