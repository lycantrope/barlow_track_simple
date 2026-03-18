import os
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from barlow_track_simple.siamese import (
    Encoder,
    ResidualEncoder3D,
    Siamese,
)


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def init_weights_kaiming(m):
    # Check if the module is a 3D Convolution
    if isinstance(m, nn.Conv3d):
        # Using 'fan_out' preserves magnitudes in the backward pass
        # 'nonlinearity' should match your activation (e.g., 'relu')
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    # Also initialize BatchNorm if present to keep signals stable
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class BarlowTwinsEmbed3D(nn.Module):
    def __init__(
        self,
        projector_str: str,
        backbone: Encoder,
        projector_final: Optional[int] = None,
    ):
        super().__init__()
        self.backbone = backbone

        # projector
        sizes = [self.backbone.embedding_dim] + list(map(int, projector_str.split("-")))
        if projector_final is not None:
            # Otherwise assume it's all in the original projector string
            sizes.append(projector_final)

        layers = []
        for i in range(len(sizes) - 2):
            layers.extend(
                [
                    nn.Linear(sizes[i], sizes[i + 1], bias=False),
                    nn.BatchNorm1d(sizes[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            )

        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        # self.bn = nn.BatchNorm1d(sizes[-1], affine=False)
        # self.bn = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(x))

    @classmethod
    def init_model(
        cls,
        projector: str,
        crop_sz: Tuple[int, int, int],
        backbone_type: Literal["ResidualEncoder3D", "Siamese"],
        projector_final: Optional[int] = None,
    ) -> "BarlowTwinsEmbed3D":
        """
        Loads a model directly from the weights file, and assumes the args are saved in the same folder as args.pickle

        """

        if backbone_type == "ResidualEncoder3D":
            backbone = ResidualEncoder3D(
                in_channels=1,
                num_levels=2,
                f_maps=4,
                crop_sz=crop_sz,
            ).apply(init_weights_kaiming)

        else:
            # This is deprecated ?
            raise ValueError("backbone_type only support ResidualEncoder3D")
            backbone = Siamese()

        model = cls(
            projector,
            backbone=backbone,
            projector_final=projector_final,
        )

        return model


class BarlowTwinsDualLoss(nn.Module):
    def __init__(self, lambd: float, lambd_obj: float, eps: float = 1e-6):
        super().__init__()
        self.lambd = lambd
        self.lambd_obj = lambd_obj
        self.eps = eps

    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def loss_from_matrix(self, c: torch.Tensor):
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        return (on_diag + self.lambd * off_diag) / c.shape[0]

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        # Feature Space Correlation (D x D)
        z1_f = (z1 - z1.mean(0)) / (z1.std(0) + self.eps)
        z2_f = (z2 - z2.mean(0)) / (z2.std(0) + self.eps)
        c_feat = torch.matmul(z1_f.T, z2_f) / z1.shape[0]

        # Object Space Correlation (N x N)
        # z1_o = (z1 - z1.mean(1, keepdim=True)) / (z1.std(1, keepdim=True) + self.eps)
        # z2_o = (z2 - z2.mean(1, keepdim=True)) / (z2.std(1, keepdim=True) + self.eps)
        # We tried to use normalization of features only to make this works
        c_obj = torch.matmul(z1_f, z2_f.T) / z1.shape[1]

        l_feat = self.loss_from_matrix(c_feat)
        l_obj = self.loss_from_matrix(c_obj)

        total_loss = (1.0 - self.lambd_obj) * l_feat + self.lambd_obj * l_obj
        return total_loss, l_feat, l_obj


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=False,
        lars_adaptation_filter=False,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p) -> bool:
        return p.ndim == 1

    @torch.no_grad()
    def step(self) -> None:  # type: ignore
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if not g["weight_decay_filter"] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if not g["lars_adaptation_filter"] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])
