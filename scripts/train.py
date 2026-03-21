import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch
from ruamel.yaml import YAML, scalarfloat
from tqdm.auto import tqdm

from barlow_track_simple.config import get_device
from barlow_track_simple.dataloader import get_train_loader, verify_input_data
from barlow_track_simple.model import (
    BarlowTwinsDualLoss,
    BarlowTwinsEmbed3D,
)
from barlow_track_simple.plot import plot_loss, plot_matrices

yaml = YAML()

# This will tell pytorch load scalarfloat safely
torch.serialization.add_safe_globals([scalarfloat.ScalarFloat])


DEVICE = get_device()

print(f"Use device: {DEVICE}")


def check_feature_collapse(features: torch.Tensor) -> torch.Tensor:
    # features: [N, 512]
    # Center the features
    features = features - features.mean(dim=0)

    # Get Singular Values
    _, S, _ = torch.svd(features)

    # Normalize singular values so they sum to 1
    S_norm = S / S.sum()

    # Calculate Shannon Entropy of the singular values
    # Higher entropy = No collapse (features are spread out)
    # Lower entropy = Collapse (only a few dimensions carry info)
    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-8))
    return entropy


def run_epoch(
    epoch: int,
    loader: torch.utils.data.DataLoader,
    model: Union[BarlowTwinsEmbed3D, torch.nn.DataParallel],
    loss_fn: BarlowTwinsDualLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
):
    model.train()
    epoch_loss = 0.0
    valid_batch_count = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    for batch_idx, (meta, y1, y2) in pbar:
        optimizer.zero_grad(set_to_none=True)
        meta = meta.to(DEVICE, non_blocking=True)  # type: torch.Tensor
        y1 = y1.to(DEVICE, non_blocking=True)  # type: torch.Tensor
        y2 = y2.to(DEVICE, non_blocking=True)  # type: torch.Tensor

        # [Batch, n_obj_max, Z, Y, X]
        Z, Y, X = y1.shape[-3:]
        # [Batch, n_obj_max, Z, Y, X] => [batch*n_obj_max, 1, Z, Y, X]
        y1 = y1.view(-1, 1, Z, Y, X)
        y2 = y2.view(-1, 1, Z, Y, X)
        # [Batch, n_obj_max, 6] => [Batch * n_obj_max, 6]
        t_indice_flat = meta[:, :, 1].view(-1)

        mask = t_indice_flat >= 0

        valid_t_indice = t_indice_flat[mask]
        unique_times = valid_t_indice.unique()

        y1_filtered = y1[mask]  # type: torch.Tensor
        y2_filtered = y2[mask]  # type: torch.Tensor

        # Forward pass
        z1 = model(y1_filtered)
        z2 = model(y2_filtered)

        total_loss = torch.tensor(0.0, device=DEVICE)
        valid_loss_count = 0
        for t in unique_times:
            # Filter features belonging only to time 't'
            t_mask = valid_t_indice == t
            z1_t = z1[t_mask]
            z2_t = z2[t_mask]
            if z1_t.size(0) < 2:
                continue

            # Check the mean of the standard deviations
            with torch.no_grad():
                epoch_variances = torch.std(z1_t, dim=0)
                batch_std = epoch_variances.mean().item()
                shannon_corr = check_feature_collapse(z1_t).item()
            if batch_std < 1e-4:
                print(
                    f"Wanring Epoch {epoch} at Batch {batch_idx}-{t} Std: {batch_std:.6f} | Shannon:{shannon_corr:.3f}"
                )

            loss, _, _ = loss_fn(z1_t, z2_t)

            total_loss += loss
            valid_loss_count += 1
            epoch_loss += loss.item()

        valid_batch_count += valid_loss_count
        # Backward pass
        if valid_loss_count == 0:
            print("No batch detect please check your dataset")
        else:
            (total_loss / valid_loss_count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        pbar.set_postfix({"loss": f"{total_loss.item():.4f} "})

    avg_loss = epoch_loss / valid_batch_count
    return avg_loss


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)

    args = parser.parse_args()

    cfg_path = Path(args.cfg_path)
    if not cfg_path.is_file():
        parser.error("cfg_path is not a valid files")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.load(f)
    print(cfg)
    w_z = cfg["target_sz_z"]
    w_xy = cfg.get("target_sz_xy", w_z)
    crop_sz = (w_z, w_xy, w_xy)

    datapaths = [
        f
        for f in Path(args.data_folder).glob("*.*")
        if f.name.endswith((".zarr", ".zarr.zip", ".tif", ".h5"))
    ]
    csvpaths = [f for f in Path(args.data_folder).glob("*.csv")]

    dataset_list = verify_input_data(
        datapaths,
        csvpaths,
        cfg.get("channel", ""),
        cfg.get("hdf_key", ""),
    )

    pretrained_model_path = cfg.get("pretrained_model_path")
    state_dict = {}
    if pretrained_model_path:
        pretrained_model_path = Path(pretrained_model_path)
        if not pretrained_model_path.is_absolute():
            # Relative to config.yaml
            pretrained_model_path = cfg_path.parent / pretrained_model_path
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        print(f"Model weights loaded: {pretrained_model_path}")
    else:
        print("Cannot find pretrain model, train model from scratch.")

    model = BarlowTwinsEmbed3D.init_model(
        cfg["projector"],
        crop_sz,
        cfg["backbone_type"],
        cfg.get("projector_final"),
    )

    if state_dict:
        model.load_state_dict(state_dict.get("model", state_dict))
    model.to(DEVICE)

    # 2. Wrap the model
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!")
        # This automatically splits the batch and replicates the model
        model = torch.nn.DataParallel(model)

    loss_fn = BarlowTwinsDualLoss(
        lambd_feat=cfg["lambd_feat"],
        lambd_obj=cfg["lambd_obj"],
        alpha=cfg["alpha"],
    )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    if "optimizer" in state_dict:
        optimizer.load_state_dict(state_dict["optimizer"])

    loaders = get_train_loader(
        dataset_list=dataset_list,
        target_sz=crop_sz,
        batch_size=cfg.get("batch_size", 64),
    )

    total_steps = len(loaders["train"]) * cfg["epochs"]
    schedular = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=cfg["lr"] * 4.0,
        total_steps=total_steps,
        pct_start=0.3,  # Spend 30% of the 50 epochs ramping up
        div_factor=10,
        final_div_factor=100,
    )
    if "schedular" in state_dict:
        schedular.load_state_dict(state_dict["schedular"])

    checkpoint_folder = cfg_path.parent / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True)
    train_losses = []
    best_val_loss = state_dict.get("val_loss_avg", float("inf"))
    start_epoch = state_dict.get("epoch", 0)
    # Training
    print("Start training")
    if start_epoch > 0:
        print(f"Resume from epoch: {start_epoch}| best_val_loss: {best_val_loss:.4f}")

    for epoch in range(start_epoch, cfg["epochs"]):
        avg_val_loss = run_epoch(
            epoch, loaders["train"], model, loss_fn, optimizer, schedular
        )

        # 3. Save Intermediate Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "schedular": schedular.state_dict(),
            },
            checkpoint_folder / "model_last.pth",
        )
        train_losses.append(avg_val_loss)

        # 2. Validation/Evaluation Phase
        model.eval()
        torch.cuda.empty_cache()
        valid_l_all = 0.0
        valid_l_feat = 0.0
        valid_l_obj = 0.0
        assert len(loaders["valid"]) > 0, "No validation found"
        with torch.no_grad():
            z1, z2 = None, None
            for meta, y1, y2 in loaders["valid"]:
                meta, y1, y2 = meta.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE)
                # [Batch, n_obj_max, Z, Y, X]
                Z, Y, X = y1.shape[-3:]
                # [Batch, n_obj_max, Z, Y, X] => [batch*n_obj_max, 1, Z, Y, X]
                y1 = y1.view(-1, 1, Z, Y, X)
                y2 = y2.view(-1, 1, Z, Y, X)
                # [Batch, n_obj_max, 6] => [Batch * n_obj_max, 6]
                t_indice_flat = meta[:, :, 1].view(-1)

                mask = t_indice_flat >= 0

                y1_filtered = y1[mask]  # type: torch.Tensor
                y2_filtered = y2[mask]  # type: torch.Tensor

                z1 = model(y1_filtered)
                z2 = model(y2_filtered)

                loss, l_feat, l_obj = loss_fn(z1, z2)
                valid_l_all += loss.item()
                valid_l_feat += l_feat.item()
                valid_l_obj += l_obj.item()

            assert isinstance(z1, torch.Tensor) and isinstance(
                z2, torch.Tensor
            ), "Sanity test: z1 and z2 were existed if validation > 0"

            n_obj = z1.shape[0]

            z1_embed = model.backbone(y1_filtered)  # type: ignore
            z2_embed = model.backbone(y2_filtered)  # type: ignore

            z1_embed = z1_embed.view(n_obj, -1)  # type: torch.Tensor
            z2_embed = z2_embed.view(n_obj, -1)  # type: torch.Tensor
            z1_embed_f = (z1_embed - z1_embed.mean(0)) / (z1_embed.std(0) + loss_fn.eps)
            z2_embed_f = (z2_embed - z2_embed.mean(0)) / (z2_embed.std(0) + loss_fn.eps)

            embed_c_feat = torch.matmul(z1_embed_f.T, z2_embed_f) / z1.shape[0]
            embed_c_feat = embed_c_feat.cpu().detach().numpy()
            embed_c_obj = torch.matmul(z1_embed_f, z2_embed_f.T) / z1.shape[1]
            embed_c_obj = embed_c_obj.cpu().detach().numpy()
            # plot cross correlation to visualize

            # Object Space Correlation (N x N)
            z1_f = (z1 - z1.mean(0)) / (z1.std(0) + loss_fn.eps)
            z2_f = (z2 - z2.mean(0)) / (z2.std(0) + loss_fn.eps)

            proj_c_feat = torch.matmul(z1_f.T, z2_f) / z1.shape[0]
            proj_c_feat = proj_c_feat.cpu().detach().numpy()
            proj_c_obj = torch.matmul(z1_f, z2_f.T) / z1.shape[1]
            proj_c_obj = proj_c_obj.cpu().detach().numpy()

        n_feature_embed = embed_c_feat.shape[0]
        n_feature_proj = proj_c_feat.shape[0]

        fig = plt.figure(figsize=(8, 7), layout="constrained", frameon=False, dpi=600)
        plot_matrices(
            embed_c_feat,
            fig.add_subplot(221),
            f"Backbone: Feature Space ({n_feature_embed}x{n_feature_embed})",
        )
        plot_matrices(
            embed_c_obj,
            fig.add_subplot(222),
            f"Backbone: Object Space ({n_obj}x{n_obj})",
            show_colorbar=True,  # only show colorbar at upper right.
        )
        plot_matrices(
            proj_c_feat,
            fig.add_subplot(223),
            f"Projector: Feature Space ({n_feature_proj}x{n_feature_proj})",
        )
        plot_matrices(
            proj_c_obj,
            fig.add_subplot(224),
            f"Projector: Object Space ({n_obj}x{n_obj})",
        )
        fig.suptitle(f"Correlation Matrices (Epoch {epoch})")
        fig.savefig(checkpoint_folder / f"barlow_val_epoch_{epoch:0>3d}_valid.png")

        valid_l_all = valid_l_all / len(loaders["valid"])
        valid_l_feat = valid_l_feat / len(loaders["valid"])
        valid_l_obj = valid_l_obj / len(loaders["valid"])
        print(
            f"Epoch{epoch}:\nLoss:{valid_l_all}\nFeature Loss:{valid_l_feat}\nObject Loss:{valid_l_obj}\n"
        )

        if valid_l_all < best_val_loss:
            best_val_loss = valid_l_all
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "schedular": schedular.state_dict(),
                    "val_loss_avg": best_val_loss,
                },
                checkpoint_folder / "model_best.pth",
            )

            fig.savefig(checkpoint_folder / "barlow_val_best.png")

            print(
                f"Best model saved at epoch {epoch} with avg. val_loss: {best_val_loss:.4f}"
            )

        plt.close(fig)

    fig = plt.figure(layout="constrained", frameon=False)

    plot_loss(train_losses, ax=fig.add_subplot(111))
    fig.tight_layout()
    fig.savefig(checkpoint_folder / "training_loss_along_epoch.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
