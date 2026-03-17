import argparse
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from barlow_track_simple.config import get_device
from barlow_track_simple.dataloader import get_train_loader, verify_input_data
from barlow_track_simple.model import (
    LARS,
    BarlowTwinsDualLoss,
    BarlowTwinsEmbed3D,
)
from barlow_track_simple.plot import plot_loss

yaml = YAML()

DEVICE = get_device()

print(f"Use device: {DEVICE}")


def run_epoch(
    epoch: int,
    loader: torch.utils.data.DataLoader,
    model: Union[BarlowTwinsEmbed3D, torch.nn.DataParallel],
    loss_fn: BarlowTwinsDualLoss,
    optimizer: LARS,
):
    model.train()
    epoch_loss = 0.0
    valid_batch_count = 0
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
    for batch_idx, (meta, x1, x2) in pbar:
        optimizer.zero_grad()
        meta, x1, x2 = meta.to(DEVICE), x1.to(DEVICE), x2.to(DEVICE)

        # [Batch, n_obj_max, Z, Y, X]
        Z, Y, X = x1.shape[-3:]
        # [Batch, n_obj_max, Z, Y, X] => [batch*n_obj_max, 1, Z, Y, X]
        x1 = x1.view(-1, 1, Z, Y, X)
        x2 = x2.view(-1, 1, Z, Y, X)
        # [Batch, n_obj_max, 6] => [Batch * n_obj_max, 6]
        t_indice_flat = meta[:, :, 1].view(-1)

        mask = t_indice_flat >= 0

        valid_t_indice = t_indice_flat[mask]
        unique_times = valid_t_indice.unique()

        x1_filtered = x1[mask]  # type: torch.Tensor
        x2_filtered = x2[mask]  # type: torch.Tensor
        x1_filtered = x1_filtered.unsqueeze(1)
        x2_filtered = x2_filtered.unsqueeze(1)

        total_loss = torch.tensor(0.0, device=DEVICE)
        valid_loss_count = 0

        # Forward pass
        z1 = model(x1_filtered)
        z2 = model(x2_filtered)

        for t in unique_times:
            # Filter features belonging only to time 't'
            t_mask = valid_t_indice == t
            z1_t = z1[t_mask]
            z2_t = z2[t_mask]
            if z1_t.size(0) < 2:
                continue

            # Check the mean of the standard deviations
            epoch_variances = torch.std(z1_t, dim=0)
            batch_std = epoch_variances.mean().item()
            if batch_std < 1e-6:
                print(
                    f"Epoch {epoch} at Batch {batch_idx}-{t} mean feature variance: {batch_std:.6f}"
                )
                continue

            loss, _, _ = loss_fn(z1_t, z2_t)

            total_loss += loss
            valid_loss_count += 1
            epoch_loss += loss.item()

        valid_batch_count += valid_loss_count
        # Backward pass
        if valid_loss_count > 0:
            (total_loss / valid_loss_count).backward()
            optimizer.step()

        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

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

    model = BarlowTwinsEmbed3D.load_model(
        cfg["projector"],
        crop_sz,
        cfg["backbone_type"],
        cfg.get("projector_final"),
        cfg.get("pretrained_model_path"),
    )

    model.to(DEVICE)

    # 2. Wrap the model
    if torch.cuda.device_count() > 1:
        print(f"Use {torch.cuda.device_count()} GPUs!")
        # This automatically splits the batch and replicates the model
        model = torch.nn.DataParallel(model)

    loss_fn = BarlowTwinsDualLoss(
        lambd=cfg["lambd"],
        lambd_obj=cfg["lambd_obj"],
    )

    optimizer = LARS(
        params=model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    loaders = get_train_loader(
        dataset_list=dataset_list,
        target_sz=crop_sz,
        batch_size=cfg.get("batch_size", 64),
    )

    checkpoint_folder = cfg_path.parent / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True)
    train_losses = []
    best_val_loss = float("inf")
    # Training
    print("Start training")
    for epoch in range(cfg["epochs"]):
        avg_val_loss = run_epoch(epoch, loaders["train"], model, loss_fn, optimizer)

        # 3. Save Intermediate Checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_folder / "model_last.pt",
        )
        train_losses.append(avg_val_loss)

        # 2. Validation/Evaluation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for _, x1, x2 in loaders["valid"]:
                x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

                z1 = model(x1)
                z2 = model(x2)
                loss, _, _ = loss_fn(z1, z2)
                val_loss += loss.item()
                print(f"Epoch {epoch}: Val Loss: {val_loss/len(loaders['valid'])}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                checkpoint_folder / "model_best.pt",
            )
            print(
                f"Best model saved at epoch {epoch} with val_loss: {best_val_loss:.4f}"
            )

    fig = plt.figure()

    plot_loss(train_losses, ax=fig.add_subplot(111))
    fig.tight_layout()
    fig.savefig(checkpoint_folder / "training_loss.png")


if __name__ == "__main__":
    main()
