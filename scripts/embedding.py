import argparse
import os
import shutil
from pathlib import Path

import hdf5plugin
import numpy as np
import polars as pl
import torch
from ruamel.yaml import YAML, scalarfloat
from tqdm.auto import tqdm

from barlow_track_simple.config import get_device
from barlow_track_simple.dataloader import verify_input_data
from barlow_track_simple.dataset import ImageDataset
from barlow_track_simple.model import BarlowTwinsEmbed3D

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

# This will tell pytorch load scalarfloat safely
torch.serialization.add_safe_globals([scalarfloat.ScalarFloat])


DEVICE = get_device()

print(f"Use device: {DEVICE}")

yaml = YAML()


def embed_using_barlow(
    model: BarlowTwinsEmbed3D,
    dataset: ImageDataset,
    use_projection_space: bool = False,
    batch_size=64,
):
    """
    Use a trained model to project a dataset into the latent space

    use_projection_space - if True, uses the post-projection head space; most SSL approaches discard the projector (i.e. set this to False)
    """
    # names = dataset.which_neurons
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        num_workers=torch.get_num_threads(),
        persistent_workers=False,
    )
    with torch.no_grad():
        for centroids, batch in loader:
            if isinstance(batch, np.ndarray):
                batch = torch.from_numpy(batch)

            batch = batch.to(DEVICE, non_blocking=True)
            if use_projection_space:
                embeds = model(batch)
            else:
                embeds = model.backbone(batch)
            yield np.asarray(centroids), embeds.cpu().detach().numpy().reshape(
                batch.size(0), -1
            )


def run_embedding():
    parser = argparse.ArgumentParser()

    parser.add_argument("--cfg_path", type=str, required=True)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
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
    if pretrained_model_path is not None:
        pretrained_model_path = Path(pretrained_model_path)
        if not pretrained_model_path.is_absolute():
            # Relative to config.yaml
            pretrained_model_path = cfg_path.parent / pretrained_model_path
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        print(f"Model weights loaded: {pretrained_model_path}")
    model = BarlowTwinsEmbed3D.init_model(
        cfg["projector"],
        crop_sz,
        cfg["backbone_type"],
        cfg.get("projector_final"),
    )
    assert state_dict, "No state_dict found. Please make sure your setting is corrects"

    model.load_state_dict(state_dict.get("model_state_dict", state_dict))

    model.to(DEVICE)
    model.eval()
    datasets = ImageDataset.load_all_volumes(dataset_list, target_sz=crop_sz)

    print("Start embedding")

    for dataset in tqdm(datasets, total=len(datasets)):

        outputdir = dataset.filepath.parent
        tmp_dir = outputdir / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(os.fspath(tmp_dir))
        tmp_dir.mkdir(exist_ok=True)

        for i, (centroids, embed) in enumerate(
            embed_using_barlow(
                model,
                dataset,
                args.use_projection_space,
                args.batch_size,  # We using batch_size from the CLI imput
            )
        ):

            df_meta = pl.from_numpy(
                centroids, schema=["object_id", "t", "z", "y", "x", "pixel_value"]
            )

            df_meta.with_columns(
                pl.Series("embedding", embed, dtype=pl.List(pl.Float32))
            ).write_parquet(tmp_dir / f"parts.{i:05d}.parquet")

        name = dataset.filepath.stem
        outputpath = outputdir / (name + "_embed.parquet")
        print(f"Save final parquet to: {outputpath}")
        pl.scan_parquet(tmp_dir / "*.parquet").sink_parquet(
            outputpath,
            compression="zstd",
            compression_level=5,
        )

        shutil.rmtree(os.fspath(tmp_dir))

    print("Finished all embedding")


if __name__ == "__main__":
    run_embedding()
