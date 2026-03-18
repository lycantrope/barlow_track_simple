import argparse
import os
from pathlib import Path

import hdf5plugin
import polars as pl
import torch
from ruamel.yaml import YAML
from tqdm.auto import tqdm

from barlow_track_simple.config import get_device
from barlow_track_simple.dataloader import verify_input_data
from barlow_track_simple.dataset import ImageDataset
from barlow_track_simple.model import BarlowTwinsEmbed3D

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH

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
    device = next(model.parameters()).device

    for t_idx in dataset.t_indice:
        for centroids, batch in dataset.batched_iter_patches_at(
            t_idx,
            batch_size=batch_size,
        ):
            with torch.no_grad():
                batch = torch.from_numpy(batch).to(device)
                if use_projection_space:
                    embeddings = model(batch)
                else:
                    embeddings = model.backbone(batch)
                embeddings = embeddings.cpu().detach().numpy()
            N = batch.shape[0]
            yield centroids, batch.reshape(N, -1)


def run_embedding():
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
    if pretrained_model_path is not None:
        pretrained_model_path = Path(pretrained_model_path)
        if not pretrained_model_path.is_absolute():
            pretrained_model_path = pretrained_model_path.relative_to(cfg_path.parent)
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        print(f"Model weights loaded: {pretrained_model_path}")

    model = BarlowTwinsEmbed3D.init_model(
        cfg["projector"],
        crop_sz,
        cfg["backbone_type"],
        cfg.get("projector_final"),
    )

    if state_dict:
        model.load_state_dict(state_dict.get("model_state_dict", state_dict))

    model.to(DEVICE)

    datasets = ImageDataset.load_all_volumes(dataset_list, target_sz=crop_sz)

    print("Start embedding")

    for dataset in tqdm(datasets, total=len(datasets)):
        name = dataset.filepath.stem
        outputpath = dataset.filepath.with_name(name + "_embed.parquet")
        all_chunks = []
        for centroids, embed in embed_using_barlow(
            model,
            dataset,
            args.use_projection_space,
            args.batch_size,
        ):

            df_meta = pl.from_numpy(
                centroids, schema=["object_id", "t", "z", "y", "x", "pixel_value"]
            )

            df_t = df_meta.with_columns(
                pl.Series("embedding", embed, dtype=pl.List(pl.Float32))
            )
            all_chunks.append(df_t)

        final_df = pl.concat(all_chunks)
        final_df.write_parquet(outputpath)
        del all_chunks, final_df

    print("Finished all embedding")


if __name__ == "__main__":
    run_embedding()
