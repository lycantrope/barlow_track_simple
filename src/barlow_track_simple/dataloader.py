from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, random_split

from barlow_track_simple.dataset import (
    HDFSequence,
    HDFStack,
    NeuronAugmentedImagePairDataset,
    Stack,
    TiffStack,
    ZarrSequence,
    ZarrStack,
)


def verify_input_data(
    datapaths: Sequence[Path],
    csvpaths: Sequence[Path],
    channel: str = "",
    hdf_key: str = "",
) -> List[Tuple[Stack, Path]]:
    assert len(datapaths) == len(csvpaths), "Image size is not same as csvpaths"
    # Verify all suffix is the same
    suffix_set = set(p.suffix for p in datapaths)

    assert len(suffix_set) == 1, f"{suffix_set}"
    suffix = suffix_set.pop()

    if suffix == ".tif":
        stack_iter = (TiffStack(p) for p in datapaths)
    elif suffix == ".h5":
        if channel:
            # Reading the hdf file like ASCENT format: t0/c0, t0/c1, t1/c0, ..., tn/c0, tn/c1
            stack_iter = (HDFSequence(p, channel=channel) for p in datapaths)
        elif hdf_key:
            # One dataset with axes of (T,Z,Y,X)
            stack_iter = (HDFStack(p, hdf_key) for p in datapaths)
        else:
            raise ValueError("No channel or hdf_key provided.")
    elif datapaths[0].name.endswith((".zarr", ".zarr.zip")):
        if channel:
            stack_iter = (ZarrSequence(p, channel=channel) for p in datapaths)
        else:
            stack_iter = (ZarrStack(p) for p in datapaths)
    else:
        raise TypeError(f"Only support TIFF, HDF5 and Zarr: {suffix}")

    return list(zip(stack_iter, csvpaths))


def get_train_loader(
    dataset_list: Sequence[Tuple[Stack, Path]],
    target_sz: Tuple[int, int, int],
    batch_size: int = 64,
    val_fraction: float = 0.1,
    transforms_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, DataLoader]:

    assert val_fraction > 0.0 and val_fraction < 1.0, "val_fraction >= 1, <=0"
    train_fraction = 1.0 - val_fraction

    dset = NeuronAugmentedImagePairDataset.from_stack_list(
        dataset_list,
        target_sz,
        transform_args=transforms_params,
    )
    train_dset, val_dset = random_split(
        dset,
        lengths=(train_fraction, val_fraction),
    )

    return {
        "train": DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            num_workers=min(torch.get_num_threads(), 8),
        ),
        "valid": DataLoader(
            val_dset,
            batch_size=1,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            num_workers=min(torch.get_num_threads(), 8),
        ),
    }
