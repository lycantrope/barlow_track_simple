import abc
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import hdf5plugin
import numpy as np
import numpy.typing as npt
import tifffile
import torch

from .augmentation import Transform

os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGINS_PATH


class Stack(abc.ABC):
    @abc.abstractmethod
    def get_filepath(self) -> Path: ...
    @abc.abstractmethod
    def init(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def data(
        self,
    ) -> Union[np.memmap, np.ndarray, h5py.Dataset, Sequence[h5py.Dataset], None]: ...

    @property
    @abc.abstractmethod
    def shape(self) -> Sequence[int]: ...

    def __len__(self) -> int:
        return self.shape[0]

    def get_normalizer(self, p_min: float = 5.0, p_max: float = 100.0):
        if self.data is None:
            self.init()

        assert self.data is not None, "data must be accessed after init."
        # Estimate the pixel value range
        frame_idx = np.arange(self.shape[0])
        #  10% of frame will be selected (max: 8 , min: 1)
        n_selected = max(min(frame_idx.size // 10, 8), 1)
        selected_idx = np.random.choice(frame_idx, n_selected, replace=False)
        # Calculate the percentile values
        p_low, p_high = np.percentile(
            [self.data[i] for i in selected_idx],
            [p_min, p_max],
        )
        ptp = p_high - p_low + 1e-8

        def _normalizer(img: np.ndarray) -> np.ndarray:
            # Clip the values
            img = np.clip(img.astype("f4"), p_low, p_high)
            # Scale to [0, 1] range
            img = (img - p_low) / ptp
            return img

        return _normalizer

    def __getstate__(self):
        # This garantee everything can be pickled.
        if self.data is not None:
            self.close()
        return super().__getstate__()

    def __del__(self):
        self.close()


def __init__(self):
    assert self.data_path.suffix.lower().endswith(
        (".tif", ".tiff")
    ), "Current, I only supported ImageJ-Tiff (T, Z, Y, X) or (T, Z, X, Y)"


class TiffStack(Stack):
    def __init__(self, tif_path: os.PathLike):
        self.tif_path = Path(tif_path)
        assert self.tif_path.suffix.lower().endswith((".tif", ".tiff"))
        self._data = None

    def get_filepath(self) -> Path:
        return self.tif_path

    def init(self) -> None:
        if self._data is None:
            self._data = tifffile.memmap(self.tif_path, mode="r")

    def close(self) -> None:
        if self._data is not None:
            del self._data
            self._data = None

    @property
    def data(
        self,
    ) -> Optional[np.memmap]:
        self.init()
        return self._data

    @property
    def shape(self) -> Sequence[int]:
        self.init()
        assert self._data is not None
        return self._data.shape


class HDFStack(Stack):
    def __init__(self, hdfpath: os.PathLike, dataset_key: str):
        # This is for one dataset contains (T,Z,Y,X)
        self.hdfpath = Path(hdfpath)
        assert self.hdfpath.suffix.lower() == ".h5"

        self.dataset_key = dataset_key
        with h5py.File(self.hdfpath, mode="r") as handler:
            if self.dataset_key not in handler.keys():
                raise KeyError(f"Cannot found key: {self.dataset_key}")

        self._data = None

    def get_filepath(self) -> Path:
        return self.hdfpath

    def init(self) -> None:
        if self._data is None:
            self._data = h5py.File(self.hdfpath, mode="r")

    def close(self) -> None:
        if self._data is not None:
            try:
                self._data.close()
            except TypeError as _:
                # h5py\_hl\files.py", line 631, in close
                # Error in TypeError: bad operand type for unary ~: 'NoneType'
                pass
            self._data = None

    @property
    def data(
        self,
    ) -> Optional[h5py.Dataset]:
        self.init()
        assert self._data is not None
        dset = self._data[self.dataset_key]
        assert isinstance(dset, h5py.Dataset), "Must be group"
        return dset

    @property
    def shape(self) -> Sequence[int]:
        self.init()
        assert self._data is not None
        dset = self._data[self.dataset_key]
        assert isinstance(dset, h5py.Dataset), "Must be group"
        return dset.shape


class HDFSequence(Stack):
    def __init__(self, hdfpath: os.PathLike, channel="c0"):
        # This is for one dataset contains (T,Z,Y,X)
        self.hdfpath = Path(hdfpath)
        assert self.hdfpath.suffix.lower() == ".h5"
        with h5py.File(self.hdfpath, mode="r") as handler:
            # sorting the t0, ..., tn
            keys = (k for k in handler.keys() if str(k).startswith("t"))
            # [t0, t1, t2, t..., tn]
            keys = sorted(keys, key=lambda x: int(x[1:]))
            assert len(keys) > 0, "No image seqeunce found in HDF5 "

            dset = handler[keys[0]]
            assert isinstance(dset, h5py.Group), "Must be group"
            assert (
                isinstance(dset, h5py.Group) and channel in dset.keys()
            ), f"No channel found in dataset: {list(dset.keys())}"

        self.keys = [f"{k}/{channel}" for k in keys]
        self._data = None
        self._data_sequence = []

    def get_filepath(self) -> Path:
        return self.hdfpath

    def init(self) -> None:
        if self._data is None:
            self._data = h5py.File(self.hdfpath, mode="r")
            tmp_seq = [self._data[k] for k in self.keys]
            self._data_sequence = [
                dset for dset in tmp_seq if isinstance(dset, h5py.Dataset)
            ]

    def close(self) -> None:
        if self._data is not None:
            try:
                self._data.close()
            except TypeError as _:
                # h5py\_hl\files.py", line 631, in close
                # Error in TypeError: bad operand type for unary ~: 'NoneType'
                pass
            self._data = None
            self._data_sequence = []

    @property
    def data(
        self,
    ) -> Optional[Sequence[h5py.Dataset]]:
        self.init()
        return self._data_sequence

    @property
    def shape(self) -> Sequence[int]:
        self.init()
        assert self._data is not None
        T = len(self._data_sequence)
        return (T, *self._data_sequence[0].shape)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagestack: Stack,
        csv_path: Path,
        target_sz: Tuple[int, int, int] = (8, 64, 64),
    ):
        super().__init__()
        # In order to synchronize the normalization used
        self.imagestack = imagestack
        self.target_sz = target_sz
        assert csv_path.name.endswith(
            ".csv"
        ), "centroids_path be a csv file with follow headers [object_id,t,z,y,x,peak_values]"

        self.centroids = np.loadtxt(
            csv_path,
            skiprows=1,
            delimiter=",",
        ).astype(int)
        assert (
            self.centroids.ndim == 2 and self.centroids.shape[1] == 6
        ), "centroids_path be a csv file with follow headers [object_id,t,z,y,x,peak_values]"

        # We should allow other imread functinos to return the memmap like or array like object.
        assert (
            len(self.imagestack.shape) == 4
        ), "TiffImageDataset only supported ImageJ-Tiff (T, Z, Y, X) or (T, Z, Y, X)"

        # Sanity test
        boundary = np.array(self.imagestack.shape)
        centroids_max = self.centroids[:, 1:5].max(axis=0)

        assert all(
            (boundary - centroids_max) > 0
        ), f"All centroid should smaller than 4D images, {boundary} {centroids_max}"

        self.n_obj = self.centroids.shape[0]
        self.t_indice = sorted(np.unique(self.centroids[1]))
        self._normalize = self.imagestack.get_normalizer()

        # Since Dataloader will pickle the status to other thread, however, the memmap or hdf file is not picklable
        # We _normalizerose the files just after init, then, reopen it in sub workers.
        self.imagestack.close()

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.imagestack.data is not None, "Must be init before access"

        centroid = self.centroids[idx]

        _id, t, z, y, x, pix_val = centroid

        _, vol_z, vol_y, vol_x = self.imagestack.shape
        wz, wy, wx = self.target_sz

        z0, z1 = z - wz // 2, z + (wz + 1) // 2
        y0, y1 = y - wy // 2, y + (wy + 1) // 2
        x0, x1 = x - wx // 2, x + (wx + 1) // 2

        # 3. Calculate how much we are "out of bounds" for padding
        pad_z = (max(0, -z0), max(0, z1 - vol_z))
        pad_y = (max(0, -y0), max(0, y1 - vol_y))
        pad_x = (max(0, -x0), max(0, x1 - vol_x))

        z0_c, z1_c = np.clip((z0, z1), 0, vol_z)
        y0_c, y1_c = np.clip((y0, y1), 0, vol_y)
        x0_c, x1_c = np.clip((x0, x1), 0, vol_x)

        patch = self.imagestack.data[t][z0_c:z1_c, y0_c:y1_c, x0_c:x1_c]
        # 6. Apply padding if the patch was at the edge
        if any(sum(p) > 0 for p in [pad_z, pad_y, pad_x]):
            patch = np.pad(
                patch,
                (pad_z, pad_y, pad_x),
                mode="constant",
                constant_values=0,
            )

        # Sanity test: np.testing.assert_equal(patch.shape, self.target_sz)
        patch = patch[None, ...]
        return centroid, self._normalize(patch)

    def __len__(self):
        return self.n_obj

    def iter_patches_at(self, t_idx: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        assert self.imagestack.data is not None, "Must be init before access"

        if t_idx not in self.t_indice:
            raise IndexError(f"No {t_idx} found in dataset")

        centroids = self.centroids[self.centroids[:, 1] == t_idx]

        pad_width = [(sz // 2 + sz % 2 - 1, sz // 2) for sz in self.target_sz]
        data = np.pad(self.imagestack.data[t_idx], pad_width=pad_width)
        data = self._normalize(data)

        wz, wy, wx = self.target_sz
        for centroid in centroids:
            _, _, z, y, x, _ = centroid
            patch = data[z : z + wz, y : y + wy, x : x + wx]
            yield centroid, patch[None, ...]

        # # Following code is to get all patches at once
        # gz, gy, gx = np.mgrid[:wz, :wy, wx]
        # all_z = centroids[:, 2, None, None, None] + gz
        # all_y = centroids[:, 3, None, None, None] + gy
        # all_x = centroids[:, 4, None, None, None] + gx

        # return data[all_z, all_y, all_x]

    def batched_iter_patches_at(
        self, t_idx: int, batch_size: int = 32
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yields batches of patches instead of single patches."""
        batch = []
        for centroid, patch in self.iter_patches_at(t_idx):
            batch.append((centroid, patch))
            if len(batch) == batch_size:
                # Stack into a single tensor for the model
                centroids, patches = tuple(map(np.stack, zip(*batch)))
                yield centroids, patches
                batch = []

        # Yield the remaining patches if they don't fill a full batch
        if batch:
            # Stack into a single tensor for the model
            centroids, patches = tuple(map(np.stack, zip(*batch)))
            yield centroids, patches

    @classmethod
    def load_all_volumes(
        cls,
        data_list: Sequence[Tuple[Stack, Path]],
        target_sz: Tuple[int, int, int] = (8, 64, 64),
    ) -> List["ImageDataset"]:
        return [cls(stack, csv_path, target_sz) for (stack, csv_path) in data_list]

    @property
    def filepath(self) -> Path:
        return self.imagestack.get_filepath()


class NeuronAugmentedImagePairDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        imagestack: Stack,
        csv_path: Path,
        target_sz: Tuple[int, int, int] = (8, 64, 64),
        transform_args: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.base_dataset = ImageDataset(
            imagestack=imagestack, csv_path=csv_path, target_sz=target_sz
        )
        transform_args = transform_args or dict()
        self.augmentor = Transform(**transform_args)

    def __len__(self):
        # Simply delegate the length to the base
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, npt.ArrayLike, npt.ArrayLike]:
        centroid, y = self.base_dataset[idx]
        y1, y2 = self.augmentor(y)
        return centroid, y1, y2

    @classmethod
    def from_stack_list(
        cls,
        data_list: Sequence[Tuple[Stack, Path]],
        target_sz: Tuple[int, int, int] = (8, 64, 64),
        transform_args: Optional[Dict[str, Any]] = None,
    ) -> torch.utils.data.ConcatDataset:

        return torch.utils.data.ConcatDataset(
            [
                cls(stack, csv_path, target_sz, transform_args)
                for (stack, csv_path) in data_list
            ]
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    home = Path(r"D:\kuan\zeng-nwb\TOY21\251203-1DA\activity\W4")
    tif_path = home / "crop-W4-activity.h5"
    centroid_path = home / "crop-W4-activity_peaks.csv"

    dataset = NeuronAugmentedImagePairDataset(
        HDFSequence(tif_path, channel="c1"), centroid_path
    )
    loader = DataLoader(dataset, batch_size=64)
    # import matplotlib.pyplot as plt

    data = None
    for k, data in enumerate(loader):
        if k > 2:
            break
    if data is not None:
        d = data[0][0, 0].numpy()
        d1 = data[1][0, 0].numpy()
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(d.max(axis=0))
        ax = fig.add_subplot(122)
        ax.imshow(d1.max(axis=0))
        plt.show()
