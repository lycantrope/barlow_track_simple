import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(slots=True)
class EmbedDataset:

    dataset: pl.LazyFrame = field(repr=False)

    centroids: pl.DataFrame = field(init=False, repr=False)

    n_object: int = field(init=False)
    n_frame: int = field(init=False)
    embed_dim: int = field(init=False)

    def __post_init__(self):
        valid_keys = {"object_id", "t", "z", "y", "x", "peak_values", "embedding"}
        dataset_keys = set(self.dataset.collect_schema().names())
        assert not any(
            dataset_keys ^ valid_keys
        ), f"Invalid keys found in: {dataset_keys}"
        self.dataset = self.dataset.set_sorted("object_id")
        self.centroids = self.dataset.select(
            ["object_id", "t", "z", "y", "x", "peak_values"]
        ).collect()

        schema = self.dataset.select(
            [
                pl.len().alias("n_object"),
                pl.col("t").n_unique().alias("n_frame"),
                pl.col("embedding").head(1).arr.len().alias("embed_dim"),
            ]
        ).collect()
        self.n_object = schema["n_object"].item()
        self.n_frame = schema["n_frame"].item()
        self.embed_dim = schema["embed_dim"].item()

    def get_embed_at(self, t_indices: Sequence[int]) -> np.ndarray:
        return (
            self.dataset.filter(pl.col("t").is_in(t_indices))
            .select("embedding")
            .collect()
            .get_column("embedding")
            .to_numpy()
        )

    def get_object_id_at(self, t_indices: Sequence[int]):
        return (
            self.centroids.filter(pl.col("t").is_in(t_indices))
            .get_column("object_id")
            .to_numpy()
        )

    @classmethod
    def from_parquet(cls, parquet_file: Path) -> "BarlowEmbedDataset":
        return cls(pl.scan_parquet(source=parquet_file))


# This is only for visualization
def umap_projection(embed, n_components=10, n_neighbors=10, **opt_umap):
    """
    Reduces high-dim embeddings (2048) to low-dim (e.g., 10) for
    stable visualization or faster tracking.
    """
    try:
        import umap
    except ImportError as e:
        warnings.warn(
            "If you want to run umap_projection, install via: `pip install umap-learn`",
            UserWarning,
        )
        raise e

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        **opt_umap,
    )

    # Returns the actual low-dim coordinates (N, n_components)
    return reducer.fit_transform(embed)
