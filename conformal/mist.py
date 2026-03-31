"""
MIST & NPLIB1 helpers

Don't forget to cd to "mist" before running these functions,
there are many hardcoded paths.

```python
import os
os.chdir("mist")
```
"""

from copy import deepcopy
from typing import Any, Literal, TypedDict

import numpy as np
import polars as pl
import torch
from mist.data.data import Mol, Spectra
from mist.models.contrastive_model import ContrastiveModel
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from mist.data import datasets, featurizers


def load_model() -> tuple[ContrastiveModel, Any]:
    """Load the contrastive embedding MIST model and its kwargs"""

    path = "quickstart/pretrained_models/mist_contrastive_canopus_pretrain.ckpt"
    checkpoints = torch.load(path)

    hyperparams = checkpoints["hyper_parameters"]

    # Load the model
    model = ContrastiveModel(**hyperparams)
    model.load_state_dict(checkpoints["state_dict"])

    base_model_hparams = deepcopy(hyperparams["base_model_hparams"])

    kwargs = (
        hyperparams
        | base_model_hparams
        | {
            "spec_features": model.main_model.spec_features(mode="test"),
            "mol_features": "none",
        }
    )

    return model, kwargs


class Batch(TypedDict):
    """
    NPLIB1 dataset batch

    Args:
        matched: bool, (batch_size)
        spec_indices: int, (batch_size)
        mol_indices: int, (batch_size)
        types: int, (batch_size, n)
        form_vec: float, (batch_size, n, ?)
        ion_vec: float, (batch_size, n)
        intens: float, (batch_size, n)
        names: mass spectra identifiers (CCMSLIB..., etc)
        num_peaks: int, (batch_size,)
        instruments: float, (batch_size,)
    """

    matched: Tensor
    spec_indices: Tensor
    mol_indices: Tensor
    types: Tensor
    form_vec: Tensor
    ion_vec: Tensor
    intens: Tensor
    names: list[str]
    num_peaks: Tensor
    instruments: Tensor


def load_dataset(split: Literal["train", "test", "val"], **kwargs) -> DataLoader[Batch]:
    """Load the given split of the NPLIB1 dataset"""
    featurizer = featurizers.get_paired_featurizer(**kwargs)

    print("Loading labels from", kwargs["labels_file"])
    print("Loading mass spectra from", kwargs["spec_folder"])

    # Load the molecules and mass spectra
    spectra_and_mols = datasets.get_paired_spectra(allow_none_smiles=True, **kwargs)
    spectra_mol_pairs: list[tuple[Spectra, Mol]] = list(zip(*spectra_and_mols))

    # Load the split
    splits = pl.read_csv(kwargs["split_file"], separator="\t")
    valid_names = splits.filter(pl.col("split") == split)["name"].to_list()

    spectra_mol_pairs = [
        (s, m) for s, m in spectra_mol_pairs if s.get_spec_name() in valid_names
    ]
    print("Loaded the", split, "split with", len(spectra_mol_pairs), "molecules")

    dataset = datasets.SpectraMolDataset(
        spectra_mol_list=spectra_mol_pairs, featurizer=featurizer, **kwargs
    )
    loader = datasets.SpecDataModule.get_paired_loader(
        dataset,
        batch_size=128,
        shuffle=False,
    )

    return loader


def batch_to(batch: Batch, device: str) -> dict:
    """Move the contents of a dict to a device"""
    return {  # type: ignore
        k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in batch.items()
    }


class SpectraEmbeddings(TypedDict):
    names: list[str]
    embeds: np.ndarray


@torch.no_grad()
def embed_loader(
    loader: DataLoader[Batch], model: ContrastiveModel, device: str
) -> SpectraEmbeddings:
    """Compute spectra embeddings from a spectra dataloader"""

    names = []
    embeds = []

    model.train()
    model.to(device)

    for batch in tqdm(loader):
        batch = batch_to(batch, device)
        _, out = model.encode_spectra(batch)
        embeds.append(out["contrast"].detach().cpu())
        names.extend(batch["names"])

    return {"names": names, "embeds": torch.cat(embeds, 0).numpy()}
