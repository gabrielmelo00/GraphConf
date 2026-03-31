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
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import h5py
import numpy as np
import numpy.typing as npt
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


def load_dataset(
    split: Literal["train", "test", "val"], **kwargs
) -> tuple[DataLoader[Batch], dict[str, str]]:
    """Load the given split of the NPLIB1 dataset.
    Also returns a map from mass spectra identifier to
    candidate molecules formula"""
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

    spectra_to_formula = {
        i.get_spec_name(): i.get_spectra_formula() for i in dataset.get_spectra_list()
    }

    return loader, spectra_to_formula


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
def embed_spectra_loader(
    loader: DataLoader[Batch], model: ContrastiveModel, device: str
) -> SpectraEmbeddings:
    """Compute spectra embeddings from a spectra dataloader"""

    names = []
    embeds = []

    model.eval()
    model.to(device)

    for batch in tqdm(loader):
        batch = batch_to(batch, device)
        _, out = model.encode_spectra(batch)
        embeds.append(out["contrast"].detach().cpu())
        names.extend(batch["names"])

    return {"names": names, "embeds": torch.cat(embeds, 0).numpy()}


class Isomers(TypedDict):
    ikeys: npt.NDArray[np.str_]
    smiles: npt.NDArray[np.str_]
    fps: npt.NDArray[np.uint8]


def isomers(hdf: h5py.File, formula: str) -> Isomers:
    """Get the isomer molecules for given formula"""
    indices = np.where(np.array(hdf["formulae"]).astype(str) == formula)[0]

    if not len(indices):
        return {
            "ikeys": np.array([]),
            "smiles": np.array([]),
            "fps": np.array([]),
        }

    offset = hdf["formula_offset"][indices[0]]
    length = hdf["formula_lengths"][indices[0]]
    sl = slice(offset, offset + length)

    return {
        "ikeys": hdf["ikeys"][sl],
        "smiles": hdf["smiles"][sl],
        "fps": np.unpackbits(hdf["fingerprints"][sl], axis=-1)[
            ..., -hdf.attrs["num_bits"] :
        ],
    }


@torch.no_grad()
def embed_candidates(
    isomers: Isomers, model: ContrastiveModel, device: str
) -> np.ndarray:
    loader = DataLoader(isomers["fps"], batch_size=128, shuffle=False)  # type: ignore

    embeds = []

    for batch in loader:
        _, out = model.encode_mol({"mols": batch.to(device)})
        embeds.append(out["contrast"].detach().cpu().numpy())

    return np.vstack(embeds)


@dataclass
class CandidateSimilarities:
    similarities: np.ndarray
    truth: int

    def normalized_truth_rank(self) -> float:
        """Closer to 1 is better"""
        rank = np.argsort(self.similarities)[self.truth]
        return rank / len(self.similarities)

    def retained_size(self, threshold: float) -> int:
        return (self.similarities >= threshold).sum()

    @staticmethod
    def topk_accuracy(cands: list["CandidateSimilarities"], k: int):

        correct = 0

        for cand in cands:
            rank = (
                len(cand.similarities) - np.argsort(cand.similarities)[cand.truth] - 1
            )

            if rank < k:
                correct += 1

        return correct / len(cands)
