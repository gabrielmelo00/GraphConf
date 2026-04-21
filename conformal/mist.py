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
from typing import Any

import numpy as np
import rdkit.Chem.rdFingerprintGenerator
import torch
from mist.data.data import Spectra
from mist.data.featurizers import PeakFormulaTest, SpecFeaturizer
from mist.models.contrastive_model import ContrastiveModel
from mist.models.mist_model import MistNet
from rdkit import Chem
from torch.utils.data import DataLoader

_generator = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def fingerprint(smile: str) -> np.ndarray:
    """Fingerprint a molecule to a (4096,) one-hot numpy array"""

    mol = Chem.MolFromSmiles(smile)
    assert mol is not None, "Bad smiles"

    fp = _generator.GetFingerprint(mol)
    array = np.zeros((0,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, array)

    return array


def load_fingerprint() -> tuple[MistNet, Any]:
    """Load the spectra fingerprint MIST model and its kwargs"""

    path = "quickstart/pretrained_models/mist_fp_canopus_pretrain.ckpt"
    checkpoints = torch.load(path, map_location="cpu")

    hyperparams = checkpoints["hyper_parameters"]

    # Load the model
    model = MistNet(**hyperparams)
    model.load_state_dict(checkpoints["state_dict"])

    kwargs = hyperparams | {
        "spec_features": model.spec_features(mode="test"),
        "mol_features": "none",
        "allow_none_smiles": True,
    }

    return model, kwargs


def load_contrastive() -> tuple[ContrastiveModel, Any]:
    """Load the contrastive embedding MIST model and its kwargs"""

    path = "quickstart/pretrained_models/mist_contrastive_canopus_pretrain.ckpt"
    checkpoints = torch.load(path, map_location="cpu")

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


def spectra_loader(spectra: list[Spectra], featurizer: SpecFeaturizer, **kwargs):
    """Given a spectra featurizer, produce a dataloader of batched spectra.
    Works for both contrastive and fingerprint prediction

    Args:
        spectra: list of spectra objects
        featurizer: spectra featurizer
        kwargs: dataloader kwargs (batch_size, num_workers...)
    """

    return DataLoader(
        [featurizer._featurize(spectrum) for spectrum in spectra],  # type: ignore
        collate_fn=PeakFormulaTest.collate_fn,
        **kwargs,
    )


def smiles_loader(smiles: list[str], **kwargs):
    """Given smiles, returns a dataloader over batched fingerprints."""

    def collate_fn(smiles: list[str]):
        fingerprints = [fingerprint(smile) for smile in smiles]
        return torch.tensor(np.stack(fingerprints), dtype=torch.float32)

    return DataLoader(smiles, collate_fn=collate_fn, **kwargs)  # type: ignore


def dict_to(batch: dict, device: str) -> dict:
    """Move the contents of a dict to a device"""
    return {
        k: v.to(device=device, non_blocking=True) if hasattr(v, "to") else v
        for k, v in batch.items()
    }
