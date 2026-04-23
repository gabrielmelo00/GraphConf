"""
JESTR1 utils
"""

from pathlib import Path

import dgl
import numpy as np
import torch
from mist.data.data import Spectra
from rdkit import Chem
from ruamel.yaml import YAML
from torch import Tensor
from torch.utils.data import DataLoader

from JESTR1.dataset import mol_to_graph
from JESTR1.models import INTER_MLP2, MolEnc, SpecEncMLP_BIN
from JESTR1.utils import get_ms_array_batch


def load_models(
    path: str = "./JESTR1",
) -> tuple[tuple[MolEnc, SpecEncMLP_BIN, INTER_MLP2], dict]:
    """
    Args:
        path: path to the JESTR1 folder
    """
    root = Path(path)
    yaml = YAML()
    params = yaml.load(open(root / "params.yaml", "r"))

    root = root / "data" / "NPLIB1"
    inter = torch.load(
        root / "pretrained_inter_model_1707829192911_best.pt", map_location="cpu"
    )
    mol_enc = torch.load(
        root / "pretrained_mol_enc_model_1707829192911_best.pt", map_location="cpu"
    )
    spec_enc = torch.load(
        root / "pretrained_spec_enc_model_1707829192911_best.pt", map_location="cpu"
    )

    mol_enc_model = MolEnc(params, 74)
    mol_enc_model.load_state_dict(mol_enc)

    spec_enc_model = SpecEncMLP_BIN(params, 1000)
    spec_enc_model.load_state_dict(spec_enc)

    inter_model = INTER_MLP2(params)
    inter_model.load_state_dict(inter)

    return (mol_enc_model, spec_enc_model, inter_model), params


def process_spectra(spectra: Spectra) -> np.ndarray:
    """Process an NPLIB1 spectra into the format JESTR1 expects

    - intensities are normalized such that the max is 999
    - frequencies above 1000 are removed
    - intensity and mz columns are swapped
    """
    assert spectra.spectra is not None
    spectra_np: np.ndarray = spectra.spectra[0]

    mz = spectra_np[:, 0]
    intensity = spectra_np[:, 1]
    intensity *= 999 / intensity.max()

    spectra_np = np.column_stack((intensity, mz))

    return spectra_np[mz <= 1000, :]


def batch_spectras(spectras: list[Spectra], **params) -> Tensor:
    """Bin and batch a ground truth's spectra list into a tensor.
    Each ground truth has few spectra, so we don't need a dataloader.

    Args:
        spectras: MIST parsed spectras
        params: JESTR1 yaml params dict

    Returns: (batch_size, 1000) binned tensor
    """

    # 1. Preprocess the spectra
    spectras_np: list[np.ndarray] = [process_spectra(spectra) for spectra in spectras]

    # 2. Bin and batch
    mz_b, *_ = get_ms_array_batch(
        spectras_np, "log10over3", params["max_mz"], params["resolution"]
    )

    return mz_b


def candidates_loader(
    candidates: list[str], params, **kwargs
) -> tuple[DataLoader[dgl.DGLHeteroGraph], list[str], list[bool]]:
    """Create a dataloader of batched graphs from a list of candidate smiles.

    Because the graph creation might fail, we need to precompute all graphs,
    and then return the list of smiles that correspond to it in order to find
    the correct embedding of the ground truth.

    Args:
        candidates: candidate smiles
        params: JESTR1 yaml params dict
        kwargs: DataLoader kwargs (batch size, num_workers...)
    """

    graphs: list[dgl.DGLHeteroGraph] = []
    smiles: list[str] = []
    mask: list[bool] = []

    for candidate in candidates:
        mol = Chem.MolFromSmiles(candidate)
        if mol is None:
            mask.append(False)
            continue
        cache = {}  # bypass builtin cache mechanism to avoid RAM explosion
        if not mol_to_graph(mol, "dummy", cache, params, "cpu"):
            mask.append(False)
            continue
        graphs.append(cache["dummy"])
        smiles.append(candidate)
        mask.append(True)

    def collate_fn(graphs: list[dgl.DGLHeteroGraph]):
        return dgl.batch(graphs)

    return (
        DataLoader(
            graphs,  # type: ignore
            collate_fn=collate_fn,
            **kwargs,
        ),
        smiles,
        mask,
    )
