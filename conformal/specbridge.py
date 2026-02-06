"""SpecBridge typed helpers"""

import pickle
from typing import TypedDict

import torch
import torch.nn.functional as F
from specbridge.data.massspecgym import MassSpecGymDataset, bin_peaks
from specbridge.models.mapper import DreamsToMolCondition
from torch import Tensor, nn
from torch.utils.data import Dataset


class MassSpectrum(TypedDict):
    """A mass spectrum in the MassSpecGym dataset.

    Args:
        mz: The mass-to-charge ratio of each peak
        intensity: peak intensities (normalized to max = 1)
        smiles: the molecule's ground truth smiles formula. It is also used as a key in the candidates dict
    """

    mz: Tensor
    intensity: Tensor
    title: str
    smiles: str
    meta: dict


def mass_spec_gym_dataset(
    path="Models/SpecBridge/data/SpecBridge_MassSpecGym_dataset.mgf",
) -> Dataset[MassSpectrum]:
    """Load the MassSpecGym dataset"""
    return MassSpecGymDataset(mgf_path=path, folds={"test"})


def mass_spec_gym_candidates(
    path="Models/SpecBridge/data/SpecBridge_MSGYM_candidates.pkl",
) -> dict[str, list[str]]:
    """Load a pickle file of molecule candidates"""
    return pickle.load(open(path, "rb"))


class SmilesPredictor(nn.Module):
    """A model that predicts SMILES from a spectra and multiple candidates.
    Internally, uses:
    - DreamsToMolCondition model for ??? we use only mu
    - ChemBERTA to embed candidate smiles
    """

    def __init__(self, model: DreamsToMolCondition, device: str):
        super().__init__()
        self.model = model.to(device)
        self.device = device

    @torch.no_grad
    def forward(
        self, spectrum: MassSpectrum, candidates: list[str], *, top_k=1
    ) -> list[tuple[str, float]]:
        """
        Predict SMILES from MS/MS spectrum (exactly like candidates.py and predict_smiles.py).

        Args:
            spectrum: molecule spectrum
            candidates: list of candidate SMILES strings for that spectrum
            top_k: number of top predictions to return

        Returns:
            List of (smiles, score) tuples
        """
        mz = spectrum["mz"].to(self.device)
        intensity = spectrum["intensity"].to(self.device)

        spec_binned = bin_peaks(mz, intensity, num_bins=2048, max_mz=2000.0).unsqueeze(
            0
        )
        peaks = torch.stack([mz, intensity], dim=-1).unsqueeze(0)

        # Add dummy SMILES to meta (required by model forward, but not used for query)
        meta = {"peaks": peaks, "smi_key": ["C"]}

        # Forward pass (exactly like specbridge/eval/candidates.py line 449)
        z_s, z_m, z_hat, mu, lv = self.model(spec_binned, meta, None, inference=True)

        # Use mu as query (like candidates.py line 474)
        z_query = mu  # [1, 768] - in ChemBERTa space

        # Get candidate embeddings (exactly like predict_smiles.py line 86)
        # (N, 768)
        candidates_embeds = self.model._chemberta_embed(candidates, self.device)

        # Compute similarities (exactly like candidates.py line 495-497)
        zq_norm = F.normalize(z_query, dim=-1)  # [1, 768]
        Z_norm = F.normalize(candidates_embeds, dim=-1)  # [N, 768]
        similarities = (zq_norm @ Z_norm.T).squeeze(0)  # [N]

        # Get top-k
        top_scores, top_indices = torch.topk(
            similarities, k=min(top_k, len(candidates))
        )

        return [
            (candidates[idx], score.item())
            for idx, score in zip(top_indices.cpu(), top_scores.cpu())
        ]
