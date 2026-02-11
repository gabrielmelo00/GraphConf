"""Quantile estimators"""

from typing import Literal

import numpy as np
import torch
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


class FixedRegressor:
    """Fixed conformal quantile estimator (for standard conformal prediction)"""

    def __init__(self, target=0.9):
        self.target = target
        self._threshold: float | None = None

    def fit(self, distances: list[float]):
        """Fit the regressor on a calibration set."""
        n = len(distances)
        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self._threshold = np.quantile(distances, conformal_quantile)

    def threshold(self) -> float:
        """Returns the fixed non-conformity threshold"""
        assert self._threshold is not None, "Regressor must be fitted before predicting"

        return self._threshold


class CandidateSizeRegressor:
    """Conformal quantile estimator that uses candidate set size as
    an indicator of task uncertainty"""

    regressor: QuantileRegressor | RandomForestQuantileRegressor

    def __init__(
        self,
        target=0.9,
        kind: Literal["linear", "random_forest"] = "linear",
        **kwargs,
    ):
        """Initialize the regressor.

        Args:
            target: the target coverage level (e.g. 0.9 for 90% coverage)
            kind: the type of regressor to use
            kwargs: additional init args for the regressor
        """

        self.target = target
        self._threshold: float | None = None
        match kind:
            case "linear":
                self.regressor = QuantileRegressor(quantile=target, alpha=0.0, **kwargs)
            case "random_forest":
                self.regressor = RandomForestQuantileRegressor(
                    default_quantiles=[target], **kwargs
                )

    def fit(self, distances: list[float], candidate_sizes: list[int]):
        """Fit the regressor on a calibration set."""
        np_distances = np.array(distances)
        np_candidate_sizes = np.array(candidate_sizes).reshape(-1, 1)

        # Fit the quantile regressor
        self.regressor.fit(np_candidate_sizes, np_distances)

        # Estimate the quantile of the residuals
        residuals = np_distances - self.regressor.predict(np_candidate_sizes)
        n = len(residuals)
        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self._threshold = np.quantile(residuals, conformal_quantile)

    def threshold(self, candidate_size: int) -> float:
        """Predict the non-conformity threshold for a given candidate set size."""
        assert self.regressor is not None, "Regressor must be fitted before predicting"
        assert self._threshold is not None, "Regressor must be fitted before predicting"

        np_candidate_size = np.array([[candidate_size]])
        return self.regressor.predict(np_candidate_size)[0] + self._threshold


class EmbeddingRegressor:
    """Conformal quantile estimator that uses embeddings as
    an indicator of task uncertainty"""

    def __init__(
        self,
        target=0.9,
        embed_dim=768,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the regressor.

        Args:
            target: the target coverage level (e.g. 0.9 for 90% coverage)
            embed_dim: embedding dimension
            device: cpu or gpu
        """

        self.target = target
        self.embed_dim = embed_dim
        self.device = device
        self._threshold: float | None = None

        hidden = embed_dim // 2

        self.model = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        ).to(device)

    def fit(self, distances: list[float], embeddings: Tensor):
        """Fit the regressor on a calibration set.

        Args:
            distances: list of non-conformity scores
            embeddings: (n_samples, embed_dim) tensor of embeddings
        """
        assert len(distances) == len(embeddings), (
            f"Mismatch between amount of distances and embeddings: {len(distances)} vs {len(embeddings)}"
        )
        assert len(embeddings[0]) == self.embed_dim, (
            f"Embedding dimension mismatch: expected {self.embed_dim}, got {len(embeddings)}"
        )

        self.model.train()
        patience = 3  # stop training after 3 epochs without improvement
        remaining = patience
        delta = 1e-4  # minimum considered improvement
        prev_loss = 1e10
        batch_size = 32
        epoch = 1

        dataset = TensorDataset(embeddings, torch.tensor(distances))
        loader = DataLoader(dataset, batch_size, shuffle=True)
        optimizer = Adam(self.model.parameters())

        # Fit the quantile regressor
        while remaining > 0:
            print(f"Epoch {epoch}", end="")
            total = 0.0

            for embeds, dists in loader:
                embeds = embeds.to(self.device)
                dists = dists.to(self.device)

                preds = self.model(embeds).squeeze()
                loss = pinball_loss(preds, dists, self.target)
                total += loss.item()

                self.model.zero_grad()
                loss.backward()
                optimizer.step()

            epoch += 1
            total /= len(loader)
            print(f" - Average loss: {total:.4f}")

            # Early stopping logic
            if total < prev_loss - delta:
                remaining = patience
                prev_loss = total
            else:
                remaining -= 1

        # Estimate the quantile of the residuals
        self.model.eval()
        preds: list[float] = []

        for embeds, _ in loader:
            embeds = embeds.to(self.device)
            with torch.no_grad():
                pred = self.model(embeds).squeeze().cpu()
            preds.extend(pred)

        residuals = np.array(distances) - np.array(preds)
        n = len(residuals)
        conformal_quantile = np.ceil((n + 1) * 0.9) / n
        self._threshold = np.quantile(residuals, conformal_quantile)

    def threshold(self, embedding: Tensor) -> float:
        """Predict the non-conformity threshold for a given embedding."""
        assert self._threshold is not None, "Regressor must be fitted before predicting"

        embedding = embedding.to(self.device)

        return self.model(embedding.unsqueeze(0)).item() + self._threshold


def pinball_loss(preds: Tensor, targets: Tensor, quantile: float):
    """Pinball loss for quantile regression."""
    residuals = targets - preds
    loss = torch.max((quantile - 1) * residuals, quantile * residuals)
    return loss.mean()
