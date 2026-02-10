import pickle
from typing import Iterable, Self

import numpy as np
from sklearn.linear_model import QuantileRegressor
from tqdm import tqdm

from conformal.fgw import FGW
from conformal.graph import Graph


class Metrics:
    def __init__(
        self,
        correct_coverage: list[bool],
        candidate_sizes: list[int],
        conformal_sizes: list[int],
    ):
        self.correct_coverage = np.array(correct_coverage)
        self.candidate_sizes = np.array(candidate_sizes)
        self.conformal_sizes = np.array(conformal_sizes)

    def save(self, path: str):
        """Save to a pickled file"""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        """Load from a pickled file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def coverage(self) -> float:
        return self.correct_coverage.mean()

    @property
    def mean_set_size(self) -> float:
        return self.conformal_sizes.mean()

    @property
    def median_set_size(self) -> float:
        return np.median(self.conformal_sizes)

    @property
    def mean_reduction(self) -> float:
        return 1 - (self.conformal_sizes / self.candidate_sizes).mean()

    @property
    def median_reduction(self) -> float:
        return 1 - np.median(self.conformal_sizes / self.candidate_sizes)

    @property
    def empty_rate(self) -> float:
        return (self.conformal_sizes == 0).mean()

    def __len__(self):
        return len(self.correct_coverage)


class ConformalPredictor:
    """Standard conformal prediction predictor."""

    def __init__(self, fgw: FGW, target=0.9):
        """
        Args:
            fgw: the distance function to use for nonconformity scores
            target: the target coverage level (e.g. 0.9 for 90% coverage)
        """
        self.fgw = fgw
        self.target = target
        self.threshold: float | None = None

    def fit(self, data: Iterable[tuple[Graph, Graph]], n: int | None = None):
        """Fit the conformal predictor on a calibration set.
        This basically computes a nonconformity threshold

        Args:
            data: iterable of (predicted graph, ground truth graph) pairs
            n: number of samples for tqdm progress display despite streaming
        """

        distances = []

        for g1, g2 in tqdm(data, "Fitting conformal predictor", total=n):
            distances.append(self.fgw(g1, g2))

        n = len(distances)
        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self.threshold = np.quantile(distances, conformal_quantile)

    def predict(
        self,
        data: Iterable[tuple[Graph, Graph, Iterable[Graph]]],
        n: int | None = None,
    ) -> Metrics:
        """Predict conformal sets and compute metrics on a test set.

        Args:
            data: iterable of (predicted graph, ground truth graph, candidate graphs) tuples
            n: number of samples for tqdm progress display despite streaming
        """

        assert self.threshold is not None, (
            "Conformal predictor must be fitted before prediction."
        )

        # Accumulate metrics
        correct_coverage: list[bool] = []
        candidate_sizes: list[int] = []
        conformal_sizes: list[int] = []

        for pred, truth, cands in tqdm(
            data,
            "Predicting with conformal predictor",
            total=n,
        ):
            candidate_size = 0
            conformal_size = 0

            # Compute conformal set size
            for candidate in cands:
                candidate_size += 1
                if self.fgw(candidate, pred) <= self.threshold:
                    conformal_size += 1

            # Check if ground truth is in conformal set
            correct_coverage.append(self.fgw(truth, pred) <= self.threshold)

            candidate_sizes.append(candidate_size)
            conformal_sizes.append(conformal_size)

        return Metrics(
            correct_coverage=correct_coverage,
            candidate_sizes=candidate_sizes,
            conformal_sizes=conformal_sizes,
        )


class CandidateSizeSCQR:
    """Score Conformal Quantile Regressor that uses candidate set size as an indicator of task uncertainty"""

    def __init__(self, fgw: FGW, target=0.9):
        """
        Args:
            fgw: the distance function to use for nonconformity scores
            target: the target coverage level (e.g. 0.9 for 90% coverage)
        """
        self.fgw = fgw
        self.target = target
        self.threshold: float | None = None
        self.regressor: QuantileRegressor | None = None

    def fit(self, data: Iterable[tuple[Graph, Graph, int]], n: int | None = None):
        """Fit the conformal predictor on a calibration set.
        This basically computes a nonconformity threshold,
        and then a quantile regression model to predict it from the candidate set size

        Args:
            data: iterable of (predicted graph, ground truth graph, candidate size) tuples
            n: number of samples for tqdm progress display despite streaming
        """

        distances = []
        set_sizes = []

        for g1, g2, size in tqdm(data, "Fitting conformal predictor", total=n):
            distances.append(self.fgw(g1, g2))
            set_sizes.append(size)

        np_distances = np.array(distances)
        np_set_sizes = np.array(set_sizes).reshape(-1, 1)

        regressor = QuantileRegressor(quantile=self.target, alpha=0)
        regressor.fit(np_set_sizes, np_distances)
        self.regressor = regressor

        residuals = np_distances - regressor.predict(np_set_sizes)
        n = len(residuals)

        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self.threshold = np.quantile(residuals, conformal_quantile)

    def predict(
        self,
        data: Iterable[tuple[Graph, Graph, Iterable[Graph], int]],
        n: int | None = None,
    ) -> Metrics:
        """Predict conformal sets and compute metrics on a test set.

        Args:
            data: iterable of (predicted graph, ground truth graph, candidate graphs, amount of candidates) tuples
            n: number of samples for tqdm progress display despite streaming
        """
        assert self.threshold is not None and self.regressor is not None, (
            "Conformal predictor must be fitted before prediction."
        )

        # Accumulate metrics
        correct_coverage: list[bool] = []
        candidate_sizes: list[int] = []
        conformal_sizes: list[int] = []

        for pred, truth, cands, size in tqdm(
            data,
            "Predicting with candidate size SCQR predictor",
            total=n,
        ):
            candidate_size = 0
            conformal_size = 0
            regressor_threshold = self.threshold + self.regressor.predict([[size]])[0]

            # Compute conformal set size
            for candidate in cands:
                candidate_size += 1
                if self.fgw(candidate, pred) <= regressor_threshold:
                    conformal_size += 1

            # Check if ground truth is in conformal set
            correct_coverage.append(self.fgw(truth, pred) <= regressor_threshold)

            candidate_sizes.append(candidate_size)
            conformal_sizes.append(conformal_size)

        return Metrics(
            correct_coverage=correct_coverage,
            candidate_sizes=candidate_sizes,
            conformal_sizes=conformal_sizes,
        )
