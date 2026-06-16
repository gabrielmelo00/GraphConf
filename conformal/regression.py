"""Quantile estimators"""

import numpy as np
import torch
from scipy.optimize import linprog, minimize
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


class NumpyRegressor:
    """Conformal quantile estimator that uses any information
    (embeddings, uncertainty...) stored in a np.array as an indicator
    of task uncertainty
    """

    def __init__(
        self,
        target=0.9,
        **kwargs,
    ):
        """Initialize the regressor.

        Args:
            target: the target coverage level (e.g. 0.9 for 90% coverage)
            kwargs: additional init args for the regressor
        """

        self.regressor = QuantileRegressor(quantile=target, alpha=0.0, **kwargs)
        self.target = target
        self._threshold: float | None = None

    def fit(self, distances: list[float], data: np.ndarray, split: float = 0.8):
        """Fit the regressor on a calibration set."""
        mask = np.random.random(len(distances)) < split
        np_distances = np.array(distances)

        # Fit the quantile regressor
        self.regressor.fit(data[mask], np_distances[mask])

        # Estimate the quantile of the residuals
        residuals = np_distances[~mask] - self.regressor.predict(data[~mask])
        n = len(residuals)
        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self._threshold = np.quantile(residuals, conformal_quantile)

    def threshold(self, data: np.ndarray) -> float:
        """Predict the non-conformity threshold for one given data (not a batch)."""
        assert self.regressor is not None, "Regressor must be fitted before predicting"
        assert self._threshold is not None, "Regressor must be fitted before predicting"

        return self.regressor.predict(data[None, :])[0] + self._threshold


class RFFEmbeddingRegressor:
    """Gaussian kernel quantile regression via Random Fourier Features.

    Approximates k(x,y)=exp(-||x-y||²/2σ²) by drawing D random features
    z(x)=√(2/D)·cos(Wᵀx+b), W~N(0,σ⁻²I), b~Uniform(0,2π), then fitting a
    quantile regressor in that feature space.

    alpha=0  → standard quantile LP (HiGHS)
    alpha>0  → L2-regularized pinball via L-BFGS-B over [w, b] only
    sigma    → None uses the median trick on training embeddings
    """

    def __init__(
        self,
        target: float = 0.9,
        n_rff_features: int = 500,
        sigma: float | None = None,
        alpha: float = 0.0,
        quantile_method: str = "higher",
        seed: int = 42,
    ):
        self.target = target
        self.n_rff_features = n_rff_features
        self.sigma = sigma
        self.alpha = alpha
        self.quantile_method = quantile_method
        self.seed = seed
        self._threshold: float | None = None
        self._W: np.ndarray | None = None
        self._rff_b: np.ndarray | None = None
        self._w: np.ndarray | None = None
        self._intercept: float = 0.0
        self.sigma_used_: float | None = None

    # ------------------------------------------------------------------
    # RFF helpers
    # ------------------------------------------------------------------

    def _fit_rff(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        d = X.shape[1]

        if self.sigma is None:
            sub = X[: min(len(X), 1000)].astype(np.float64)
            norms_sq = (sub**2).sum(axis=1)
            dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * sub @ sub.T
            np.clip(dists_sq, 0.0, None, out=dists_sq)
            triu = dists_sq[np.triu_indices(len(sub), k=1)]
            sigma = float(np.sqrt(np.median(triu) / 2.0)) if triu.size else 1.0
            sigma = max(sigma, 1e-6)
        else:
            sigma = float(self.sigma)

        self.sigma_used_ = sigma
        self._W = rng.normal(0.0, 1.0 / sigma, size=(d, self.n_rff_features)).astype(
            np.float32
        )
        self._rff_b = rng.uniform(0.0, 2 * np.pi, size=(self.n_rff_features,)).astype(
            np.float32
        )

    def _apply_rff(self, X: np.ndarray) -> np.ndarray:
        assert self._W is not None and self._rff_b is not None
        Z = X.astype(np.float32) @ self._W + self._rff_b
        return np.sqrt(2.0 / self.n_rff_features) * np.cos(Z)

    def _solve_lp(self, Z: np.ndarray, y: np.ndarray) -> None:
        """Standard quantile LP via HiGHS (alpha=0).

        Variables: [w (D), b (1), ξ⁺ (n), ξ⁻ (n)]
        Minimize:  τ·Σξ⁺ + (1-τ)·Σξ⁻
        s.t.       y - Zw - b ≤ ξ⁺,  Zw + b - y ≤ ξ⁻,  ξ⁺,ξ⁻ ≥ 0
        """
        n, D = Z.shape
        tau = self.target
        Z64 = Z.astype(np.float64)
        y64 = y.astype(np.float64)

        c = np.concatenate([np.zeros(D + 1), tau * np.ones(n), (1 - tau) * np.ones(n)])
        ones_col = np.ones((n, 1), dtype=np.float64)
        I_n = np.eye(n, dtype=np.float64)
        zeros_nn = np.zeros((n, n), dtype=np.float64)

        A_ub = np.vstack(
            [
                np.hstack([-Z64, -ones_col, -I_n, zeros_nn]),
                np.hstack([Z64, ones_col, zeros_nn, -I_n]),
            ]
        )
        b_ub = np.concatenate([-y64, y64])
        bounds = [(-np.inf, np.inf)] * (D + 1) + [(0.0, np.inf)] * (2 * n)

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not result.success:
            raise RuntimeError(f"LP did not converge: {result.message}")

        self._w = result.x[:D].astype(np.float32)
        self._intercept = float(result.x[D])

    def _solve_qp(self, Z: np.ndarray, y: np.ndarray) -> None:
        """L2-regularized quantile regression via L-BFGS-B over [w, b] only.

        Minimize: (α/2)||w||² + Σᵢ ρ_τ(yᵢ - zᵢ·w - b)
        where ρ_τ(r) = τ·max(r,0) + (1-τ)·max(-r,0)
        """
        n, D = Z.shape
        tau, alpha = self.target, self.alpha
        Z64, y64 = Z.astype(np.float64), y.astype(np.float64)

        def obj_and_grad(params: np.ndarray):
            w, b = params[:D], params[D]
            r = y64 - Z64 @ w - b
            loss = (
                0.5 * alpha * np.dot(w, w)
                + tau * np.maximum(r, 0).sum()
                + (1 - tau) * np.maximum(-r, 0).sum()
            )
            sg = np.where(r > 0, tau, np.where(r < 0, -(1 - tau), 0.0))
            return loss, np.append(alpha * w - Z64.T @ sg, -sg.sum())

        result = minimize(
            obj_and_grad,
            np.zeros(D + 1),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        assert result is not None
        self._w = result.x[:D].astype(np.float32)
        self._intercept = float(result.x[D])

    def _predict_raw(self, Z: np.ndarray) -> np.ndarray:
        assert self._w is not None
        return Z @ self._w + self._intercept

    def fit(
        self, distances: list[float], embeddings: np.ndarray, split: float = 0.8
    ) -> None:
        mask = np.random.random(len(distances)) < split
        X_train = embeddings[mask]
        self._fit_rff(X_train)
        Z_train = self._apply_rff(X_train)
        y_train = np.array(distances)[mask]

        if self.alpha > 0.0:
            self._solve_qp(Z_train, y_train)
        else:
            self._solve_lp(Z_train, y_train)

        Z_calib = self._apply_rff(embeddings[~mask])
        y_calib = np.array(distances)[~mask]

        residuals = y_calib - self._predict_raw(Z_calib)
        n = len(residuals)
        conformal_quantile = np.ceil((n + 1) * self.target) / n
        self._threshold = float(
            np.quantile(residuals, conformal_quantile, method=self.quantile_method)  # type: ignore
        )

    def threshold(self, embedding: np.ndarray) -> float:
        assert self._threshold is not None, "Regressor must be fitted before predicting"
        Z = self._apply_rff(embedding.reshape(1, -1))
        return float(self._predict_raw(Z)[0]) + self._threshold


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

    def fit(self, distances: list[float], embeddings: Tensor, split: float = 0.8):
        """Fit the regressor on a calibration set.

        The regressor is trained on a split of the distances and embeddings,
        then residuals are computed on the remaining split.

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

        mask = np.random.random(len(distances)) < split

        dataset = TensorDataset(embeddings[mask], torch.tensor(distances)[mask])
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

        # Estimate the quantile of the residuals on the remaining split
        self.model.eval()
        preds: list[float] = []

        dataset = TensorDataset(embeddings[~mask], torch.tensor(distances)[~mask])
        loader = DataLoader(dataset, batch_size, shuffle=True)

        for embeds, _ in loader:
            embeds = embeds.to(self.device)
            with torch.no_grad():
                pred = self.model(embeds).squeeze().cpu()
            preds.extend(pred)

        residuals = np.array(distances)[~mask] - np.array(preds)
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
