from typing import Literal

import numpy as np
import ot
import scipy
import scipy.linalg
import torch

from conformal.graph import Graph


def normalize(M: np.ndarray) -> np.ndarray:
    return M / (np.abs(M).max() + 1e-8)


class FGW:
    """Fused Gromov-Wasserstein distance between two graphs."""

    def __init__(
        self,
        cost: Literal["adjacency", "laplacian", "shortest_path"] = "adjacency",
        alpha=0.5,
        k=1,
        lmbda: float | None = None,
        diffusion=False,
        prior: Literal["identity", "sinkhorn"] = "identity",
        loss: Literal["square_loss", "kl_loss"] = "square_loss",
    ):
        """
        Args:
            cost: the cost matrix to use
            alpha: trade-off parameter
            k: cost matrix exponent
            lmbda: use exp(- lmbda * C) instead of C as cost matrix
            diffusion: whether to diffuse node features with the cost matrix
            prior: FGW transport prior of the G0 matrix
            loss: solver loss function
        """
        self.cost = cost
        self.alpha = alpha
        self.k = k
        self.lmbda = lmbda
        self.diffusion = diffusion
        self.prior = prior
        self.loss = loss
        # GPU acceleration for expm (takes a lot of CPU)
        self.gpu: str | None = "cuda" if torch.cuda.is_available() else None

    def cost_matrix(self, g: Graph) -> np.ndarray:
        match self.cost:
            case "adjacency":
                return g.A
            case "laplacian":
                return g.L
            case "shortest_path":
                return g.shortest_paths

    def expm(self, A: np.ndarray) -> np.ndarray:
        """Do matrix exponential on GPU if available, otherwise on CPU."""
        if self.gpu is not None:
            return torch.matrix_exp(torch.from_numpy(A).to(self.gpu)).cpu().numpy()
        else:
            return scipy.linalg.expm(A)

    def __call__(self, g1: Graph, g2: Graph) -> float:

        C1 = self.cost_matrix(g1)
        C2 = self.cost_matrix(g2)

        if self.lmbda is not None:
            C1 = self.expm(-self.lmbda * C1)
            C2 = self.expm(-self.lmbda * C2)

        elif self.k > 1:
            C1 = np.linalg.matrix_power(C1, self.k)
            C2 = np.linalg.matrix_power(C2, self.k)

        f1 = g1.F
        f2 = g2.F

        # Diffuse node features with the cost matrix
        if self.diffusion:
            f1 = C1 @ f1
            f2 = C2 @ f2

            # Normalize features
            f1 = normalize(f1)
            f2 = normalize(f2)

        # Feature distance matrix
        M: np.ndarray = ot.dist(f1, f2, metric="euclidean")  # type: ignore

        # Normalize matrices to avoid numerical errors
        # (reduces distance a lot between identical graphs!)
        M = normalize(M)
        C1 = normalize(C1)
        C2 = normalize(C2)

        p = np.ones(len(f1)) / len(f1)
        q = np.ones(len(f2)) / len(f2)

        # Initialization prior
        G0: np.ndarray | None = None

        match self.prior:
            case "sinkhorn":
                G0 = ot.bregman.sinkhorn(p, q, M, reg=1e-2)  # type: ignore
            case "identity":
                if len(p) == len(q):
                    G0 = np.eye(len(p)) / len(p)

        # Because the prior might not conform to the solver marginal constraints,
        # we fallback to the default p^T.q initialization on exception
        try:
            return ot.gromov.fused_gromov_wasserstein2(  # type: ignore
                M,
                C1,
                C2,
                p,
                q,
                G0=G0,
                loss_fun=self.loss,
                alpha=self.alpha,
            )
        except ValueError:
            return ot.gromov.fused_gromov_wasserstein2(  # type: ignore
                M,
                C1,
                C2,
                p,
                q,
                # No G0, use the default ones / n one
                loss_fun=self.loss,
                alpha=self.alpha,
            )
