from typing import Literal

import numpy as np
import ot
import scipy

from conformal.graph import Graph


class FGW:
    """Fused Gromov-Wasserstein distance between two graphs."""

    def __init__(
        self,
        cost: Literal["adjacency", "laplacian", "shortest_path"] = "adjacency",
        alpha=0.5,
        k=1,
        lmbda: float | None = None,
        diffusion=False,
        loss: Literal["square_loss", "kl_loss"] = "square_loss",
    ):
        """
        Args:
            cost: the cost matrix to use
            alpha: trade-off parameter
            k: cost matrix exponent
            lmbda: use exp(- lmbda * C) instead of C as cost matrix
            diffusion: whether to diffuse node features with the cost matrix
            loss: solver loss function
        """
        self.cost = cost
        self.alpha = alpha
        self.k = k
        self.lmbda = lmbda
        self.diffusion = diffusion
        self.loss = loss

    def cost_matrix(self, g: Graph) -> np.ndarray:
        match self.cost:
            case "adjacency":
                return g.A
            case "laplacian":
                return g.L
            case "shortest_path":
                return g.shortest_paths

    def __call__(self, g1: Graph, g2: Graph) -> float:

        C1 = self.cost_matrix(g1)
        C2 = self.cost_matrix(g2)

        if self.lmbda is not None:
            C1 = scipy.linalg.expn(-self.lmbda * C1)
            C2 = scipy.linalg.expn(-self.lmbda * C2)

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
            f1 /= np.abs(f1).max()
            f2 /= np.abs(f2).max()

        # Feature distance matrix
        M = ot.dist(f1, f2, metric="euclidean")

        # Normalize matrices to avoid numerical errors
        # (reduces distance a lot between identical graphs!)
        M /= np.abs(M).max()
        C1 /= np.abs(C1).max()
        C2 /= np.abs(C2).max()

        p = np.ones(len(f1)) / len(f1)
        q = np.ones(len(f2)) / len(f2)

        return ot.gromov.fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            p,
            q,
            loss_fun=self.loss,
            alpha=self.alpha,
        )
