from typing import Iterable, Literal

import numpy as np
import ot
from tqdm import tqdm

from conformal.graph import Graph


class FGW:
    """Fused Gromov-Wasserstein distance between two graphs."""

    def __init__(
        self,
        cost: Literal["adjacency", "laplacian", "shortest_path"] = "adjacency",
        alpha=0.5,
        k=1,
        diffusion=False,
        loss: Literal["square_loss", "kl_loss"] = "square_loss",
    ):
        """
        Args:
            cost: the cost matrix to use
            alpha: trade-off parameter
            k: cost matrix exponent
            diffusion: whether to diffuse node features with the cost matrix
            loss: solver loss function
        """
        self.cost = cost
        self.alpha = alpha
        self.k = k
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

        if self.k > 1:
            C1 = np.linalg.matrix_power(C1, self.k)
            C2 = np.linalg.matrix_power(C2, self.k)

        f1 = g1.F
        f2 = g2.F

        # Diffuse node features with the cost matrix
        if self.diffusion:
            f1 = C1 @ f1
            f2 = C2 @ f2

        # Feature distance matrix
        M = ot.dist(f1, f2, metric="euclidean")

        return ot.gromov.fused_gromov_wasserstein2(
            M,
            C1,
            C2,
            loss_fun=self.loss,
            alpha=self.alpha,
        )

    def stream(
        self, g1s: Iterable[Graph], g2s: Iterable[Graph], n: int | None = None
    ) -> Iterable[float]:
        """Compute distances for a list of graph pairs,
        in a stream-compatible way (without keeping large matrices in memory)."""

        for g1, g2 in tqdm(zip(g1s, g2s), desc="computing FGW", total=n):
            yield self(g1, g2)
