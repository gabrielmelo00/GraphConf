from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import ot
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Graph:
    A: np.ndarray   # (n, n) adjacency matrix
    F: np.ndarray   # (n, d) node features


def fgw_k(
    g1: Graph,
    g2: Graph,
    k: int = 1,
    alpha: float = 0.5,
    loss_fun: str = "square_loss",
    sp: bool = False,
) -> float:
    """
    Compute FGW distance between two graphs using A^k.

    For k = 1, recovers standard FGW.
    """

    # Number of nodes
    n1, n2 = g1.A.shape[0], g2.A.shape[0]

    # Uniform node distributions
    p = np.ones(n1) / n1
    q = np.ones(n2) / n2

    # Feature distance matrix (Euclidean)
    C = ot.dist(g1.F, g2.F, metric="euclidean")

    # k-hop structural information
    A1_k = np.linalg.matrix_power(g1.A, k)
    A2_k = np.linalg.matrix_power(g2.A, k)

    if sp:
        # Compute shortest-path distance matrices
        A1_k = shortest_path(A1_k, directed=False, unweighted=True)
        A2_k = shortest_path(A2_k, directed=False, unweighted=True)

    fgw2 = ot.gromov.fused_gromov_wasserstein2(
        C,
        A1_k,
        A2_k,
        p,
        q,
        loss_fun=loss_fun,
        alpha=alpha,
        verbose=False,
    )
    return fgw2


