"""
Benchmarking functions
"""

from time import perf_counter
from typing import Literal

import numpy as np
import ot

from conformal.fgw import FGW
from conformal.graph import Graph


def _cost_matrix(prior: Literal["FD", "LFD", "LFD-sym"], graph: Graph) -> np.ndarray:
    match prior:
        case "FD":
            return graph.A
        case "LFD":
            return graph.L
        case "LFD-sym":
            return graph.L_normalized


def bench_prior(
    prior: Literal["FD", "LFD", "LFD-sym"], graphs: list[Graph], n: int = 1
):
    """Benchmark the FGW prior for a series of graphs
    with the same amount of nodes (a common assumption in our candidate sets).

    Only benchmarks the pot.dist and pot.emd functions.

    Computes pairwise distances in the list of graphs, n times.
    Total amount of calls to pot.emd is len(graphs) * len(graphs) * n

    Returns: average time per call.
    """
    assert len(graphs) > 0
    num_nodes = len(graphs[0].F)
    assert all(map(lambda g: len(g.F) == num_nodes, graphs)), (
        "All graphs do not have the same amount of nodes"
    )

    p = np.ones(num_nodes) / num_nodes

    # Precompute features
    features = [
        np.concat((graph.F, _cost_matrix(prior, graph) @ graph.F), axis=1)
        for graph in graphs
    ]

    start = perf_counter()
    for _ in range(n):
        for f1 in features:
            for f2 in features:
                M = ot.dist(f1, f2, metric="euclidean")
                ot.emd(p, p, M)
    end = perf_counter()

    return (end - start) / (len(graphs) * len(graphs) * n)


def bench_fgw(fgw: FGW, graphs: list[Graph], n: int = 1):
    """Benchmark the FGW distance for a series of graphs
    with the same amount of nodes (a common assumption in our candidate sets).

    Computes pairwise distances in the list of graphs, n times.
    Total amount of calls to FGW is len(graphs) * len(graphs) * n

    Returns: average time per call.
    """
    assert len(graphs) > 0
    num_nodes = len(graphs[0].F)
    assert all(map(lambda g: len(g.F) == num_nodes, graphs)), (
        "All graphs do not have the same amount of nodes"
    )

    start = perf_counter()
    for _ in range(n):
        for g1 in graphs:
            for g2 in graphs:
                fgw(g1, g2)
    end = perf_counter()

    return (end - start) / (len(graphs) * len(graphs) * n)
