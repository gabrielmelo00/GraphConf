"""Any2Graph typed helpers"""

from typing import TypedDict

import torch
from Any2Graph.graphs.custom_graphs_classes import BatchedContinuousGraphs
from Any2Graph.Img2Graph.Coloring.Coloring_Dataset import ColoringDataset
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse


class A2GGraph(TypedDict):
    """Any2Graph graph

    Args:
        A: (n, n) adjacency matrix
        F: (n, 4) one-hot node colors
    """

    A: Tensor
    F: Tensor


# How much of each color a graph has.
# Graphs with the same color distribution are in the same equivalence class.
GraphClass = tuple[int, int, int, int]


def graph_class(graph: A2GGraph) -> GraphClass:
    """Get the graph's equivalence class, defined by its color distribution."""
    return tuple(graph["F"].sum(dim=0).int().tolist())


def equivalence_classes(dataset: ColoringDataset) -> dict[GraphClass, list[int]]:
    """Compute all equivalence classes of indices in the dataset."""

    classes: dict[GraphClass, list[int]] = {}

    for _, graph, idx in iter(dataset):
        c = graph_class(graph)  # type: ignore
        if c in classes:
            classes[c].append(idx)
        else:
            classes[c] = [idx]

    return classes


def graph_to_sparse(graph: A2GGraph):
    """Convert a graph to a sparse torch geometric graph."""

    edge_index, edge_weight = dense_to_sparse(graph["A"])
    return Data(x=graph["F"], edge_index=edge_index, edge_weight=edge_weight)


def sparse_to_graph(data: Data) -> A2GGraph:
    """Convert a sparse torch geometric graph to a dense A2GGraph."""
    from torch_geometric.utils import to_dense_adj

    assert data.edge_index is not None
    assert data.edge_weight is not None
    assert data.x is not None

    A = to_dense_adj(data.edge_index, edge_attr=data.edge_weight).squeeze()
    F = data.x
    return {"A": A, "F": F}


def sparse_from_batch(batch: BatchedContinuousGraphs) -> list[Data]:
    """Convert a batch of predicted continuous graphs into a list of sparse torch geometric graphs."""

    graphs: list[Data] = []

    for i in range(len(batch)):
        A = batch.A[i]
        F = batch.F[i]
        h = batch.h[i]

        nodes = torch.where(h > 0.5)[0]
        F = F[nodes]
        A = A[nodes][:, nodes]

        graph: A2GGraph = {"A": A, "F": F}
        graphs.append(graph_to_sparse(graph))

    return graphs
