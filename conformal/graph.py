from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Self

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from scipy.sparse.csgraph import shortest_path


@dataclass(frozen=True)
class Graph:
    A: np.ndarray  # (n, n) adjacency matrix
    F: np.ndarray  # (n, d) node features

    @cached_property
    def L(self) -> np.ndarray:
        """Graph Laplacian."""
        D = np.diag(self.A.sum(axis=1))
        return D - self.A

    @cached_property
    def shortest_paths(self) -> np.ndarray:
        """Shortest paths between node pairs."""
        return shortest_path(self.A, directed=False, unweighted=True)

    @staticmethod
    def stream_from_smiles(smiles: Iterable[str], verbose=False) -> Iterable[Self]:
        """Stream Graph objects from an iterable of SMILES strings.
        Do this to process large smiles datasets without keeping large matrices in memory."""
        for s in smiles:
            try:
                yield Graph.from_smiles(s)
            except Exception as e:
                if verbose:
                    print(f"Error processing SMILES: {e}")

    @staticmethod
    def stream_from_smiles_pair(
        preds: Iterable[str], truths: Iterable[str], verbose=False
    ) -> Iterable[tuple[Self, Self]]:
        """Stream pairs of (predicted graph, ground truth graph) from an iterable of SMILES string pairs.
        This is necessary because smiles to graph can fail if the molecule is disconnected,
        and we don't want to introduce a shift in the (pred, truth) pairs by dropping only one of them."""

        for p, t in zip(preds, truths):
            try:
                yield Graph.from_smiles(p), Graph.from_smiles(t)
            except Exception as e:
                if verbose:
                    print(f"Error processing SMILES pair: {e}")

    @staticmethod
    def stream_from_smiles_pair_and_candidates(
        preds: Iterable[str],
        truths: Iterable[str],
        candidates: Iterable[Iterable[str]],
        verbose=False,
    ) -> Iterable[tuple[Self, Self, Iterable[Self]]]:
        """Stream triples of (predicted graph, ground truth graph, candidate graphs)
        from an iterable of SMILES string pairs and candidate lists.
        Same reason as `stream_from_smiles_pair`.
        Also, candidates can fail, that's not an issue, the resulting set is always non-empty.
        """

        for p, t, cands in zip(preds, truths, candidates):
            try:
                yield (
                    Graph.from_smiles(p),
                    Graph.from_smiles(t),
                    Graph.stream_from_smiles(cands, verbose=verbose),
                )
            except Exception as e:
                if verbose:
                    print(f"Error processing SMILES pair and candidates: {e}")

    @classmethod
    def from_smiles(cls, smiles: str, use_onehot=True) -> Self:
        """Convert a SMILES string to a Graph object.

        Args:
            smiles: the SMILES string to convert
            use_onehot: whether to use one-hot encoding for atom types. Else, atomic number
        """

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        if len(rdmolops.GetMolFrags(molecule)) > 1:
            raise ValueError(f"Disconnected molecule: {smiles}")

        n_atoms = molecule.GetNumAtoms()
        A = np.zeros((n_atoms, n_atoms), dtype=np.float32)

        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            A[i, j] = 1
            A[j, i] = 1

        atom_types = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
        atom_to_idx = {a: i for i, a in enumerate(atom_types)}
        n_atom_types = len(atom_types) + 1

        F = []
        for atom in molecule.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if use_onehot:
                onehot = np.zeros(n_atom_types, dtype=np.float32)
                onehot[atom_to_idx.get(atomic_num, -1)] = 1.0
                F.append(onehot)
            else:
                F.append([atomic_num])

        return Graph(A=A, F=np.array(F, dtype=np.float32))
