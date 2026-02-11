import typing
from dataclasses import dataclass
from functools import cached_property
from typing import Self

import numpy as np
from scipy.sparse.csgraph import shortest_path

if typing.TYPE_CHECKING:
    from conformal.any2graph import A2GGraph


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

    @classmethod
    def from_a2g(cls, graph: "A2GGraph") -> Self:
        """Convert an Any2Graph graph to a Graph object."""
        return cls(A=graph["A"].numpy(), F=graph["F"].numpy())

    @classmethod
    def from_smiles(cls, smiles: str, use_onehot=True) -> Self:
        """Convert a SMILES string to a Graph object.

        Args:
            smiles: the SMILES string to convert
            use_onehot: whether to use one-hot encoding for atom types. Else, atomic number
        """

        # Lazy import to avoid rdkit dependency if not used
        from rdkit import Chem
        from rdkit.Chem import rdmolops

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
                F.append([atomic_num])  # type: ignore

        return Graph(A=A, F=np.array(F, dtype=np.float32))
