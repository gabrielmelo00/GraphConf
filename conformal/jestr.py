"""
JESTR1 utils
"""

import os
import pickle
from pathlib import Path

import dgllife
from dgl import DGLGraph
from rdkit import Chem

from JESTR1.dataset import get_atom_featurizer, get_bond_featurizer


def smiles_to_graph(smiles: str, params) -> DGLGraph:
    """Convert a smile formula to a DGL graph.
    May raise exceptions."""

    if "." in smiles:
        raise ValueError("Disconnected smile")

    mol = Chem.MolFromSmiles(smiles)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    if len(atoms) <= 1:
        raise ValueError("At least 2 atoms required")

    if any(atom not in params["element_list"] for atom in atoms):
        raise ValueError("Invalid atom encountered")

    node_feat = get_atom_featurizer(params["atom_feature"], params["element_list"])
    edge_feat = get_bond_featurizer(params["bond_feature"], self_loop=True)

    graph = dgllife.utils.mol_to_bigraph(
        mol,
        node_featurizer=node_feat,
        edge_featurizer=edge_feat,
        add_self_loop=True,
        num_virtual_nodes=0,
    )

    assert graph is not None

    return graph


class FSDict:
    """A "dict" that actually reads from a folder
    Used to read the extracted contents of "molgraph_dict.pkl" and "data_dict.pkl"
    which take up an insane amount of RAM when loaded directly.
    """

    def __init__(self, path: str, ext: str = "pkl"):
        self.ext = ext
        self.root = Path(path)

    def keys(self):
        return sorted(map(lambda x: os.path.splitext(x)[0], os.listdir(self.root)))

    def __len__(self):
        return len(self.keys())

    def __getitem__(self, name: str):
        return pickle.load(open(self.root / f"{name}.{self.ext}", "rb"))
