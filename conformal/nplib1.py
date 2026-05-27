"""
NPLIB1 dataset loading utils.
"""

from pathlib import Path
from typing import Literal, TypedDict

import h5py
import numpy as np
import polars as pl
from mist.data.data import Spectra
from tqdm import tqdm


def make_nplib1(root: Path = Path("mist/data/paired_spectra/canopus_train")):
    """Compute csv files from the NPLIB1 preprocessed files of the MIST repository"""
    labels = pl.read_csv(root / "labels.tsv", separator="\t")
    splits = pl.read_csv(root / "splits" / "canopus_hplus_100_0.tsv", separator="\t")
    splits = splits.rename({"name": "spec"})

    # Form spectra file
    joined = splits.join(labels, on="spec")
    specs = joined.select(
        pl.col("spec").alias("spectrum"),
        pl.col("split"),
        pl.col("formula"),
        pl.col("inchikey"),
    )
    specs.write_csv("spectra.csv")

    inchikeys = []
    formulas = []
    smiles = []

    with h5py.File(
        root / "retrieval_hdf" / "intpubchem_with_morgan4096_retrieval_db.h5"
    ) as hdf:
        for formula in tqdm(specs["formula"].unique()):
            index = np.where(np.array(hdf["formulae"]).astype(str) == formula)[0]

            if len(index) == 0:
                continue

            length = hdf["formula_lengths"][index[0]]
            offset = hdf["formula_offset"][index[0]]
            sub_ikeys = hdf["ikeys"][offset : offset + length]
            sub_smiles = hdf["smiles"][offset : offset + length]

            for ikey, smi in zip(sub_ikeys, sub_smiles):
                inchikeys.append(ikey.decode())
                smiles.append(smi.decode())
                formulas.append(formula)

    candidates = pl.DataFrame(
        {"inchikey": inchikeys, "formula": formulas, "smiles": smiles}
    )
    candidates.write_csv("candidates.csv")


class Entry(TypedDict):
    candidates: list[str]  # smiles
    spectrum: str  # spectrum id
    truth: str  # truth smile


def load_split(
    split: Literal["train", "test", "val"], path: Path = Path("./NPLIB1")
) -> list[Entry]:
    """Load all entries for a specific split"""
    candidates = pl.read_csv(path / "candidates.csv")
    spectra = pl.read_csv(path / "spectra.csv")
    spectra = spectra.filter(pl.col("split") == split)

    entries: list[Entry] = []

    for spectrum, _, formula, inchikey in spectra.iter_rows():
        smiles = candidates.filter(pl.col("formula") == formula)
        truth = smiles.filter(pl.col("inchikey") == inchikey).head(1)["smiles"]

        if len(truth) != 1:
            continue

        entries.append(
            {
                "spectrum": spectrum,
                "candidates": smiles["smiles"].to_list(),
                "truth": truth.item(),
            }
        )

    return entries


def load_spectrum(spectrum: str, root: Path = Path("./NPLIB1")) -> Spectra:
    spec = Spectra(
        spectra_name=spectrum,
        spectra_file=root / "spectra" / f"{spectrum}.ms",  # type: ignore
    )
    spec._load_spectra()
    return spec
