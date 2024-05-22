import string
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

LATTICE_STRUCTURE = {
    "pyroxene": "amorphous",
    "forsterite": "crystalline",
    "enstatite": "crystalline",
    "silica": "crystalline",
    "carbon": "amorphous"
    }

CHEMICAL_FORMULAS = {
    "pyroxene": r"\ce{Mg_{x}Fe_{1-x}SiO3}",
    "forsterite": r"\ce{Mg2SiO4}",
    "enstatite": r"\ce{MgSiO3}",
    "silica": r"\ce{SiO2}",
    "carbon": r"\ce{C}"
    }


def read_dust_species_to_table(
        weights: np.ndarray, names: List[str],
        methods: List[str], sizes: List[float],
        fmaxs: Optional[List[float]] = None) -> None:
    """Read dust species to table."""
    data = {"type": [], "structure": [], "formula": [],
            "method": [], "fmax": [], "grain_size": [], "weight": weights}

    letters = string.ascii_lowercase
    for index, size in enumerate(sizes):
        name = [names[index].title()+r"$\tablefootmark{"+letters[index]+r"}$"]
        structure = [LATTICE_STRUCTURE[names[index]].title()]
        formula = [CHEMICAL_FORMULAS[names[index]]]
        method = [methods[index]]
        fmax = [fmaxs[index]] if fmaxs is not None else None

        if len(size) > 1:
            name += [""]*(len(size) - 1)
            structure += [""]*(len(size) - 1)
            formula += [""]*(len(size) - 1)
            method += [""]*(len(size) - 1)
            if fmax is not None:
                fmax += [""]*(len(size) - 1)

        data["type"].extend(name)
        data["structure"].extend(structure)
        data["formula"].extend(formula)
        data["method"].extend(method)
        data["fmax"].extend(fmax)
        data["grain_size"].extend(size)

    df = pd.DataFrame(data)
    df.to_csv("dust_species.csv", index=False, header=False)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/fitting_data/hd142527")
    weights = [73.2, 8.6, 0.6, 14.2, 2.4, 1.0, r"c"]
    names = ["pyroxene", "forsterite", "enstatite", "silica", "carbon"]
    method = ["DHS"]*len(names)
    fmaxs = [1.0, 1.0, 1.0, 0.7, 1.0]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5], [0.1]]

    read_dust_species_to_table(weights, names, method, sizes, fmaxs)
