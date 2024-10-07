import string
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from matadrs.utils.readout import ReadoutFits
from ppdmod.analysis import restore_from_fits
from ppdmod.component import Component
from ppdmod.plot import format_labels
from pylatex import Document, Section, Tabular
from pylatex.utils import NoEscape


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


def read_to_table(fits_files: List[Path], save_as_csv: Optional[bool] = False,
                  savefig: Optional[Path] = False):
    """Read a FITS file and return the data as an astropy table."""
    data = {"instrument": [], "date": [], "seeing": [], "tau0": [],
            "stations": [], "array": [], "name": [], "ldd": [],
            "time": [], "resolution": [], "comment": []}

    for fits_file in fits_files:
        readout = ReadoutFits(fits_file)

        data["instrument"].append(readout.instrument_mode.upper())
        data["date"].append(readout.date[:-8])
        data["seeing"].append(round(readout.seeing, 1))
        data["tau0"].append(round(readout.tau0, 1))
        data["stations"].append(readout.stations)
        data["array"].append(readout.array)

        cal_name, cal_time, cal_ldd = readout.get_calib_info()
        data["name"].append(cal_name)
        data["ldd"].append(cal_ldd)
        data["time"].append(cal_time)

        comment, resolution = "", ""
        if readout.instrument.lower() == "pionier":
            comment = "H"
        elif readout.instrument.lower() == "gravity":
            comment = "K"
        elif readout.instrument.lower() == "matisse":
            if readout.array == "UTs":
                comment = "LN"
            else:
                comment = "L"

            resolution = "/".join([readout.primary_header[f"hierarch eso ins di{band} name"] for band in comment])

        data["comment"].append(comment)
        data["resolution"].append(resolution)

    df = pd.DataFrame(data).sort_values(by="date")
    df.to_csv("observations.csv", index=False, header=False)


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


def best_fit_parameters(labels: np.ndarray, units: np.ndarray,
                        values: np.ndarray,
                        uncertainties: Optional[np.ndarray] = None,
                        savefig: Optional[Path] = None,
                        save_as_csv: Optional[bool] = True) -> None:
    """Make a (.pdf) file containing a table of the fit parameters."""
    labels, units = format_labels(labels, units, split=True)
    uncertainties = np.round(np.abs(uncertainties - values[:, np.newaxis]), 2)
    values = [f"{value:.2e}" if np.abs(value) < 1e-2 else f"{value:.2f}" for value in values]

    if uncertainties is not None:
        values = [f"${{{value}}}_{{-{uncertainty[0]}}}^{{+{uncertainty[1]}}}$"
                  for value, uncertainty in zip(values, uncertainties)]

    if save_as_csv:
        data = {"Parameter": labels, "Unit": units, "Value": values}
        df = pd.DataFrame(data)
        df.to_csv(savefig, index=False, header=False, sep=";")
    else:
        doc = Document()
        with doc.create(Section("Fit Parameters")):
            with doc.create(Tabular("c | c | c")) as table:
                table.add_hline()
                table.add_row(("Parameter", "Unit", "Value"))
                table.add_hline()

                for label, unit, value in zip(labels, units, values):
                    table.add_row((NoEscape(label), NoEscape(unit), value))

                table.add_hline()

        doc.generate_pdf(savefig, clean_tex=False)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-10-02/downsampled")
    component_labels, components = restore_from_fits(path)
    make_fit_parameters_table(components)

