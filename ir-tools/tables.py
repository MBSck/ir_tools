import string
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matadrs.utils.readout import ReadoutFits
from ppdmod.plot import format_labels
from pylatex import Document, MultiColumn, Section, Tabular
from pylatex.utils import NoEscape

LATTICE_STRUCTURE = {
    "pyroxene": "amorphous",
    "forsterite": "crystalline",
    "enstatite": "crystalline",
    "silica": "crystalline",
    "carbon": "amorphous",
}

CHEMICAL_FORMULAS = {
    "pyroxene": r"\ce{Mg_{x}Fe_{1-x}SiO3}",
    "forsterite": r"\ce{Mg2SiO4}",
    "enstatite": r"\ce{MgSiO3}",
    "silica": r"\ce{SiO2}",
    "carbon": r"\ce{C}",
}


def observations(fits_files: List[Path], savefig: Path | None = None):
    """Read a FITS file and return the data as an astropy table."""
    data = {
        "instrument": [],
        "date": [],
        "sci_seeing": [],
        "sci_tau0": [],
        "stations": [],
        "array": [],
        "cal_name": [],
        "cal_ldd": [],
        "cal_time": [],
        "cal_seeing": [],
        "cal_tau0": [],
        "resolution": [],
        "comment": [],
    }

    already_added = set()
    for fits_file in fits_files:
        readout = ReadoutFits(fits_file)
        time = datetime.strptime(
            readout.date.split("T")[1],
            "%H:%M:%S.%f" if "." in readout.date else "%H:%M:%S",
        )

        date = "T".join([readout.date.split("T")[0], time.strftime("%H:%M")])
        if date in already_added:
            continue

        already_added.add(date)
        data["instrument"].append(readout.instrument_mode.upper())
        data["date"].append(date)
        data["sci_seeing"].append(round(readout.seeing, 1))
        data["sci_tau0"].append(round(readout.tau0, 1))
        data["stations"].append(readout.stations)
        data["array"].append(readout.array)

        cal_name, cal_time, cal_ldd, cal_tau0, cal_seeing = readout.get_calib_info()
        data["cal_name"].append(cal_name)
        data["cal_ldd"].append(cal_ldd)
        data["cal_time"].append(cal_time)
        data["cal_tau0"].append(round(cal_tau0, 1) if cal_tau0 else "")
        data["cal_seeing"].append(round(cal_seeing, 1) if cal_seeing else "")

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

            resolution = "/".join(
                [
                    readout.primary_header[f"hierarch eso ins di{band} name"]
                    for band in comment
                ]
            )

        data["comment"].append(comment)
        data["resolution"].append(resolution)

    df = pd.DataFrame(data).sort_values(by="date")
    df.to_csv(
        savefig if savefig is not None else "observations.csv",
        index=False,
        header=False,
    )


def dust_species(
    weights: np.ndarray,
    names: List[str],
    methods: List[str],
    sizes: List[float],
    fmaxs: List[float] | None = None,
) -> None:
    """Read dust species to table."""
    data = {
        "type": [],
        "structure": [],
        "formula": [],
        "method": [],
        "fmax": [],
        "grain_size": [],
        "weight": weights,
    }

    letters = string.ascii_lowercase
    for index, size in enumerate(sizes):
        name = [names[index].title() + r"$\tablefootmark{" + letters[index] + r"}$"]
        structure = [LATTICE_STRUCTURE[names[index]].title()]
        formula = [CHEMICAL_FORMULAS[names[index]]]
        method = [methods[index]]
        fmax = [fmaxs[index]] if fmaxs is not None else None

        if len(size) > 1:
            name += [""] * (len(size) - 1)
            structure += [""] * (len(size) - 1)
            formula += [""] * (len(size) - 1)
            method += [""] * (len(size) - 1)
            if fmax is not None:
                fmax += [""] * (len(size) - 1)

        data["type"].extend(name)
        data["structure"].extend(structure)
        data["formula"].extend(formula)
        data["method"].extend(method)
        data["fmax"].extend(fmax)
        data["grain_size"].extend(size)

    df = pd.DataFrame(data)
    df.to_csv("dust_species.csv", index=False, header=False)


def best_fit_parameters(
    labels: np.ndarray,
    units: np.ndarray,
    values: np.ndarray,
    uncertainties: np.ndarray | None = None,
    savefig: Path | None = None,
    save_as_csv: bool = True,
) -> None:
    """Make a (.pdf) file containing a table of the fit parameters."""
    labels, units = format_labels(labels, units, split=True)
    if uncertainties is not None:
        uncertainties = np.round(np.abs(uncertainties - values[:, np.newaxis]), 2)

    values = [
        f"{value:.2e}" if np.abs(value) < 1e-2 else f"{value:.2f}" for value in values
    ]

    if uncertainties is not None:
        values = [
            f"${{{value}}}_{{-{uncertainty[0]}}}^{{+{uncertainty[1]}}}$"
            for value, uncertainty in zip(values, uncertainties)
        ]

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
                    table.add_row((NoEscape(label), NoEscape(unit), NoEscape(value)))

                table.add_hline()

        doc.generate_pdf(savefig, clean_tex=False)


def opacities(weight_file: Path) -> None:
    _, weights = np.load(weight_file)
    weights = np.round(weights.astype(float) / weights.astype(float).sum() * 1e2, 2)

    labels = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
    boekel_weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0])
    juhasz_weights = np.array([0, 22.98, 3.23, 6.27, 5.7, 0, 6.56, 4.55, 37.14, 13.57])

    labels = [""] + [MultiColumn(2, align="c |", data=label) for label in labels]
    table = " ".join(
        ["c" if index % 2 == 0 else "c |" for index in range(1, weights.size + 2)]
    )

    doc = Document()
    with doc.create(Section("Opacity weights")):
        with doc.create(Tabular(table)) as table:
            table.add_row(labels)
            table.add_hline()
            table.add_row([""] + ["Small", "Large"] * (len(labels) - 1))
            table.add_hline()
            table.add_row(["Mine"] + weights.tolist())
            table.add_row(["Juhasz et al. (2009)"] + juhasz_weights.tolist())
            table.add_row(["Boekel et al. (2005)"] + boekel_weights.tolist())
            table.add_hline()

    doc.generate_pdf("hd142527_opacity", clean_tex=False)


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    fits_files = list((data_dir / "fitting_data" / "hd142527").glob("*.fits"))
    observations(fits_files)

    # sed_fit_dir = data_dir / "opacities" / "silicate_labels_and_weights.npy"
    # opacities(sed_fit_dir)
