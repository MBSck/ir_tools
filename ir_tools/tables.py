import string
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from matadrs.utils.readout import ReadoutFits
from ppdmod.plot import format_labels
from pylatex import Document, MultiColumn, NoEscape, Section, Tabular

from . import variables as var
from .utils import query


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
        structure = [var.LATTICE_STRUCTURE[names[index]].title()]
        formula = [var.CHEMICAL_FORMULAS[names[index]]]
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
    fit_method: str = "dynesty",
) -> None:
    """Make a (.pdf) file containing a table of the fit parameters."""
    labels, units = format_labels(labels, units, split=True)
    if uncertainties is not None:
        new_values = []
        for value, uncertainty in zip(values, uncertainties):
            if value < 1e-2:
                value = f"{value:.2e}" 
                uncertainty = f"_{{-{uncertainty[0]:.2e}}}^{{+{uncertainty[1]:.2e}}}"
            else:
                value = f"{value:.2f}" 
                uncertainty = f"_{{-{uncertainty[0]:.2f}}}^{{+{uncertainty[1]:.2f}}}"

            new_values.append(f"${{{value}}}{{{uncertainty}}}$")
        values = new_values
    else:
        values = [
            f"{value:.2e}" if np.abs(value) < 1e-2 else f"{value:.2f}" for value in values
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


def get_source(excel_file: Path, target: str) -> pd.DataFrame:
    """Gets the night a target was observed for some criteria."""
    df = pd.read_excel(excel_file, skiprows=1, header=[0, 1])
    target_column = df["Target"].iloc[:, 0].astype(str)
    df_cleaned = df.dropna(subset=[("Target", target_column.name)])
    return df_cleaned[
        df_cleaned["Target"].iloc[:, 0].str.contains(target, case=False, regex=True)
    ]


def get_sources(
    excel_file: Path, sheet_name: str = "target_list_status", band: str = "lband"
) -> List[str]:
    """Gets the list of source names to sort from the excel file."""
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=[0, 1])
    lband_selection = df["Selected"]["for L paper"].astype(str)
    df = df.dropna(subset=[("Selected", lband_selection.name)])
    df = df[~lband_selection.isin(["no", "tbd"])]

    if band == "nband":
        nband_selection = df["Selected"]["for LN paper"].astype(str)
        df = df[nband_selection != "no"]

    return sorted(df["Name"].iloc[:, 0].tolist())


def get_dir_name(source: str) -> str:
    """Returns the directory name for a source from its excel sheet name."""
    if " " in source:
        return source.replace(" ", "_")

    if source.startswith("HD"):
        return source[:2] + "_" + source[2:]

    return source


def get_date(night: str, nights: pd.DataFrame) -> str | None:
    """Compares the obsevation dates of the file name to the one found
    in the excel sheet and returns the one found in the file name.

    Parameters
    ----------
    night : str
        The night to compare.
    nights : pd.DataFrame
        The nights found in the excel sheet.

    Returns
    -------
    date :  str
        The date found in the file name.
    """
    night = datetime.strptime(night, "%Y-%m-%d")
    night_before = (night - timedelta(days=1)).strftime("%Y-%m-%d")
    night_after = (night + timedelta(days=1)).strftime("%Y-%m-%d")
    night = night.strftime("%Y-%m-%d")

    if night in nights.values():
        key = night
    elif night_before in nights.values():
        key = night_before
    elif night_after in nights.values():
        key = night_after
    else:
        key = None

    return key


# TODO: Make a sorting here for the chopped then non-chopped data, etc.
# Do the same for the new data type and also sort by bands.
# Also reimplement the arrays (but from the header this time).
# Chopped, NCcorrCtot, NOCHOP is the progressing of the files themselves
def source_info(
    data_dir: Path, sources: List[str], save_dir: Path | None = None
) -> None:
    save_dir = Path().cwd() if save_dir is None else save_dir

    def sorting_key(row) -> int:
        if row["band"] == "LM":
            if row["chopped"]:
                return 0
            else:
                return 1
        else:
            return 2

    dfs = []
    sources_sorted = list(
        map(get_dir_name, query(sources).sort_values(by="RA")["source"].tolist())
    )
    for source in sources_sorted:
        info = {key: [] for key in var.SOURCE_INFO_KEYS}
        source_dir = data_dir / source / "matisse" / "non_treated"
        for fits_file in list(source_dir.glob("*.fits")):
            if "NCcorrCtot" in fits_file.stem:
                continue

            with fits.open(fits_file) as hdul:
                header = hdul[0].header
                array = var.ARRAY_CONFIGS.get(
                    "-".join(hdul["oi_array"].data["sta_name"]), "other"
                )

            info["source"].append(source)
            tpl_start = header["HIERARCH ESO TPL START"]
            date = datetime.strptime(tpl_start.split("T")[0], "%Y-%m-%d")
            info["year"].append(date.year)
            info["date"].append(f"{date.strftime('%B')[:3]} {date.day}")
            info["tpl_start"].append(tpl_start)
            info["pipe_version"].append(
                header["HIERARCH ESO PRO REC1 PIPE ID"].split("/")[1]
            )
            sci_seeing_start = header["HIERARCH ESO ISS AMBI FWHM START"]
            sci_seeing_end = header["HIERARCH ESO ISS AMBI FWHM END"]
            info["sci_seeing_start"].append(sci_seeing_start)
            info["sci_seeing_end"].append(sci_seeing_end)

            sci_tau0_start = np.round(
                header["HIERARCH ESO ISS AMBI TAU0 START"] * 1e3, 1
            )
            sci_tau0_end = np.round(header["HIERARCH ESO ISS AMBI TAU0 END"] * 1e3, 1)
            info["sci_tau0_start"].append(sci_tau0_start)
            info["sci_tau0_end"].append(sci_tau0_end)

            cal_name, cal_ra, cal_dec = "", 0, 0
            cal_diam, cal_diam_err, cal_seeing = "", "", ""
            cal_tau0 = ""
            if "HIERARCH ESO PRO CAL NAME" in header:
                cal_name = header["HIERARCH ESO PRO CAL NAME"]
                cal_ra = header["HIERARCH ESO PRO CAL RA"]
                cal_dec = header["HIERARCH ESO PRO CAL DEC"]
                cal_diam = header["HIERARCH ESO PRO CAL DB DIAM"]
                cal_diam_err = header["HIERARCH ESO PRO CAL DB ERRDIAM"]
                cal_seeing = np.round(header["HIERARCH ESO PRO CAL FWHM"], 1)
                cal_tau0 = np.round(header["HIERARCH ESO PRO CAL TAU0"] * 1e3, 1)

            if cal_name.startswith("HD"):
                cal_name = cal_name.replace("_", "")
            else:
                cal_name = cal_name.replace("_", " ")

            info["cal_name"].append(cal_name)
            info["cal_ra"].append(cal_ra)
            info["cal_dec"].append(cal_dec)
            info["cal_diam"].append(cal_diam)
            info["cal_diam_err"].append(cal_diam_err)
            info["cal_seeing"].append(cal_seeing)
            info["cal_tau0"].append(cal_tau0)
            info["file_name"].append(fits_file.stem)

            info["array"].append(array)
            info["band"].append(header["HIERARCH ESO DET NAME"].split("-")[1])
            info["chopped"].append("CHOPPED" in fits_file.stem)

        source_df = pd.DataFrame(info)
        source_df["tpl_start"] = pd.to_datetime(source_df["tpl_start"])
        source_df["sort_key"] = source_df.apply(sorting_key, axis=1)
        source_df = source_df.sort_values(by=["tpl_start", "sort_key"])
        dfs.append(source_df.drop(columns=["sort_key"]))

    df = pd.concat(dfs, axis=0, ignore_index=True)
    coord = SkyCoord(
        ra=df["cal_ra"].tolist() * u.deg,
        dec=df["cal_dec"].tolist() * u.deg,
        frame="icrs",
    )
    df["cal_ra"] = coord.ra.to_string(unit=u.hour, sep=":", precision=2)
    df["cal_dec"] = [
        f"'{value}"
        for value in coord.dec.to_string(
            unit=u.deg, sep=":", precision=2, alwayssign=True
        )
    ]
    df["tpl_start"] = df["tpl_start"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    df.to_excel(save_dir / "source_tpl_table.xlsx", index=False)


def fit_results(
    sources: List[str],
    parameter_labels: List[str],
    results: List[List[float]],
    chi_sqs: List[float],
    model_save_dir: Path,
) -> None:
    table = " ".join(["l"] + ["c" for _ in range(parameter_labels.size + 1)])
    labels = ["Object"] + [NoEscape(r"$\chi^2r$")] + parameter_labels.tolist()

    doc = Document()
    with doc.create(Section(" ".join(model_save_dir.name.split("_")).title())):
        with doc.create(Tabular(table)) as table:
            table.add_row(labels)
            table.add_hline()

            for source, chi_sq, result in zip(sources, chi_sqs, results):
                result = [
                    NoEscape(f"${r[0]:.2f}^{{+{r[1]:.2f}}}_{{-{r[2]:.2f}}}$")
                    for r in result
                ]
                table.add_row([source] + [f"{chi_sq:.2f}"] + result)

            table.add_hline()

    doc.generate_pdf(model_save_dir / "results", clean_tex=False)


if __name__ == "__main__":
    # data_dir = Path().home() / "Data"
    # fits_files = list((data_dir / "fitting_data" / "hd142527").glob("*.fits"))
    # observations(fits_files)

    # sed_fit_dir = data_dir / "opacities" / "silicate_labels_and_weights.npy"
    # opacities(sed_fit_dir)

    source_dir = Path().home() / "Data" / "reduced"
    df = source_info(
        source_dir,
        get_sources(source_dir / "MATISSE data overview.xlsx"),
        source_dir,
    )
