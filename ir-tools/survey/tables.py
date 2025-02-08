from datetime import datetime
from itertools import permutations
from pathlib import Path
from typing import Dict, List

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from pylatex import Document, NoEscape, Section, Tabular

from .query import query
from .sort_data import get_dir_name, get_sources


def add_array_config(
    dictionary: Dict[str, str], key: str, value: str
) -> Dict[str, str]:
    """Adds all permutations of the configuration to the dictionary."""
    perms = map(lambda x: "-".join(x), permutations(key.split("-")))
    perms = ["".join(perm.split("-")) for perm in perms]
    return {**dictionary, **{perm: value for perm in perms}}


ARRAY_CONFIGS = {}
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B2-D0-C1", "AT small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-B2-C1-D0", "AT small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "K0-G2-D0-J3", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-H0-G1-I1", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-G2-J3-K0", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-D0-J3", "AT medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G2-J2-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-K0", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-I1", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-J3", "AT large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-J2-J6", "AT extended")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "U1-U2-U3-U4", "UTs")


# TODO: Make a sorting here for the chopped then non-chopped data, etc.
# Also reimplement the arrays (but from the header this time).
def source_info(data_dir: Path, sources: List[str]) -> None:
    keys = [
        "source",
        "tpl_start",
        "year",
        "date",
        "sci_seeing_start",
        "sci_seeing_end",
        "sci_tau0_start",
        "sci_tau0_end",
        "cal_seeing",
        "cal_tau0",
        "array",
        "band",
        "chopped",
        "pipe_version",
        "cal_name",
        "cal_ra",
        "cal_dec",
        "cal_diam",
        "cal_diam_err",
        "cal_jdsc",
    ]
    dfs = []
    sources_sorted = query(sources).sort_values(by="RA")["source"].tolist()
    for source in sources_sorted:
        info = {key: [] for key in keys}
        for fits_file in list((data_dir / get_dir_name(source)).glob("*.fits")):
            with fits.open(fits_file, "readonly") as hdul:
                header = hdul[0].header

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

            sci_tau0_start = np.round(header["HIERARCH ESO ISS AMBI TAU0 START"] * 1e3, 1)
            sci_tau0_end = np.round(header["HIERARCH ESO ISS AMBI TAU0 END"] * 1e3, 1)
            info["sci_tau0_start"].append(sci_tau0_start)
            info["sci_tau0_end"].append(sci_tau0_end)

            cal_name, cal_ra, cal_dec = "", 0, 0
            cal_diam, cal_diam_err, cal_seeing = "", "", ""
            cal_tau0, cal_jsdc = "", False
            if "HIERARCH ESO PRO CAL NAME" in header:
                cal_name = header["HIERARCH ESO PRO CAL NAME"]
                cal_ra = header["HIERARCH ESO PRO CAL RA"]
                cal_dec = header["HIERARCH ESO PRO CAL DEC"]
                cal_diam = header["HIERARCH ESO PRO CAL DB DIAM"]
                cal_diam_err = header["HIERARCH ESO PRO CAL DB ERRDIAM"]
                cal_seeing = np.round(header["HIERARCH ESO PRO CAL FWHM"], 1)
                cal_tau0 = np.round(header["HIERARCH ESO PRO CAL TAU0"] * 1e3, 1)
                cal_jsdc = (
                    header["HIERARCH ESO PRO CAL DB DBNAME"] == "calib_spec_db_v10.fits"
                )

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
            info["cal_jdsc"].append(cal_jsdc)

            # config = re.search(r"[A-Z]\d[A-Z]\d[A-Z]\d[A-Z]\d", fits_file.name)
            # config = ARRAY_CONFIGS[config.group()] if config else "other"
            info["array"].append("")
            info["band"].append(header["HIERARCH ESO DET NAME"].split("-")[1])
            info["chopped"].append("nochop" in fits_file.name.lower())

        source_df = pd.DataFrame(info)
        source_df["tpl_start"] = pd.to_datetime(source_df["tpl_start"])
        dfs.append(source_df.sort_values(by="tpl_start"))

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
    df["tpl_start"] = df["tpl_start"].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df.to_excel("source_tpl_table.xlsx", index=False)


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
    data_dir = Path().home() / "Data"
    excel_file = data_dir / "survey" / "MATISSE data overview.xlsx"
    source_dir = data_dir / "reduced_data" / "jozsef_reductions" / "targets5"
    df = source_info(
        data_dir / "reduced" / "targets6",
        get_sources(excel_file),
    )
