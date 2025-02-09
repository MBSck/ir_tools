import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from astropy.io import fits
from tqdm import tqdm
from uncertainties import unumpy


def correct_vis_to_vis2(fits_file: Path) -> None:
    """The Varga reduction save "OI_VIS" in "OI_VIS2", this corrects it
    to be "OI_VIS2" to be oimodeler compliant."""
    with fits.open(fits_file, mode="update") as hdul:
        for card in hdul:
            header = card.header
            if "EXTNAME" not in header:
                continue

            if header["EXTNAME"] == "OI_VIS2":
                vis = unumpy.uarray(card.data["VIS2DATA"], np.abs(card.data["VIS2ERR"]))
                card.data["VIS2DATA"] = unumpy.nominal_values(vis**2)
                card.data["VIS2ERR"] = unumpy.std_devs(vis**2)

        hdul.flush()


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


# TODO: Add here the functionality to automatically make an exel sheet out of this
def sort_target(sheet: Path, source: str, source_dir: Path, target_dir: Path) -> None:
    """Sorts the data for a target.

    Parameters
    ----------
    sheet : Path
        The excel file containing the information about the observations.
    source : str
        The source for which to sort the data.
    source_dir : Path
        The directory containing the directories with the targets.
    target_dir : Path
        The directory where to save the sorted data.
    """
    dir_name = get_dir_name(source)
    target_dir /= dir_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    for quality in ["bad", "medium", "good", "tba"]:
        quality_dir = target_dir / quality
        quality_dir.mkdir(parents=True, exist_ok=True)

    df = get_source(sheet, source)
    nights = df["Night"].iloc[:, 0].apply(lambda x: x.strftime("%Y-%m-%d")).to_dict()
    reversed_nights = {v: k for k, v in nights.items()}

    for fits_file in list((source_dir / dir_name).glob("*.fits")):
        fits_name = fits_file.name
        date = re.search(r"\d{4}-\d{2}-\d{2}", fits_name)
        if date is None:
            continue

        key = get_date(date.group(), nights)
        new_fits_file = target_dir / "tba" / fits_name
        if key is not None:
            row = df.loc[reversed_nights[key]]
            config = row["Data description"]["Tel. config."]
            band = "L" if any(letter == "L" for letter in fits_name.split("_")) else "N"

            if band == "N" and "at" in config.lower():
                new_fits_file = target_dir / "tba" / fits_name
            else:
                quality = str(row["Data quality"][f"{band} band"])
                if "good" in quality:
                    quality = "good"
                elif "so so" in quality:
                    quality = "medium"
                elif "bad" in quality:
                    quality = "bad"
                else:
                    quality = "tba"

                new_fits_file = target_dir / quality / fits_name

        shutil.copy(fits_file, new_fits_file)
        correct_vis_to_vis2(new_fits_file)


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    excel_file = data_dir / "survey" / "MATISSE data overview.xlsx"
    source_dir = data_dir / "reduced_data" / "jozsef_reductions" / "targets5"
    target_dir = data_dir / "survey"
    sources = get_sources(excel_file)

    for target in tqdm(sources, "Sorting data..."):
        sort_target(excel_file, target, source_dir, target_dir)
