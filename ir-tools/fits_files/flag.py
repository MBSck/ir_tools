import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import astropy.units as u
import numpy as np
import yaml
from astropy.io import fits
from tqdm import tqdm


def flag_gravity(fits_file: Path) -> None:
    """Flags gravity files.

    Notes
    -----
    Flags the first index for the GRAVITY data, which usually has detector issues.
    Also adds a flag column to the "oi_flux" if there is none.
    """
    with fits.open(fits_file, "update") as hdul:
        ...


def flag_wavelength_range(hdu: fits.BinTableHDU, wavelengths: List[float],
                          wavelength_ranges: List[Tuple[float, float]]) -> None:
    """Flags data in the given HDU outside specific wavelength ranges.

    Parameters
    ----------
    hdul : astropy.io.fits.BinTableHDU
    wavelenght : list of float
    wavelength_ranges : list of tuple of float
    """
    if wavelength_ranges:
        masks = [(wavelengths >= lower) & (wavelengths <= upper) for lower, upper in wavelength_ranges]
        flag = ~np.logical_or.reduce(masks) | hdu.data["flag"]
    else:
        flag = np.ones_like(hdu.data["flag"]).astype(bool)

    hdu.data["flag"] = flag


def flag_baselines(hdu: fits.BinTableHDU, baselines_to_flag):
    """
    Flags specific baselines in the given HDU.
    """
    if "flag" in hdu.columns.names:
        # TODO: Find a good way to do this
        # baseline_ids = hdu.data["BASELINE_ID"]
        # mask = np.isin(baseline_ids, baselines_to_flag)
        # hdu.data["flag"][mask] = True
        ...

# TODO: Finish this (especially relevant to one of the N band files)
def remove_flag(hdu: fits.BinTableHDU):
    ...


def flag_oifits(fits_file: Path, flagging_rules: Dict, save_dir: Path) -> None:
    """Flags a fits file for a certain wavelength range or baselines.

    Parameters
    ----------
    fits_file : pathlib.Path
    extensions : list of str
        The extensions to be flagged.
    wavelenght_range : list of float
        A wavelength range to be flagged.
    """
    flagged_fits = save_dir / fits_file.name
    shutil.copy(fits_file, flagged_fits)
    with fits.open(flagged_fits, mode="update") as hdul:
        wavelengths = (hdul["oi_wavelength"].data["eff_wave"] * u.m).to(u.um).value
        for extension, rules in flagging_rules.items():
            for hdu in hdul:
                if hdu.header.get("extname") == extension.upper():
                    if "keep_wavelengths" in rules:
                        flag_wavelength_range(hdu, wavelengths, rules["keep_wavelengths"])
                    if "baselines" in rules:
                        flag_baselines(hdu, rules["baselines"])

        hdul.flush()


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting" / "hd142527"
    save_dir = path / "flagged"
    # TODO: Check if this works
    with open(save_dir / "flagging.yaml", "r") as f:
        data = yaml.safe_load(f)

    for fits_file, flagging_rules in tqdm(data.items(), desc="Flagging files..."):
        flag_oifits((path / fits_file).with_suffix(".fits"), flagging_rules, save_dir)

