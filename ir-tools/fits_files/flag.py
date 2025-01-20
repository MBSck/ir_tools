import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import astropy.units as u
import numpy as np
import yaml
from astropy.table import Table
from astropy.io import fits
from tqdm import tqdm


def flag_gravity_tracker(fits_file: Path, save_dir: Path) -> None:
    """Flags a GRAVITY file to remove the first tracker (index=20) wavelength."""
    flagged_fits = save_dir / fits_file.name
    shutil.copy(fits_file, flagged_fits)
    with fits.open(flagged_fits, "update") as hdul:
        for key in ["oi_flux", "oi_vis", "oi_vis2", "oi_t3"]:
            if key in hdul:
                if "FLAG" in hdul[key, 20].columns.names:
                    hdul[key, 20].data["flag"][:, 0] = True
                else:
                    sub_key = "flux" if "FLUX" in hdul[key, 20].columns.names else "fluxdata"
                    flag = np.zeros_like(hdul[key, 20].data[sub_key]).astype(bool)
                    flag[:, 0] = True
                    table = Table(hdul[key, 20].data)
                    table["FLAG"] = flag

                    new_hdu = fits.BinTableHDU(data=table, name=hdul[key, 20].name)
                    for label, value in hdul[key, 20].header.items():
                        if label.upper() not in new_hdu.header:
                            new_hdu.header[label] = value

                    hdul[key, 20] = new_hdu

        hdul.flush()


def flag_wavelength_range(hdu: fits.BinTableHDU, wavelengths: List[float],
                          flag_ranges: List[Tuple[float, float]], reflag: bool = False) -> None:
    """Flags data in the given HDU outside specific wavelength ranges.

    Parameters
    ----------
    hdul : astropy.io.fits.BinTableHDU
    wavelenght : list of float
    flag_ranges : list of tuple of float
    reflag : bool, optional
    """
    if flag_ranges:
        masks = [(wavelengths >= lower) & (wavelengths <= upper) for lower, upper in flag_ranges]
        previous_flag = np.zeros_like(hdu.data["flag"]).astype(bool) if reflag else hdu.data["flag"]
        flag = ~np.logical_or.reduce(masks) | previous_flag
    else:
        flag = np.ones_like(hdu.data["flag"]).astype(bool)

    hdu.data["flag"] = flag


def flag_baselines(hdu: fits.BinTableHDU,
                   array_to_sta: Dict[int, str],
                   baselines_to_flag: List[str]) -> None:
    """Flags specific baselines in the given HDU.

    Parameters
    ----------
    hdu : astropy.io.fits.BinTableHDU
    baselines_to_flag : list of str
    """
    sta_names = np.vectorize(lambda x: str(array_to_sta.get(x)))(hdu.data["sta_index"])
    for index, baseline in enumerate(["-".join(arr) for arr in sta_names]):
        if baseline in baselines_to_flag:
            hdu.data[index]["flag"] = np.ones_like(hdu.data[index]["flag"])


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
                        reflag = rules["reflag"] if "reflag" in rules else False
                        flag_wavelength_range(hdu, wavelengths, rules["keep_wavelengths"], reflag)

                    if "baselines" in rules:
                        array = hdul["oi_array"].data
                        array_to_sta = dict(zip(map(int, array["sta_index"]), array["sta_name"]))
                        flag_baselines(hdu, array_to_sta, rules["baselines"])

        hdul.flush()


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting" / "hd142527"
    save_dir = path / "flagged"

    for fits_file in tqdm(list(path.glob("GRAV*")), desc="Flagging GRAVITY..."):
        flag_gravity_tracker(fits_file, save_dir)

    with open(save_dir / "flagging.yaml", "r") as f:
        data = yaml.safe_load(f)

    for fits_file, flagging_rules in tqdm(data.items(), desc="Flagging MATISSE..."):
        flag_oifits((path / fits_file).with_suffix(".fits"), flagging_rules, save_dir)

    with open(save_dir / "flagging_downsampled.yaml", "r") as f:
        data = yaml.safe_load(f)

    for fits_file, flagging_rules in tqdm(data.items(), desc="Flagging MATISSE downsampled..."):
        flag_oifits((path / "downsampled" / fits_file).with_suffix(".fits"), flagging_rules, save_dir)


