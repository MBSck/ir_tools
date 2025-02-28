import shutil
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits
from astropy.stats import biweight_location, biweight_midvariance


# TODO: For io use my io script
def average_flux(
    fits_files: Path | List[Path],
    method: str = "biweight",
    save_dir: Path | None = None,
) -> None:
    """Averages the "oi_flux" extension of a single fits_file or a list of fits files.

    Parameters
    ----------
    fits_files : pathlib.Path or list of pathlib.Path
    method : str, optional
    save_dir : pathlib.Path, optional

    Notes
    -----
    If a single file is given, it will average all the fluxes in it to return one
    """
    if not isinstance(fits_files, (list, tuple, np.ndarray)):
        fits_files = [fits_files]

    if method == "biweight":
        avgfunc, avgvar = biweight_location, biweight_midvariance
    elif method == "mean":
        avgfunc = np.nanmean
    else:
        avgfunc = np.nanmedian

    flux, fluxerr = [], []
    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            flux.append(hdul["oi_flux"].data["fluxdata"])
            fluxerr.append(hdul["oi_flux"].data["fluxerr"])

    flux, fluxerr = np.array(flux), np.array(fluxerr)
    if len(fits_files) <= 1:
        flux, fluxerr = flux[0], fluxerr[0]

    avg_flux = avgfunc(flux, axis=0)
    avg_fluxerr = np.sqrt(avgvar(flux, axis=0) + np.nanmean(fluxerr, axis=0)**2.0)

    if save_dir:
        shutil.copy(fits_files[0], save_dir)
        with fits.open(save_dir, mode="update") as outhdul:
            outhdul["oi_flux"].data = outhdul["oi_flux"].data[0:1]
            outhdul["oi_flux"].data["fluxdata"] = avg_flux
            outhdul["oi_flux"].data["fluxerr"] = avg_fluxerr
            outhdul.flush()


if __name__ == "__main__":
    data_dir = Path().home() / "Data" / "reduced" / "HD_142527" / "matisse"
    fits_file = list(data_dir.glob("*.fits"))[0]
    average_flux(fits_file, save_dir=data_dir / f"{fits_file.stem}_AVG_FLUX.fits")
