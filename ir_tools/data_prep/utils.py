from pathlib import Path
from typing import List

import numpy as np
from astropy.stats import biweight_location


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
        avgfunc = biweight_location
    elif method == "mean":
        avgfunc = np.nanmean
    else:
        avgfunc = np.nanmedian

    flux, flux_err = [], []
    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            flux.append(hdul["oi_flux"].data["fluxdata"])
            flux_err.append(hdul["oi_flux"].data["fluxerr"])

    flux, flux_err = np.array(flux), np.array(flux_err)

    if len(fits_files) > 1:
        avg_flux = avgfunc(flux, axis=0)
    else:
        avg_flux = avgfunc(flux)

    # TODO: Make sure the error calculation here is correct
    # if len(flux_arr) > 3:
    #     avg_fluxerr = np.sqrt(
    #         np.nanstd(flux_arr, axis=0) ** 2.0 + np.nanmean(fluxerr_arr, axis=0) ** 2.0
    #     )
    # else:
    #     avg_fluxerr = np.nanmean(fluxerr_arr, axis=0)

    # TODO: Check how to save the total flux properly as only one dataset
    # outhdul["OI_FLUX"].data = outhdul["OI_FLUX"].data[0:1]
    # outhdul["OI_FLUX"].data["FLUXDATA"] = avg_flux
    # outhdul["OI_FLUX"].data["FLUXERR"] = avg_fluxerr


if __name__ == "__main__":
    ...
