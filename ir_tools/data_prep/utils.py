import shutil
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits
from uncertainties import unumpy


# TODO: Exchange this all here with the io library that I am writing
def average_flux(fits_files: List[Path]):
    new_fits = None
    wavelengths, fluxes, errors = [], [], []
    for fits_file in fits_files:
        if new_fits is None:
            new_fits = "averaged_flux.fits"
            shutil.copy(fits_file, new_fits)

        with fits.open(fits_file) as hdul:
            wavelengths.append(hdul["oi_wavelength"].data["eff_wave"].flatten())
            fluxes.append(hdul["oi_flux"].data["fluxdata"].flatten())
            errors.append(hdul["oi_flux"].data["fluxerr"].flatten())

    wavelengths = np.array(wavelengths)
    fluxes, errors = np.array(fluxes), np.array(errors)
    fluxes[0] = np.interp(wavelengths[1], wavelengths[0], fluxes[0])
    fluxes = unumpy.umatrix(fluxes, errors)
    wavelengths, fluxes = wavelengths[1], np.nanmean(fluxes, axis=0)
    fluxes, errors = unumpy.nominal_values(fluxes), unumpy.std_devs(fluxes)

    with fits.open(new_fits, "update") as hdul:
        hdul["oi_wavelength"].data["eff_wave"] = wavelengths
        hdul["oi_flux"].data["fluxdata"] = fluxes
        hdul["oi_flux"].data["fluxerr"] = errors
        hdul.flush()
