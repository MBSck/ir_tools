import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits
from uncertainties import unumpy


if __name__ == "__main__":
    new_fits = None
    wavelengths, fluxes, errors = [], [], []
    for fits_file in list((Path().home() / "Data" / "fitting_data" / "hd142527").glob("*_N_*.fits")):
        if new_fits is None:
            new_fits = "hd142527_N_AVERAGE_SED.fits"
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
