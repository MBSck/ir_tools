from typing import List
from pathlib import Path

import astropy.units as u
import astropy.constants as const
import numpy as np
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.utils import load_data


def get_flux_ratio(fits_files: List[Path], flux_file: Path) -> np.ndarray:
    """Gets the star's flux interpolated to the provided wavelength grid
    from the (.fits)-files as well as the flux ratio.
    """
    wl, star_flux = load_data(flux_file)
    data = set_data(fits_files, wavelengths="all", wavelength_range=[2.8, 5]*u.um, fit_data=["flux"])
    wavelengths, total_flux = get_all_wavelengths(), data.flux.value.squeeze()
    star_flux = np.interp(wavelengths.value, wl, star_flux)
    return wavelengths.value, star_flux / total_flux

    
# TODO: Get high resolution file to do this (UT in best case)
def get_photometric_slope(fits_files: List[Path], flux_file: Path) -> np.ndarray:
    set_data(fits_files, wavelengths="all", wavelength_range=[2.8, 5]*u.um, fit_data=["flux"])
    wavelengths = get_all_wavelengths()
    wl, star_flux = load_data(flux_file)
    star_flux = np.interp(wavelengths.value, wl, star_flux)
    nu = (const.c / wavelengths.to(u.m)).to(u.Hz).value
    return wavelengths.value, np.gradient(np.log(star_flux), np.log(nu))

    
if __name__ == "__main__":
    fits_files = list(Path("/Users/scheuck/Data/fitting_data/hd142527").glob("*21-03-27*HAW*"))
    flux_file = Path("/Users/scheuck/Data/flux_data/hd142527/HD142527_stellar_model.txt")
    wl, flux_ratio = get_flux_ratio(fits_files, flux_file)
    np.save("hd142527_flux_ratio.npy", np.array([wl, flux_ratio]))
    
    wl, ks = get_photometric_slope(fits_files, flux_file)
    np.save("hd142527_slope.npy", np.array([wl, ks]))
