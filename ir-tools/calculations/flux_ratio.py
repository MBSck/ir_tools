from typing import List
from pathlib import Path

import astropy.units as u
import astropy.constants as const
import numpy as np
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.utils import load_data


def get_star_observables(fits_files: List[Path], flux_file: Path) -> np.ndarray:
    """Gets the star's flux interpolated to the provided wavelength grid
    from the (.fits)-files as well as the flux ratio.
    """
    wl, star_flux = load_data(flux_file)
    data = set_data(fits_files, wavelengths="all", wavelength_range=[2.8, 5]*u.um, fit_data=["flux"])
    wavelengths, total_flux = get_all_wavelengths(), data.flux.value.squeeze()
    star_flux = np.interp(wavelengths.value, wl, star_flux)
    return wavelengths.value, star_flux, star_flux / total_flux

    
if __name__ == "__main__":
    fits_files = list(Path("/Users/scheuck/Data/fitting_data/hd142527").glob("*2021-03-11*"))
    flux_file = Path("/Users/scheuck/Data/flux_data/hd142527/HD142527_stellar_model.txt")
    wl, flux, flux_ratio = get_star_observables(fits_files, flux_file)
    nu = (const.c / (wl*u.um).to(u.m)).to(u.Hz).value
    k = np.gradient(np.log(flux), np.log(nu))
    np.save("hd142527_flux_observables.npy", np.array([wl, flux_ratio, k]))
