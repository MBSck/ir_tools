from pathlib import Path

import numpy as np
from astropy.io import fits



def print_relative_flux(fits_file: Path) -> None:
    """Plots the model information stored in the (.fits)-file.

    Parameters
    ----------
    fits_file : pathlib.Path
    pixel_size : float, optional
    """
    with fits.open(fits_file) as hdul:
        wavelengths = hdul["STAR"].data["wavelength"]
        star_flux = hdul["STAR"].data["flux"]
        inner_disk_flux = hdul["INNER_RING"].data["flux"]
        outer_disk_flux = hdul["OUTER_RING"].data["flux"]
        total_flux = star_flux + inner_disk_flux + outer_disk_flux
        relative_flux_star = np.round(star_flux / total_flux, 2)*100
        relative_flux_outer = np.round(outer_disk_flux / total_flux, 2)*100
        relative_flux_inner = np.round(inner_disk_flux / total_flux, 2)*100
        print(f"Wavelengths: {wavelengths}")
        print(f"Relative flux star: {relative_flux_star}")
        print(f"Relative flux inner: {relative_flux_inner}")
        print(f"Relative flux outer: {relative_flux_outer}")


if __name__ == "__main__":
    model_file = Path("/Users/scheuck/Pictures/Meetings/thomas_and_roy/2024-01-12/nsteps10000_nw100_hklband/model.fits")
    print_relative_flux(model_file)
