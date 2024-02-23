from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import BlackBody
from astropy.visualization import quantity_support



if __name__ == "__main__":
    flux_dir = Path("/Users/scheuck/Data/flux_data/hd142666/")
    wl, flux = np.array([]), np.array([])
    for flux_file in ["HD_142666_sws.txt", "HD_142666_timmi2.txt"]:
        data = np.loadtxt(flux_dir / flux_file, unpack=True)
        wl, flux = np.concatenate((wl, data[0])), np.concatenate((flux, data[1]))
    wl, flux = wl * u.um, flux * u.Jy
    quantity_support()  
    plt.plot(wl, flux)
    plt.show()
    # bb = BlackBody(1500*u.K)(wl)
    breakpoint()
