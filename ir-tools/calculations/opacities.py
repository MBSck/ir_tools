from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod.utils import get_opacity


if __name__ == "__main__":
    data_dir = Path("/Users/scheuck/Data/opacities")
    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    fmaxs = [1.0, 1.0, 1.0, None]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

    wavelengths = {"hband": [1.7] * u.um, "kband": [2.15] * u.um,
                   "lband": np.linspace(3.3, 3.8, 5) * u.um,
                   "mband": np.linspace(4.6, 4.9, 3) * u.um,
                   "nband": np.linspace(8, 13, 35) * u.um,
                   }

    wavelength = np.concatenate((wavelengths["hband"], wavelengths["kband"],
                                wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))

    wl_boekel, opacity_boekel = get_opacity(
        data_dir, weights, sizes, names, "boekel", )
    wl_qval, opacity_qval= get_opacity(
        data_dir, weights, sizes, names, "qval", fmaxs=fmaxs, fill_value="extrapolate")

    lower_ind = np.where(wl_qval < wl_boekel[0])[0]

    wl = np.concatenate((wl_qval[lower_ind], wl_boekel), axis=0)
    opacity = np.concatenate((opacity_qval[lower_ind], opacity_boekel), axis=0)
    np.save("hd142527_boekel_qval_silicates.npy", np.array([wl, opacity]))
