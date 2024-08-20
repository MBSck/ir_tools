from pathlib import Path

import numpy as np
from ppdmod.utils import get_opacity


if __name__ == "__main__":
    data_dir = Path("/Users/scheuck/Data/opacities")
    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    fmaxs = [1.0, 1.0, 1.0, None]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

    wl_boekel, opacity_boekel = get_opacity(
        data_dir, weights, sizes, names, "boekel")
    wl_qval, opacity_qval= get_opacity(
        data_dir, weights, sizes, names, "qval", fmaxs=fmaxs)
    ind = np.where(wl_qval < wl_boekel[0])[0]

    wl = np.concatenate((wl_qval[ind], wl_boekel), axis=0)
    opacity = np.concatenate((opacity_qval[ind], opacity_boekel), axis=0)
    np.save("hd142527_boekel_qval_silicates.npy", np.array([wl, opacity]))
