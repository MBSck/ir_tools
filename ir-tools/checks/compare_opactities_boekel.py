from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from ppdmod.utils import get_opacity


def comparison_boekel_vs_qval(path: Path) -> None:
    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    fmaxs = [1.0, 1.0, 1.0, None]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]
    wl_roy, opacity_roy = get_opacity(path, weights, sizes, names, "boekel")
    wl_qval, opacity_qval = get_opacity(path, weights, sizes, names, "qval", fmaxs=fmaxs)
    plt.plot(wl_roy, opacity_roy, label="boekel")
    plt.plot(wl_qval, opacity_qval, label="qval")
    plt.legend()
    plt.xlim([6, 15])
    plt.title("HD 142527 - Combined Silicates - Boekel vs QVAL")
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\kappa_\mathrm{abs}$ $\left(\frac{cm^2}{g}\right)$")
    plt.savefig("comparison_boekel_vs_qval.pdf", format="pdf")
    plt.close()


def comparison_boekel_vs_qval_vs_grf(path: Path) -> None:
    weights = np.array([73.2, 8.6, 0.6, 14.2]) / 100
    names = ["pyroxene", "forsterite", "enstatite"]
    fmaxs = [1.0, 1.0, 1.0]
    sizes = [[1.5], [0.1], [0.1, 1.5]]
    wl_roy, opacity_roy = get_opacity(path, weights, sizes, names, "boekel")
    wl_qval, opacity_qval = get_opacity(path, weights, sizes, names, "qval", fmaxs=fmaxs)
    wl_grf, opacity_grf = get_opacity(path, weights, sizes, names, "grf")

    plt.plot(wl_roy, opacity_roy, label="boekel")
    plt.plot(wl_qval, opacity_qval, label="qval")
    plt.plot(wl_grf, opacity_grf, label="grf")
    plt.legend()
    plt.xlim([6, 15])
    plt.title("HD 142527 - Combined Silicates (No Silica) - Boekel vs QVAL vs GRF")
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\kappa_\mathrm{abs}$ $\left(\frac{cm^2}{g}\right)$")
    plt.savefig("comparison_boekel_vs_qval_vs_grf.pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/opacities")
    comparison_boekel_vs_qval(path)
    comparison_boekel_vs_qval_vs_grf(path)
