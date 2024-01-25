from collections.abc import Callable
from pathlib import Path
from typing import List, Optional

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from ppdmod import utils
from ppdmod.data import ReadoutFits


def calculate_coefficient(silicate: np.ndarray,
                          continuum: np.ndarray) -> np.ndarray:
    """Calculates the coefficient for a 2/3 to 1/3 silicate to
    continuum ratio."""
    return 2*silicate/(continuum+2*silicate)

# def transform_grid():
#     utils.


def plot_combined_grains(files: List[Path],
                         weights: List[float],
                         wavelength_range: Optional[List[float]] = [1, 14],
                         load_func: Optional[Callable] = None,
                         ax: Optional = None,
                         label: Optional[str] = "qval") -> None:
    total_opacity = None
    for weight, opacity_file in zip(weights, files):
        if load_func is not None:
            wavelength, opacity = map(lambda x: x.value
                                      if isinstance(x, u.Quantity)
                                      else x, load_func(opacity_file))
        else:
            wavelength, opacity, *_ = np.loadtxt(opacity_file, unpack=True)

        indices = np.where((wavelength > wavelength_range[0])
                           & (wavelength < wavelength_range[1]))

        if total_opacity is None:
            total_opacity = opacity[indices]*weight
        else:
            total_opacity += opacity[indices]*weight

    if ax is not None:
        ax.plot(wavelength[indices], total_opacity, label=label.upper())
    else:
        plt.plot(wavelength[indices], total_opacity, label=label.upper())
        plt.legend()
        plt.show()


def plot_all_combined_grains(qval_paths, grf_paths,
                             continuum_file, weights,
                             wavelength_range: Optional[List[float]] = [1, 14]):
    wavelength, opacity = map(lambda x: x.value, utils.qval_to_opacity(continuum_file))
    indices = np.where((wavelength > wavelength_range[0])
                       & (wavelength < wavelength_range[1]))
    _, ax = plt.subplots()
    plot_combined_grains(qval_paths, weights,
                         load_func=utils.qval_to_opacity,
                         ax=ax, label="qval")
    plot_combined_grains(grf_paths, weights, ax=ax, label="grf")
    ax.plot(wavelength[indices], opacity[indices], label="Continuum")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    plt.legend()
    plt.savefig("combined_grains.pdf", format="pdf")


def plot_mass_fractions(wavelength_grid: np.ndarray,
                        grf_files: Path, qval_files: Path,
                        continuum: Path, weights: List[float]):
    opacity = utils.linearly_combine_data(
        qval_files, weights, wavelength_axes,
        load_func=utils.qval_to_opacity)
    continuum_opacity = utils.data_to_matisse_grid(
        wavelength_axes, data_file=continuum,
        load_func=utils.qval_to_opacity)
    dhs_rel = calculate_coefficient(opacity, continuum_opacity)

    grf_opacity = utils.linearly_combine_data(
        grf_files, weights, wavelength_grid)
    grf_rel = calculate_coefficient(grf_opacity, continuum_opacity)

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(10, 10))
    ax.plot(wavelength_axes, opacity, label="silicates ($A_1$)")
    ax.plot(wavelength_axes, continuum_opacity, label="continuum ($A_2$)")
    ax.set_title("Opacity (DHS)")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    ax.legend()

    bx.plot(wavelength_axes, dhs_rel, label=r"$c=\frac{2A_{1}}{A_{2}+2A_{1}}$")
    bx.set_title("Mass fraction (DHS)")
    bx.set_xlabel(r"$\lambda$ ($\mu$m)")
    bx.legend()

    cx.plot(wavelength_axes, grf_opacity, label="silicates (GRF)")
    cx.plot(wavelength_axes, continuum_opacity, label="continuum (DHS)")
    cx.set_title("Opacity")
    cx.set_xlabel(r"$\lambda$ ($\mu$m)")
    cx.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    cx.legend()

    dx.plot(wavelength_axes, grf_rel, label=r"$c=\frac{2A_{1}}{A_{2}+2A_{1}}$")
    dx.set_title("Mass fraction (GRF)")
    dx.set_xlabel(r"$\lambda$ ($\mu$m)")
    dx.legend()
    plt.savefig("mass_fractions.pdf", format="pdf")


if __name__ == "__main__":
    path = Path("/Users/scheuck/Code/modelling/ppdmod/tests/data")
    readouts = list(map(ReadoutFits, (path / "fits").glob("*fits")))
    wavelength_axes = list(map(lambda x: x.wavelength, readouts))
    wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

    grf_dir = Path("/Users/scheuck/Data/opacities/GRF")
    grf_files = ["MgOlivine0.1.Combined.Kappa",
                 "MgOlivine2.0.Combined.Kappa",
                 "MgPyroxene2.0.Combined.Kappa",
                 "Forsterite0.1.Combined.Kappa",
                 "Forsterite2.0.Combined.Kappa",
                 "Enstatite2.0.Combined.Kappa"]
    grf_paths = list(map(lambda x: grf_dir / x, grf_files))

    qval_file_dir = path / "qval"
    qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
                  "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
                  "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
                  "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
    dhs_continuum_file = qval_file_dir / "Q_amorph_c_rv0.1.dat"
    # dhs_continuum_file = qval_file_dir / "Q_iron_0.10um_dhs_0.99.dat"

    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    plot_all_combined_grains(qval_paths, grf_paths,
                             dhs_continuum_file, weights)
    plot_mass_fractions(wavelength_axes, grf_paths,
                        qval_paths, dhs_continuum_file, weights)
