from pathlib import Path
from typing import List

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


def plot_mass_fractions(wavelength_grid: np.ndarray,
                        grf_files: Path, qval_files: Path,
                        continuum: Path, weights: List[float]):
    wl_cont, op_cont = utils.load_data(continuum, load_func=utils.qval_to_opacity)
    op_cont = np.interp(wavelength_grid, wl_cont*u.um, op_cont)

    wl_dhs, op_dhs = utils.load_data(qval_files, load_func=utils.qval_to_opacity)
    op_dhs = utils.linearly_combine_data(op_dhs, weights)
    op_dhs = np.interp(wavelength_grid, wl_dhs[0]*u.um, op_dhs)

    wl_grf, op_grf = utils.load_data(grf_files)
    op_grf = utils.linearly_combine_data(op_grf, weights)
    op_grf = np.interp(wavelength_grid, wl_grf[0]*u.um, op_grf)

    dhs_rel = calculate_coefficient(op_dhs, op_cont)
    grf_rel = calculate_coefficient(op_grf, op_cont)

    _, ((ax, bx), (cx, dx)) = plt.subplots(2, 2, figsize=(10, 10))
    ax.plot(wavelength_axes, op_dhs, label="silicates ($A_1$)")
    ax.plot(wavelength_axes, op_cont, label="continuum ($A_2$)")
    ax.set_title("Opacity (DHS)")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    ax.legend()

    bx.plot(wavelength_axes, dhs_rel, label=r"$c=\frac{2A_{1}}{A_{2}+2A_{1}}$")
    bx.set_title("Mass fraction (DHS)")
    bx.set_xlabel(r"$\lambda$ ($\mu$m)")
    bx.legend()

    cx.plot(wavelength_axes, op_grf, label="silicates")
    cx.plot(wavelength_axes, op_cont, label="continuum (DHS)")
    cx.set_title("Opacity (GRF)")
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
    grf_files = list(map(lambda x: grf_dir / x, grf_files))

    qval_file_dir = path / "qval"
    qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
                  "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
                  "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
                  "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
                  "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
    qval_files = list(map(lambda x: qval_file_dir / x, qval_files))
    dhs_continuum_file = qval_file_dir / "Q_amorph_c_rv0.1.dat"

    weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
    plot_mass_fractions(wavelength_axes, grf_files,
                        qval_files, dhs_continuum_file, weights)
