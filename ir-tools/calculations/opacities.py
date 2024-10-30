from pathlib import Path
from typing import Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from ppdmod.utils import get_opacity, load_data, qval_to_opacity
from astropy.modeling.models import BlackBody

NAMES = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]


def interpolation_smooth(
    grid: np.ndarray, value: np.ndarray, indices: Tuple[int, int]
) -> np.ndarray:
    return np.interp(
        grid[slice(*indices)],
        [grid[indices[0]], grid[indices[-1]]],
        [value[indices[0]], value[indices[-1]]],
    )


def generate_opacity_file(
    data_dir: Path,
    weights: np.ndarray,
    method: str = "grf",
    fits_file: Path | None = None,
    best_fit_file: Path | None = None,
    continuum_data: Tuple[np.ndarray, np.ndarray] | None = None,
    do_plot: bool = False,
) -> None:
    """
    Generate opacity file for silicate grains.

    If a fits file is provided, the continuum opacity is calculated as well.

    Parameters
    ----------
    data_dir : Path
        Directory containing the data.
    weights : np.ndarray
        Weights for the different silicate species.
    method : str, optional
        Method to use for the opacity calculation. Default is "grf".
    fits_file : Path, optional
        Fits file containing the flux data. Default is None.
    best_fit_file : Path, optional
        Best fit file containing the best fit parameters. Default is None.
    continuum_opacity : tuple of np.ndarrays, optional
        The continuum wavelength and opacity. Default is None.
    do_plot : bool, optional
        Whether to plot the opacity. Default is False.
    """
    wl, opacity = get_opacity(data_dir / "opacities", weights, NAMES, method)

    if fits_file is not None:
        _, opacity_normed = get_opacity(data_dir / "opacities", weights / weights.sum(), NAMES, method)
        opacity_scaling_factor = (opacity_normed / opacity)[0]
        grid_continuum, opacity_continuum = continuum_data
        labels, params = np.load(best_fit_file)
        params = params.astype(float)

        characteristic_temperature = params[np.where(labels == "tempc")][0]
        continuum_weight = params[np.where(labels == "weight_cont")][0] / 1e2
        scaling_factor = params[np.where(labels == "f")][0]

        with fits.open(fits_file) as hdul:
            wl_flux = (hdul["OI_WAVELENGTH"].data["EFF_WAVE"] * u.m).to(u.um).value
            ind = np.where((wl >= wl_flux[0]) & (wl <= wl_flux[-1]))[0]
            flux = hdul["OI_FLUX"].data["FLUXDATA"][0]

        flux_scaling = (
            10.0**-scaling_factor
            * BlackBody(characteristic_temperature * u.K)(wl[ind] * u.um)
            * u.sr
        ).to(u.Jy)
        flux = np.interp(wl[ind], wl_flux, flux)
        continuum_opacity = (
            np.interp(wl[ind], grid_continuum, opacity_continuum) * continuum_weight
        )
        opacity[ind] = (
            flux - continuum_opacity * flux_scaling.value
        ) / flux_scaling.value

        lower, upper = (ind[0] - 1, ind[0] + 4), (ind[-1] - 3, ind[-1] + 2)
        opacity[slice(*lower)] = interpolation_smooth(wl, opacity, lower)
        opacity[slice(*upper)] = interpolation_smooth(wl, opacity, upper)
        opacity *= opacity_scaling_factor

    if do_plot:
        empirical = "_empirical" if fits_file is not None else ""

        plt.plot(wl, opacity)
        plt.xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        plt.ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$)")
        plt.xlim([7, 15])
        plt.ylim([None, 3500])
        if empirical:
            plt.ylim([None, 3000])

        plt.savefig(f"silicate_opacities_{method}{empirical}.pdf", format="pdf")
        plt.close()

    np.save(
        data_dir / "opacities" / f"hd142527_silicate_{method}_opacities.npy",
        np.array([wl, opacity]),
    )


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    fits_file = list(
        (data_dir / "fitting_data" / "hd142527" / "sed_fit" / "averaged").glob("*")
    )[0]

    method = "grf"
    if method == "boekel":
        weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0])
    else:
        weights = np.load(data_dir / "opacities" / "silicate_labels_and_weights.npy")

    weights /= 1e2
    continuum_data = load_data(
        data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat",
        load_func=qval_to_opacity,
    )

    # generate_opacity_file(data_dir, weights / weights.sum(), method=method, do_plot=True)
    generate_opacity_file(
        data_dir,
        weights,
        method=method,
        fits_file=fits_file,
        best_fit_file=data_dir / "opacities" / "best_fit.npy",
        continuum_data=continuum_data,
        do_plot=True,
    )
