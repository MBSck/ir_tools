from pathlib import Path
from typing import Tuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.modeling.models import BlackBody
from ppdmod.utils import get_opacity, load_data, qval_to_opacity

NAMES = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]


def plot_sed(
    data_dir: Path,
    source_dir: Path,
    flux_data: Tuple[np.ndarray, np.ndarray],
    continuum_data: Tuple[np.ndarray, np.ndarray],
    savefig: str | None = None,
    method: str = "grf",
) -> None:
    labels, params = np.load(source_dir / "best_sed_fit.npy")
    params = params.astype(float)

    tempc = params[np.where(labels == "tempc")][0]
    weight_cont = params[np.where(labels == "weight_cont")][0] / 1e2
    offset = params[np.where(labels == "f")][0]
    indices = list(
        map(
            labels.tolist().index,
            filter(lambda x: "weight" in x and "cont" not in x, labels),
        )
    )

    weights = params[indices] / 1e2
    wl_cont, op_cont = continuum_data
    wl_sil, op_sil = get_opacity(data_dir / "opacities", weights, NAMES, method)
    op_sil = np.interp(wl_cont, wl_sil, op_sil)

    wl_flux, flux = flux_data
    bb = (BlackBody(tempc * u.K)(wl_cont * u.um) * 10.0**-offset * u.sr).to(u.Jy)
    flux_cont = bb * op_cont * weight_cont
    flux_model = bb * (op_sil + op_cont * weight_cont)

    _, ax = plt.subplots()
    ax.plot(wl_flux, flux, label="Data")
    ax.plot(wl_cont, flux_model, label="Model")
    ax.plot(wl_cont, flux_cont, label="Continuum")
    ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    ax.set_ylabel(r"$F_{\nu}$ (Jy)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim([0.1, 100])
    ax.set_ylim([0.1, 100])
    ax.legend()

    if savefig is not None:
        plt.savefig(savefig, format="pdf")
    else:
        plt.show()

    plt.close()


def interpolation_smooth(
    grid: np.ndarray, value: np.ndarray, indices: Tuple[int, int]
) -> np.ndarray:
    return np.interp(
        grid[slice(*indices)],
        [grid[indices[0]], grid[indices[-1]]],
        [value[indices[0]], value[indices[-1]]],
    )


def calculate_opacity(
    data_dir: Path,
    weights: np.ndarray,
    method: str = "grf",
    best_fit_file: Path | None = None,
    continuum_data: Tuple[np.ndarray, np.ndarray] | None = None,
    flux_data: Tuple[np.ndarray, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
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
    best_fit_file : Path, optional
        Best fit file containing the best fit parameters. Default is None.
    continuum_data : tuple of np.ndarrays, optional
        The continuum wavelength and opacity. Default is None.
    flux_data : Path, optional
        The flux wavelength and flux data. Default is None.
    """
    wl, op_sil = get_opacity(data_dir / "opacities", weights, NAMES, method)

    if flux_data is not None:
        wl_cont, op_cont = continuum_data
        wl_flux, flux = flux_data
        ind = np.where((wl >= wl_flux[0]) & (wl <= wl_flux[-1]))[0]

        labels, params = np.load(best_fit_file)
        params = params.astype(float)

        tempc = params[np.where(labels == "tempc")][0]
        weight_cont = params[np.where(labels == "weight_cont")][0] / 1e2
        offset = params[np.where(labels == "f")][0]

        flux_scaling = (
            (10.0**-offset * BlackBody(tempc * u.K)(wl[ind] * u.um) * u.sr)
            .to(u.Jy)
            .value
        )
        flux = np.interp(wl[ind], wl_flux, flux)
        op_cont = np.interp(wl[ind], wl_cont, op_cont) * weight_cont
        op_sil[ind] = flux / flux_scaling - op_cont

    return wl, op_sil


# TODO: Fix the calculation of the opacity at its edges
if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    plot_dir = Path("plots")
    source_dir = data_dir / "model_results" / "hd142527"
    fits_file = list(
        (data_dir / "fitting_data" / "hd142527" / "sed_fit" / "averaged").glob("*")
    )[0]

    with fits.open(fits_file) as hdul:
        wl_flux = (hdul["OI_WAVELENGTH"].data["EFF_WAVE"] * u.m).to(u.um).value
        flux = hdul["OI_FLUX"].data["FLUXDATA"][0]
        flux_data = wl_flux, flux

    method = "grf"
    if method == "boekel":
        weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0])
    else:
        weights = np.load(source_dir / f"silicate_weights_{method}.npy")

    weights /= 1e2

    continuum_data = load_data(
        data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat",
        load_func=qval_to_opacity,
    )

    plot_sed(
        data_dir,
        source_dir,
        flux_data,
        continuum_data,
        method=method,
        savefig=plot_dir / "nband_fit.pdf",
    )

    _, op_sil_norm = calculate_opacity(data_dir, weights / weights.sum(), method=method)
    _, op_sil = calculate_opacity(data_dir, weights, method=method)
    opacity_scale = max(op_sil_norm / op_sil)

    wl, op_new_sil = calculate_opacity(
        data_dir,
        weights,
        method=method,
        best_fit_file=source_dir / "best_sed_fit.npy",
        flux_data=flux_data,
        continuum_data=continuum_data,
    )

    fig1, ax = plt.subplots()
    ax.plot(wl, op_sil * opacity_scale, label="From fit")
    ax.plot(wl, op_new_sil * opacity_scale, label="From data.")
    ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    ax.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$)")
    ax.set_xlim([7.5, 13.15])
    ax.set_ylim([0, 3000])
    ax.legend()
    plt.savefig(plot_dir / "op_sil_comparison.pdf", format="pdf")

    labels, params = np.load(source_dir / "best_sed_fit.npy")
    np.save(
        source_dir / f"silicate_{method}_opacities.npy",
        np.array([wl, op_new_sil * opacity_scale]),
    )
