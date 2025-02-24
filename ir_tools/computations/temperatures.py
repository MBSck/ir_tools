import concurrent.futures
import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import List

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import ppdmod
from astropy.modeling.models import BlackBody
from ppdmod.utils import load_data, qval_to_opacity
from scipy.optimize import root_scalar
from tqdm import tqdm

np.seterr(over="ignore", divide="ignore")


def p_out(temperature: float, nu: u.Hz, opacity: u.cm**2 / u.g) -> np.ndarray:
    """Computes the outgoing radiation of a dust grain.

    Parameters
    ----------
    temperature : float
        The temperature of the dust grain.
    nu : astropy.units.Hz
        The frequency of the radiation.
    opacity : astropy.units.cm**2 / astropy.units.g
        The opacity of the dust.

    Returns
    -------
    radiation : numpy.ndarray
        The outgoing radiation of the dust grain.
    """
    bb = BlackBody(temperature * u.K)(nu)
    return 4 * np.pi * u.sr * np.trapezoid(bb * opacity, nu)


def thermal_equilibrium(
    temperature: float, p_in: np.ndarray, nu: u.Hz, opacity: u.cm**2 / u.g
) -> float:
    """Computes the difference between the incoming and outgoing radiation of a dust grain.

    Parameters
    ----------
    temperature : float
        The temperature of the dust grain.
    p_in : numpy.ndarray
        The incoming radiation power.
    nu : astropy.units.Hz
        The frequency of the radiation.
    opacity : astropy.units.cm**2 / astropy.units.g
        The opacity of the dust.

    Returns
    -------
    radiation_difference : float
        The difference between the incoming and outgoing radiation.
    """
    return (p_in - p_out(temperature, nu, opacity)).value


def compute_temperature(
    radial_power: u.erg / u.s / u.cm**2,
    nu: u.Hz,
    opacity: u.cm**2 / u.g,
    temperature_range: List[float],
) -> float:
    thermal_eq = partial(
        thermal_equilibrium,
        p_in=radial_power,
        nu=nu,
        opacity=opacity,
    )
    return root_scalar(thermal_eq, bracket=temperature_range).root


def compute_temperatures(
    p_in: np.ndarray,
    nu: u.Hz,
    opacity: u.cm**2 / u.g,
    temperature_range: List[float] = [0, 10000],
    njobs: int = 4,
) -> np.ndarray:
    """Calculates the dust temperature for a given radial power.

    Parameters
    ----------
    p_in : numpy.ndarray
        The incoming radiation power.
    opacity : astropy.units.cm**2 / astropy.units.g
        The opacity of the dust.
    nu : astropy.units.Hz
        The frequency of the radiation.
    temperature_range : list of float, optional
        The temperature range to consider, by default [0, 10000].

    Returns
    -------
    radial_grid : numpy.ndarray
        The dust temperature for the given radial power.
    """
    temp_eq = partial(
        compute_temperature, nu=nu, opacity=opacity, temperature_range=temperature_range
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as executor:
        temperatures = executor.map(temp_eq, p_in)

    return np.array(list(temperatures))


def compute_temperature_grid(
    wavelengths: u.um,
    distance: u.pc,
    flux_star: u.Jy,
    op_sil: np.ndarray,
    op_cont: np.ndarray,
    radial_range: List[float] = [0.1, 100],
    radial_dim: int = 2048,
    temperature_range: List[float] = [0, 10000],
    weight_steps: float = 0.01,
    njobs: int = 4,
) -> np.ndarray:
    """Computes the dust temperature for an opacity curve and different
    continuum weights for a grid of radii.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        The wavelengths of the opacities.
    distance : astropy.units.pc
        The distance to the star.
    flux_star : astropy.units.Jy
        The flux of the star.
    sil_op : numpy.ndarray
        The silicate opacities.
    cont_op : numpy.ndarray
        The continuum opacities.
    radial_range : list of float, optional
        The range of radii to consider, by default [0.1, 100].
    radial_dim : int, optional
        The number of radii to consider, by default 2048.
    temperature_range : list of float, optional
        The temperature range to consider, by default [0, 10000].
    weight_steps : float, optional
        The steps to consider for the weights, by default 0.01.
    njobs : int, optional
        The number of jobs to use for the computation, by default 4.

    Returns
    -------
    temperature_grid : numpy.ndarray
        A matrix of temperatures for the different continuum weights.
    """
    flux_star = flux_star.to(u.erg / u.s / u.cm**2 / u.Hz)
    nu = (const.c / (wavelengths * u.um).to(u.m)).to(u.Hz)
    radii = np.logspace(
        np.log10(radial_range[0]), np.log10(radial_range[-1]), radial_dim
    )
    radiation_field = (
        flux_star * (distance.to(u.au) / (radii[:, np.newaxis] * u.au)) ** 2
    )

    weights = np.arange(0, 1.0 + weight_steps, weight_steps)
    matrix = np.zeros((weights.size, radii.size))
    for index, cont_weight in enumerate(
        tqdm(
            weights,
            desc="Calculating opacity temperatures...",
        )
    ):
        opacity = (1 - cont_weight) * op_sil + cont_weight * op_cont
        p_in = np.trapezoid(opacity[np.newaxis, :] * radiation_field, nu)

        matrix[index] = compute_temperatures(
            p_in, nu, opacity, temperature_range, njobs
        )

    return weights, radii, matrix * u.K


if __name__ == "__main__":
    plot_dir = Path("plots")
    data_dir = Path(ppdmod.__file__).parent.parent / "data"
    opacity_dir = data_dir / "opacities"
    source_dir = data_dir / "model_results" / "hd142527"

    method = "grf"
    wl_op, op_sil = np.load(source_dir / f"silicate_{method}_opacities.npy")
    wl_flux, _, flux = np.load(
        data_dir / "flux" / "hd142527" / "HD142527_stellar_model.npy"
    )
    wl_cont, op_cont = load_data(
        data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat",
        load_func=qval_to_opacity,
    )

    # NOTE: Constrain the flux wavlengths
    min, max = max(wl_op[0], wl_cont[0]), min(wl_op[-1], wl_cont[-1])
    ind = np.where((wl_flux >= min) & (wl_flux <= max))
    wl_flux, flux = wl_flux[ind], flux[ind]

    interp_op_sil = np.interp(wl_flux, wl_op, op_sil)
    interp_op_cont = np.interp(wl_flux, wl_cont, op_cont)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    ax.plot(wl_op, op_sil, label="Silicate")
    ax.plot(wl_cont, op_cont, label="Continuum")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    ax.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1})$")
    ax.set_xlim([-5, 100])
    ax.set_ylim([10, None])
    ax.set_title("Original Opacities")

    bx.plot(wl_flux, interp_op_sil, label="Silicate")
    bx.plot(wl_flux, interp_op_cont, label="Continuum")
    bx.set_yscale("log")
    bx.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    bx.set_xlim([-5, 100])
    bx.set_title("Interpolated Opacities")
    bx.set_ylim([10, None])
    plt.savefig(plot_dir / "interp_op_to_flux.pdf", format="pdf")

    weights, radii, matrix = compute_temperature_grid(
        wl_flux, 158.51 * u.pc, flux * u.Jy, interp_op_sil, interp_op_cont, njobs=50
    )

    data = SimpleNamespace(weights=weights, radii=radii, values=matrix)
    with open(source_dir / "opacity_temps.pkl", "wb") as save_file:
        pickle.dump(data, save_file)
