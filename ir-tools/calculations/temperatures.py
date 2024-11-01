import pickle
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import List

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.modeling.models import BlackBody
from joblib import Parallel, delayed
from ppdmod.utils import load_data, qval_to_opacity
from scipy.optimize import root_scalar


def p_out(temperature: float, opacity: u.cm**2 / u.g, wavelengths: u.um) -> np.ndarray:
    bb = BlackBody(temperature * u.K)(wavelengths).value
    return 4 * np.pi * np.trapezoid(bb * opacity, wavelengths.value)


def thermal_equilibrium(
    temperature: float, p_in: np.ndarray, wavelengths: u.um, opacity: u.cm**2 / u.g
) -> float:
    """Computes the difference between the incoming and outgoing radiation of a dust grain."""
    breakpoint()
    return p_in - p_out(temperature, opacity, wavelengths)


def calculate_radial_elements(
    p_in: np.ndarray,
    opacity: u.cm**2 / u.g,
    wavelengths: u.um,
    temperature_range: List[float] = [0, 10000],
) -> np.ndarray:
    """Calculates the dust temperature for a given radial power."""
    radial_grid = []
    for radial_power in p_in:
        thermal_eq = partial(
            thermal_equilibrium,
            p_in=radial_power,
            wavelengths=wavelengths,
            opacity=opacity,
        )

        test = root_scalar(thermal_eq, bracket=temperature_range, method="brentq")
        breakpoint()
    return np.array(radial_grid)


def compute_temperature_grid(
    wavelengths: u.um,
    distance: u.pc,
    flux_star: u.Jy,
    silicate_opacity: np.ndarray,
    continuum_opacity: np.ndarray,
    radial_range: List[float] = [0.1, 100],
    radial_dim: int = 2048,
    temperature_range: List[float] = [0, 10000],
    weight_steps: float = 0.01,
    ncores: int = 6,
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
    silicate_opacity : numpy.ndarray
        The silicate opacities.
    continuum_opacity : numpy.ndarray
        The continuum opacities.
    radial_range : list of float, optional
        The range of radii to consider, by default [0.1, 100].
    radial_dim : int, optional
        The number of radii to consider, by default 2048.
    temperature_range : list of float, optional
        The temperature range to consider, by default [0, 10000].
    weight_steps : float, optional
        The steps to consider for the weights, by default 0.01.

    Returns
    -------
    temperature_grid : numpy.ndarray
        A matrix of temperatures for the different continuum weights.
    """
    silicate_opacity = silicate_opacity[::-1]
    continuum_opacity = continuum_opacity[::-1]
    flux_star = flux_star.to(u.erg / u.s / u.cm**2 / u.Hz)[::-1]
    wavelengths = (wavelengths * u.um)[::-1]

    nu = (const.c / wavelengths.to(u.m)).to(u.Hz)
    radii = np.logspace(
        np.log10(radial_range[0]), np.log10(radial_range[-1]), radial_dim
    )

    weight_grid = []
    for weight in np.arange(0, 1.0 + weight_steps, weight_steps):
        opacity = (1 - weight) * silicate_opacity + weight * continuum_opacity
        opacity = opacity[np.newaxis, :]

        radiation_field = (
            flux_star * (distance.to(u.au) / (radii[:, np.newaxis] * u.au)) ** 2
        )
        p_in = np.trapezoid(opacity * radiation_field, nu).value

        # TODO: Change this into multiprocessing here
        radial_elements = calculate_radial_elements(p_in, opacity, wavelengths, temperature_range)


if __name__ == "__main__":
    # data_dir = Path("/data/beegfs/astro-storage/groups/matisse/scheuck/data")
    data_dir = Path().home() / "Data"

    method = "grf"
    opacity_dir = data_dir / "opacities"
    wl_op, silicate_op = np.load(
        opacity_dir / f"hd142527_silicate_{method}_opacities.npy"
    )

    wl_flux, flux = load_data(
        data_dir / "flux" / "hd142527" / "HD142527_stellar_model.txt", usecols=(0, 2)
    )
    wl_cont, cont_op = load_data(
        data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat",
        load_func=qval_to_opacity,
    )
    # NOTE: Constrain the flux wavlengths
    min, max = max(wl_op[0], wl_cont[0]), min(wl_op[-1], wl_cont[-1])
    ind = np.where((wl_flux >= min) & (wl_flux <= max))
    wl_flux, flux = wl_flux[ind], flux[ind]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 6), sharey=True, sharex=True)
    ax.plot(wl_op, silicate_op, label="Silicate")
    ax.plot(wl_cont, cont_op, label="Continuum")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    ax.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1})$")
    ax.set_xlim([-5, 100])

    silicate_op = np.interp(wl_flux, wl_op, silicate_op)
    cont_op = np.interp(wl_flux, wl_cont, cont_op)

    bx.plot(wl_flux, silicate_op, label="Silicate")
    bx.plot(wl_flux, cont_op, label="Continuum")
    bx.set_yscale("log")
    bx.set_xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    bx.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1})$")
    bx.set_xlim([-5, 100])
    plt.savefig("opacities.pdf", format="pdf")

    test = BlackBody(1 * u.K)(1 * u.Hz)
    breakpoint()

    weights, radii, temperatures = compute_temperature_grid(
        wl_flux,
        158.51 * u.pc,
        flux * u.Jy,
        silicate_op,
        cont_op,
        radial_dim=1024,
        ncores=50,
    )

    data = SimpleNamespace(weights=weights, radii=radii, values=temperatures)
    with open("hd142527_dust_temperatures.pkl", "wb") as save_file:
        pickle.dump(data, save_file)
