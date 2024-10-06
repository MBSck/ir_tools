import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.modeling.models import BlackBody
from joblib import Parallel, delayed
from ppdmod.utils import load_data, qval_to_opacity
from scipy.interpolate import interp1d


def integrate_slice(integrand, nu):
    return np.trapz(integrand, nu, axis=-1)


def integrate_values(integrand, nu, n_jobs=6):
    slices = np.array_split(integrand, n_jobs, axis=0)
    results = Parallel(n_jobs=n_jobs)(delayed(integrate_slice)(s, nu) for s in slices)
    return np.concatenate(results, axis=0)


def compute_temperature_grid(
        wavelengths: u.um,
        distance: u.pc,
        flux_star: u.Jy,
        silicate_opacities: np.ndarray,
        continuum_opacities: np.ndarray,
        radial_range: Optional[List[float]] = [0.1, 50],
        weight_steps: Optional[float] = 0.01,
        temperature_steps: Optional[float] = 0.25,
        temperature_range: Optional[List[float]] = [100, 2600],
        radial_dim: Optional[int] = 512,
        ncores: Optional[int] = 6) -> np.ndarray:
    """Computes the dust temperature for an opacity curve and different
    continuum weights for a grid of radii.

    Parameters
    ----------
    silicate_opacities : numpy.ndarray
        The silicate opacities.
    continuum_opacities : numpy.ndarray
        The continuum opacities.

    Returns
    -------
    temperature_grid : numpy.ndarray
        A matrix of temperatures for the different continuum weights.
    """
    silicate_opacities = silicate_opacities[::-1]
    continuum_opacities = continuum_opacities[::-1]
    flux_star = flux_star.to(u.erg/u.s/u.cm**2/u.Hz)[::-1]
    wavelengths = (wavelengths * u.um)[::-1]

    weights = np.arange(0, 1.01, weight_steps)
    nu = (const.c / wavelengths.to(u.m)).to(u.Hz)
    radii = np.logspace(np.log10(radial_range[0]), np.log10(radial_range[-1]), radial_dim)
    combined_opacities = (1 - weights[:, np.newaxis]) * silicate_opacities + weights[:, np.newaxis] * continuum_opacities
    radiation_fields = flux_star * (distance.to(u.au) / radii[:, np.newaxis] * u.au) ** 2
    p_in = np.trapz((combined_opacities[:, np.newaxis, :] * radiation_fields[np.newaxis, ...]).value, nu.value, axis=-1)

    temps = np.arange(temperature_range[0], temperature_range[1] + 1, temperature_steps)[:, np.newaxis]
    bb = 4 * np.pi * u.sr * BlackBody(temperature=temps * u.K)(wavelengths)[np.newaxis, ...]
    p_out = integrate_values((bb * combined_opacities[:, np.newaxis, :]).value, nu.value, njobs=ncores)
    difference = p_in[..., np.newaxis] - p_out[:, np.newaxis, :]
    return weights, radii, temps.flatten()[np.argmin(np.abs(difference), axis=-1)]


if __name__ == "__main__":
    data_dir = Path("/Users/scheuck/Data")
    opacity_dir = data_dir / "opacities"
    wl_op, silicate_op = np.load(opacity_dir / "hd142527_combined_silicate_opacities.npy")
    wl_flux, flux = load_data(
        data_dir / "flux_data" / "hd142527" / "HD142527_stellar_model.txt", usecols=(0, 2))
    wl_cont, cont_op = np.load(data_dir / "opacities" / "optool" / "preibisch_amorph_c_rv0.1.npy")

    import matplotlib.pyplot as plt
    plt.plot(wl_op, silicate_op, label="Silicate")
    plt.plot(wl_cont, cont_op, label="Continuum")
    plt.yscale("log")
    plt.xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
    plt.ylabel(r"$\kappa$ (cm$^2$ g$^{-1})$")
    plt.xlim([-5, 100])
    plt.savefig("opacities.pdf", format="pdf")

    weights, radii, temperatures = compute_temperature_grid(
        wl_flux, 158.51 * u.pc, flux * u.Jy, silicate_op, cont_op,
        radial_dim=1024, temperature_steps=0.1)

    data = SimpleNamespace(weights=weights, radii=radii, values=temperatures)
    with open("hd142527_dust_temperatures.pkl", "wb") as save_file:
        pickle.dump(data, save_file)
