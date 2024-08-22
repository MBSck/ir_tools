import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, List

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.modeling.models import BlackBody
from ppdmod.utils import load_data, qval_to_opacity, distance_to_angular


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
        radial_dim: Optional[int] = 512) -> np.ndarray:
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
    radiation_fields = flux_star * ((distance_to_angular(distance, 2 * radii[:, np.newaxis] * u.au).to(u.rad).value / u.rad) ** 2).to(1/u.sr)
    p_in = np.trapz(combined_opacities[:, np.newaxis, :] * radiation_fields[np.newaxis, ...], nu, axis=-1)

    temps = np.arange(temperature_range[0], temperature_range[1] + 1, temperature_steps)[:, np.newaxis]
    bb = BlackBody(temperature=temps * u.K)(wavelengths)[np.newaxis, ...]
    p_out = np.trapz(bb * combined_opacities[:, np.newaxis, :], nu, axis=-1)
    difference = p_in[..., np.newaxis] - p_out[:, np.newaxis, :]
    return weights, radii, temps.flatten()[np.argmin(np.abs(difference), axis=-1)]


if __name__ == "__main__":
    data_dir = Path("/Users/scheuck/Data")
    opacity_dir = data_dir / "opacities"
    wl_op, silicate_op = np.load(opacity_dir / "hd142527_boekel_qval_silicates.npy")
    wl_flux, flux = load_data(data_dir / "flux_data" / "hd142527" / "HD142527_stellar_model.txt")
    wl_cont, cont_op = load_data(
        opacity_dir / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity)
    ind = np.where(wl_cont <= 14)
    wl_cont, cont_op = wl_cont[ind], cont_op[ind]
    flux = np.interp(wl_cont, wl_flux, flux)
    silicate_op = np.interp(wl_cont, wl_op, silicate_op)
    weights, radii, temperatures = compute_temperature_grid(
        wl_cont, 158.51 * u.pc, flux * u.Jy,
        silicate_op, cont_op,
        radial_dim=1024, temperature_steps=0.01)

    data = SimpleNamespace(weights=weights, radii=radii, values=temperatures)
    with open("hd142527_dust_temperatures.npy", "wb") as save_file:
        pickle.dump(data, save_file)
