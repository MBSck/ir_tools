from functools import partial

import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.modeling.models import BlackBody
from ppdmod.utils import distance_to_angular, angular_to_distance


def calculate_dust_temperature(luminosity, radius):
    return (luminosity.to(u.W) / (16 * np.pi * radius.to(u.m)**2 * const.sigma_sb)) ** (1/4)


def calculate_intensity(wl, temps):
    return BlackBody(temperature=temps)(wl)


def calculate_opacity(wl, beta):
    nu = const.c / wl.to(u.m)
    return 0.1 * (nu / (1e12 * u.Hz).to(u.Hz))**beta * u.cm**2/u.g


def calculate_emissivity(radius, sigma0, p, opacity):
    surface_density = (sigma0 * (radius/(1 * u.au))**(-p)) * u.g/u.cm**2
    optical_depth = surface_density * opacity
    return 1 - np.exp(-optical_depth)


def integrate_flux(luminosity, radial_range, wl, distance, sigma0, p):
    radius = np.logspace(
        np.log10(radial_range[0].value),
        np.log10(radial_range[1].value), 2048) * radial_range.unit
    temps = calculate_dust_temperature(luminosity, radius)
    intensity = calculate_intensity(wl, temps)
    cmb_intensity = calculate_intensity(wl, 2.73 * u.K)
    intensity -= cmb_intensity

    intensity = intensity.to(u.erg/u.s/u.cm**2/u.Hz/u.mas**2)
    opacity = calculate_opacity(wl, 1)
    intensity *= calculate_emissivity(radius, sigma0, p, opacity)
    radius_ang = distance_to_angular(radius, distance)
    return 2 * np.pi * np.trapz(radius_ang * intensity, radius_ang).to(u.Jy)


def integrate_dust_mass(p, sigma0, radial_range):
    radius = np.logspace(
        np.log10(radial_range[0].value),
        np.log10(radial_range[1].value), 2048) * radial_range.unit
    surface_density = (sigma0 * (radius/(1 * u.au))**(-p)) * u.g/u.cm**2
    return (2 * np.pi * np.trapz(surface_density * radius.to(u.cm), radius.to(u.cm)))


if __name__ == "__main__":
    continuum_flux = 4.1 * u.mJy
    bb_flux = integrate_flux(22.39*u.Lsun, [0.1, 4]*u.au, 0.88*u.mm, 158.51*u.pc, 0, 0)

    # working_combinations = {}
    # for p in np.arange(0.5, 1.6, 0.1):
    #     p = np.round(p, 1)
    #     working_combinations[p] = []
    #     for sigma0 in np.arange(0, 10, 0.1):
    #         flux = integrate_flux(22.39*u.Lsun, [0.1, 4]*u.au, 0.88*u.mm, 158.51*u.pc, sigma0, p)
    #         if np.abs(flux - continuum_flux) < (1 * u.mJy):
    #             working_combinations[p].append(np.round(rho, 2))

    dust_mass = integrate_dust_mass(1, 3.1, [0.1, 4] * u.au)
    dust_mass_e = np.round(dust_mass.to(u.M_earth), 2)
    dust_mass_jp = np.round(dust_mass.to(u.M_jup), 5)
    gas_mass_e = dust_mass_e * 100
    gas_mass_jp = dust_mass_jp * 100
    print(gas_mass_e, gas_mass_jp)
