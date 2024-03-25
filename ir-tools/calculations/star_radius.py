import astropy.units as u
import astropy.constants as const
import numpy as np


def stellar_radius(luminosity: u.Lsun, temperature: u.K):
    return np.sqrt(luminosity.to(u.W)/(4*np.pi*const.sigma_sb*temperature**4)).to(u.Rsun)


if __name__ == "__main__":
    star_radius = stellar_radius(8.69*u.Lsun, 7500*u.K)
    print(star_radius)
