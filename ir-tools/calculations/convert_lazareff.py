import astropy.units as u
import numpy as np


def calculate_ak(la: float, lk: float) -> float:
    """Calculate the FWHM of the Gaussian beam from the FWHM and the width."""
    return np.sqrt(10 ** (2 * la) / (1 + 10 ** (-2 * lk)))


def calculate_ar(la: float, lk: float) -> float:
    """Calculate the radius of the Gaussian beam from the FWHM and the width."""
    return np.sqrt(10 ** (2 * la) / (1 + 10 ** (2 * lk)))


def calculate_a_and_phi(c1: float, s1: float) -> float:
    """Calculate A and Phi from c1 and s1."""
    return np.hypot(c1, s1), np.arctan2(c1, s1) * u.rad.to(u.deg)
    

if __name__ == "__main__":
    la, lk = 0.08, 0.54
    c1, s1 = -0.6, -0.57
    ak = calculate_ak(la, lk)
    ar = calculate_ar(la, lk)
    a, phi = 1, 0
    print("c1, s1", a * np.cos(phi), a * np.sin(phi))
    print("FWHM:", 2 * 10**la)
    print("hlr:", 10**la)
    print("ak:", ak)
    print("ar:", ar)
    print("w:", ak / np.hypot(ar, ak))
    print("ak (HD45677) vs ak (calculated)", 0.48 * 10**0.98, calculate_ak(0.98, -0.26))
    print("ak (HD144668) vs ak (calculated)", 0.85 * 10**0.09, calculate_ak(0.09, 0.32))

    a, phi = calculate_a_and_phi(c1, s1)
    print("a, phi:", a, phi)
    print("pa:", 2.13 * u.rad.to(u.deg))
    print("pa:", 0.09 * u.rad.to(u.deg))
