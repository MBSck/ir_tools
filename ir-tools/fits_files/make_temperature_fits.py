import pickle
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ppdmod.utils import load_data, qval_to_opacity
from scipy.interpolate import interp1d


if __name__ == "__main__":
    flux_dir = Path("/Users/scheuck/Data/flux_data/hd142527/")
    opacity_dir = Path("/Users/scheuck/Data/opacities")
    with open(flux_dir / "hd142527_dust_temperatures.pkl", "rb") as f:
        temperatures = pickle.load(f)

    wl_op, silicate_opacity = np.load(opacity_dir / "hd142527_boekel_qval_silicates.npy")
    wl_cont, cont_opacity = load_data(
        opacity_dir / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity)
    ind_cont = np.where(wl_cont <= 14)
    wl_cont, cont_opacity = wl_cont[ind_cont], cont_opacity[ind_cont]
    silicate_opacity = np.interp(wl_cont, wl_op, silicate_opacity)

    primary_hdu = fits.PrimaryHDU(temperatures.values)
    grid_hdu = fits.BinTableHDU(Table(data={"weights": [temperatures.weights],
                                            "radii": [temperatures.radii]}),
                                name="grid".upper())
    opacity_hdu = fits.BinTableHDU(Table(data={"silicates": [wl_op, silicate_opacity],
                                               "continuum": [wl_cont, cont_opacity]}),
                                   name="opacity".upper())
    hdul = fits.HDUList([primary_hdu, grid_hdu, opacity_hdu])
    hdul.writeto("temperatures.fits", overwrite=True)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6))
    const_kappa_temp = np.sqrt((3.46 * u.Rsun).to(u.au) / (2 * temperatures.radii)) * (6500 * u.K)
    ax.plot(temperatures.radii, const_kappa_temp, label=r"Const. $\kappa$", c="k")
    for index, weight in enumerate(np.arange(0, 1.1, 0.1)):
        ax.plot(temperatures.radii, interp1d(temperatures.weights, temperatures.values, axis=0)(weight), label=f"{weight:.1f}")
        bx.plot(wl_cont, (1 - weight) * silicate_opacity + weight * cont_opacity, label=f"{weight:.1f}")

    ax.set_title("Temperatures")
    ax.set_xlabel("$R$ (au)")
    ax.set_ylabel("$T$ (K)")
    ax.set_xscale("log")
    ax.legend()

    bx.set_title("Opacities")
    bx.set_yscale("log")
    bx.set_xlabel(r"$\lambda$ ($\mu$ m)")
    bx.set_ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$)")
    plt.savefig("temperatures.png", format="png", dpi=300)
