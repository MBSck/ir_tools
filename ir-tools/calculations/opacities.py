from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from ppdmod.utils import get_opacity

SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]


def generate_opacity_file(
    data_dir: Path,
    weights: np.ndarray,
    method: str = "grf",
    fits_file: Path | None = None,
    do_plot: bool = False,
) -> None:
    wl, opacity = get_opacity(data_dir / "opacities", weights, NAMES, method)

    if fits_file is not None:
        with fits.open(fits_file) as hdul:
            wl_flux = (hdul["OI_WAVELENGTH"].data["EFF_WAVE"] * u.m).to(u.um).value
            flux = hdul["OI_FLUX"].data["FLUXDATA"]
            ind = np.where((wl >= 8.5) & (wl <= wl_flux[-1]))
            flux = np.interp(wl[ind], wl_flux, flux.flatten())
            opacity[ind] = flux / flux.max() * opacity[ind].max()

    if do_plot:
        empirical = "_empirical" if fits_file is not None else ""

        plt.plot(wl, opacity)
        plt.xlabel(r"$\lambda$ ($\mathrm{\mu}$m)")
        plt.ylabel(r"$\kappa$ (cm$^2$ g$^{-1}$)")
        plt.yscale("log")
        plt.xlim([0, 30])
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
        weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0]) / 1e2
    else:
        _, weights = np.load(data_dir / "opacities" / "silicate_labels_and_weights.npy")
        weights = weights.astype(float) / weights.astype(float).sum()

    generate_opacity_file(data_dir, weights, method=method, do_plot=True)
    # generate_opacity_file(
    #     data_dir, weights, method=method, fits_file=fits_file, do_plot=True
    # )
