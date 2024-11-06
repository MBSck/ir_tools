import shutil
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d


def sampled_binning():
    """This function is a placeholder for the binning function."""
    ...


def sampled_interpolation(
    wavelengths: np.ndarray, wavelength_grid: np.ndarray, values: np.ndarray, **kwargs
) -> np.ndarray:
    """Interpolates the values to the wavelength grid.

    Notes
    -----
    Problematic at edge regions or when the grid to downsample to has a broader coverage than
    the higher spectrally resolved grid.

    Parameters
    ----------
    wavelengths : np.ndarray
        The wavelengths to downsample to.
    wavelength_grid : np.ndarray
        The wavelength grid to downsample from.
    values : np.ndarray
        The values to downsample.

    Returns
    -------
    downsampled_values : np.ndarray
        The downsampled values.
    """
    sampling_factor = wavelength_grid.size // wavelengths.size
    oversampled_wavelengths = np.linspace(
        wavelengths.min(), wavelengths.max(), wavelengths.size * sampling_factor
    )
    new_values = interp1d(
        wavelength_grid, values, kind="cubic", fill_value="extrapolate"
    )(oversampled_wavelengths)

    downsampled_values = np.zeros_like(wavelengths)
    window = np.diff(oversampled_wavelengths)[0]
    for index, wl in enumerate(wavelengths):
        indices = np.where(
            (oversampled_wavelengths >= (wl - window))
            & (oversampled_wavelengths <= (wl + window))
        )
        downsampled_values[index] = new_values[indices].mean()

    downsampled_values[np.where((wavelengths <= 8e-6))] = np.nan
    return downsampled_values


def downsample(
    fits_to_downsample: Path,
    fits_to_sample_from: np.ndarray,
    cards: List[str] = ["oi_flux", "oi_vis2", "oi_vis", "oi_t3"],
    sampling_function: Callable = sampled_interpolation,
    sampling_kwargs: Dict | None = {},
    do_plot: bool = False,
) -> None:
    """
    This function downsamples a fits file to the wavelengths of another fits file.

    Parameters
    ----------
    fits_to_downsample : Path
        The (.fits)-file to downsample.
    fits_to_sample_from : Path
        The (.fits)-file to get the wavelength from and sample to.
        The final (.fits)-file will also have the same structure as this.
    cards : list of str, optional
        The cards to downsample.
    sampling_function : Callable, optional
        The function to use for resampling.
    sampling_kwargs : Dict, optional
        The keyword arguments to pass to the sampling function.
    do_plot : bool, optional
        Whether to plot the downsampled data vs the original data.
    """
    downsampled_fits = fits_to_downsample.with_name(f"{fits_to_downsample.name}_DOWNSAMP.fits")
    shutil.copy(fits_to_sample_from, downsampled_fits)

    keys = []
    for card in cards:
        if "flux" in card:
            keys.append(["fluxdata", "fluxerr"])
        if card == "oi_vis2":
            keys.append(["vis2data", "vis2err"])
        if card == "oi_vis":
            keys.append(["visamp", "visamperr"])
        if card == "oi_t3":
            keys.append(["t3phi", "t3phierr"])

    axarr = []
    if do_plot:
        axarr = plt.subplots(len(cards), 1, figsize=(7, 12))[1].flatten()

    with fits.open(fits_to_sample_from) as hdul:
        wavelengths = hdul["oi_wavelength"].data["eff_wave"]

    with fits.open(fits_to_downsample, "readonly") as hdul, fits.open(
        downsampled_fits, "update"
    ) as hdul_new:
        wavelength_grid = hdul["oi_wavelength"].data["eff_wave"]

        for index, (card, key) in enumerate(zip(cards, keys)):
            values = hdul[card].data[key[0]]
            errs = hdul[card].data[key[1]]
            if values.shape[0] == 1:
                values, errs = [values], [errs]

            interp_values, interp_errs = [], []
            for value, err in zip(values, errs):
                interp_value = sampling_function(
                    wavelengths, wavelength_grid, value.flatten(), **sampling_kwargs
                )
                interp_err = sampling_function(
                    wavelengths, wavelength_grid, err.flatten(), **sampling_kwargs
                )
                interp_values.append(interp_value)
                interp_errs.append(interp_err)

                if do_plot:
                    axarr[index].plot(wavelengths, interp_value, label="Downsampled")
                    axarr[index].plot(wavelength_grid, value.flatten(), label="Original")
                    # axarr[index].legend()

            interp_values, interp_errs = np.array(interp_values), np.array(interp_errs)
            hdul_new[card].data[key[0]] = interp_values
            hdul_new[card].data[key[1]] = interp_errs

        hdul_new.flush()

        if do_plot:
            plt.savefig("downsampling_comparison.pdf", format="pdf")


if __name__ == "__main__":
    fits_dir = Path().home() / "Data" / "fitting_data" / "hd142527"
    low_res_fits = list(fits_dir.glob("*2022-03-23*"))[0]
    high_res_fits = list((fits_dir / "sed_fit" / "only_high").glob("*INT.fits"))[0]
    downsample(high_res_fits, low_res_fits, do_plot=True)
