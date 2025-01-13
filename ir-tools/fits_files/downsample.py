import shutil
from pathlib import Path
from functools import partial
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.stats import circmean


def sampled_interpolation(
    new_grid: np.ndarray, grid: np.ndarray, values: np.ndarray,
    errors: bool = False, phases: bool = False,
    **kwargs
) -> np.ndarray:
    """Interpolates the values to the wavelength grid.

    Notes
    -----
    Problematic at edge regions or when the grid to downsample to has a broader coverage than
    the higher spectrally resolved grid.

    These values might need to be flagged.

    Parameters
    ----------
    new_grid : np.ndarray
        The new_grid.
    grid : np.ndarray
        The original grid.
    values : np.ndarray
        The original values.

    Returns
    -------
    downsampled_values : np.ndarray
        The downsampled values.
    """
    sampling_factor = grid.size // new_grid.size
    oversampled_wavelengths = np.linspace(
        new_grid.min(), new_grid.max(), new_grid.size * sampling_factor
    )
    new_values = interp1d(
        grid, values, fill_value="extrapolate"
    )(oversampled_wavelengths)

    if phases:
        mean_func = partial(circmean, low=-180, high=180)
    else:
        mean_func = np.mean

    downsampled_values = np.zeros_like(new_grid)
    window = np.diff(oversampled_wavelengths)[0]
    for index, wl in enumerate(new_grid):
        indices = np.where(
            (oversampled_wavelengths >= (wl - window))
            & (oversampled_wavelengths <= (wl + window))
        )
        if errors:
            downsampled_values[index] = np.sqrt(np.sum(new_values[indices]**2)) / new_values[indices].size
        else:
            downsampled_values[index] = mean_func(new_values[indices])

    return downsampled_values


# TODO: Think of a way to downsample L band data without pulling window into it
def downsample(
    save_dir : Path,
    fits_to_downsample: Path,
    fits_to_sample_from: np.ndarray,
    cards: List[str] = ["oi_flux", "oi_vis2", "oi_vis", "oi_t3"],
    sampling_function: Callable = sampled_interpolation,
    sampling_kwargs: Dict | None = {},
    use_flags: bool = False,
    do_plot: bool = False,
) -> None:
    """
    This function downsamples a fits file to the wavelengths of another fits file.

    Parameters
    ----------
    fits_to_downsample : pathlib.Path
        The (.fits)-file to downsample.
    fits_to_sample_from : pathlib.Path
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
    ylim : list of float, optional
        The range for the y-limits.
    use_flags : bool, optional
        Whether to use the flags in the data to do the downsampling.
    save_dir : pathlib.Path
        The save directory for the plots and downsampled fits.
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    downsampled_fits = save_dir / f"{fits_to_downsample.stem}_DOWNSAMP.fits"
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
        axarr = plt.subplots(len(cards), 3, figsize=(10, 12))[1]

    with fits.open(fits_to_sample_from) as hdul:
        wavelengths = hdul["oi_wavelength"].data["eff_wave"]

    with fits.open(fits_to_downsample, "readonly") as hdul, fits.open(
        downsampled_fits, "update"
    ) as hdul_new:
        grid = hdul["oi_wavelength"].data["eff_wave"]

        for index, (card, key) in enumerate(zip(cards, keys)):
            values = hdul[card].data[key[0]]
            errs = hdul[card].data[key[1]]
            phases = True if "phi" in key[0] else False

            if values.shape[0] == 1:
                values, errs = [values], [errs]

            if use_flags:
                flags = hdul[card].data["flag"]
                values = [np.ma.masked_array(value, mask=flag) for value, flag in zip(values, flags)]
            else:
                values = [np.ma.masked_array(value, mask=np.zeros_like(value).astype(bool)) for value in values]

            interp_values, interp_errs = [], []
            for value, err, new_flag in zip(values, errs, hdul_new[card].data["flag"]):
                interp_value = sampling_function(
                    wavelengths, grid, value.flatten(),
                    False, phases, **sampling_kwargs
                )
                interp_err = sampling_function(
                    wavelengths, grid, err.flatten(), 
                    True, phases, **sampling_kwargs,
                )
                interp_values.append(interp_value)
                interp_errs.append(interp_err)

                if do_plot:
                    if use_flags:
                        interp_value = np.ma.masked_array(interp_value, mask=new_flag)
                        interp_err = np.ma.masked_array(interp_err, mask=new_flag)

                    if card in ["oi_flux", "oi_vis"]:
                        ylim = (0, np.percentile(value.data, 80))
                    elif card in ["oi_vis2"]:
                        ylim = (0, 1)
                    else:
                        ylim = (np.percentile(value.data, 10), np.percentile(value.data, 90))

                    slices = slice(2, -2)
                    axarr[0, 0].set_title("Original vs Downsampled")
                    axarr[index, 0].plot(grid[slices], value.flatten()[slices], label="Original")
                    axarr[index, 0].plot(wavelengths[slices], interp_value[slices], label="Downsampled")
                    axarr[index, 0].set_ylim(ylim)

                    axarr[0, 1].set_title("Original")
                    line = axarr[index, 1].plot(grid[slices], value.flatten()[slices], label="Original")
                    axarr[index, 1].fill_between(
                        grid[slices],
                        (value.flatten() - err.flatten())[slices],
                        (value.flatten() + err.flatten())[slices],
                        color=line[0].get_color(),
                        alpha=0.5,
                    )
                    axarr[index, 1].set_ylim(ylim)

                    axarr[0, 2].set_title("Downsampled")
                    line = axarr[index, 2].plot(wavelengths[slices], interp_value[slices], label="Downsampled")
                    axarr[index, 2].fill_between(
                        wavelengths[slices],
                        (interp_value - interp_err)[slices],
                        (interp_value + interp_err)[slices],
                        color=line[0].get_color(),
                        alpha=0.5,
                    )
                    axarr[index, 2].set_ylim(ylim)

            interp_values, interp_errs = np.array(interp_values), np.array(interp_errs)
            hdul_new[card].data[key[0]] = interp_values
            hdul_new[card].data[key[1]] = interp_errs

        hdul_new.flush()

        if do_plot:
            plt.savefig(save_dir / f"{fits_to_downsample.stem}_downsampled.pdf", format="pdf")
            plt.close()


if __name__ == "__main__":
    fits_dir = Path().home() / "Data" / "fitting_data" / "hd142527"
    # low_res_fits = list(fits_dir.glob("*2022-03-23*_N_*"))[0]
    # fits_file = list((fits_dir / "nband_fit" / "only_high").glob("*.fits"))[0]
    # downsample(fits_dir / "downsampled", fits_file, low_res_fits, use_flags=False, do_plot=True)
    low_res_fits = fits_dir / "HD_142527_2021-03-11T06_47_07_K0G2D0J3_L_TARGET_CHOPPED_FINALCAL_INT.fits"
    fits_files = list(fits_dir.glob("*2022-03-14*"))
    for fits_file in fits_files:
        downsample(fits_dir / "downsampled", fits_file, low_res_fits, use_flags=True, do_plot=True)
