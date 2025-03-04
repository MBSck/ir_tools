import shutil
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.stats import circmean
from tqdm import tqdm


def sampled_interpolation(
    new_grid: np.ndarray,
    grid: np.ndarray,
    values: np.ndarray,
    errors: bool = False,
    phases: bool = False,
    **kwargs,
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
    new_values = interp1d(grid, values, fill_value="extrapolate")(
        oversampled_wavelengths
    )

    if phases:
        mean_func = partial(circmean, low=-180, high=180)
    else:
        mean_func = np.ma.mean

    downsampled_values = np.zeros_like(new_grid)
    window = np.diff(oversampled_wavelengths)[0]
    for index, wl in enumerate(new_grid):
        indices = np.where(
            (oversampled_wavelengths >= (wl - window))
            & (oversampled_wavelengths <= (wl + window))
        )
        if errors:
            downsampled_values[index] = (
                np.ma.sqrt(np.ma.sum(new_values[indices] ** 2))
                / new_values[indices].size
            )
        else:
            downsampled_values[index] = mean_func(new_values[indices])

    return downsampled_values


def replace_columns(
    hdu: fits.BinTableHDU, names: List[str], data: List[np.ndarray]
) -> fits.BinTableHDU:
    """This replaces the columns by names with the data.

    Parameters
    ----------
    hdu : astropy.io.fits.BinTableHDU
    names : list of str
    data : list of numpy.ndarray

    Returns
    -------
    astropy.io.fits.BinTableHDU
    """
    table = Table(hdu.data)
    for name, value in zip(names, data):
        value = value.astype(np.float64) if name != "flag" else value.astype(bool)
        table.replace_column(name.upper(), value)

    new_hdu = fits.BinTableHDU(data=table, name=hdu.name)
    for key, value in hdu.header.items():
        if key.upper() not in new_hdu.header:
            new_hdu.header[key] = value

    return new_hdu


def downsample(
    save_dir: Path,
    fits_to_downsample: Path,
    fits_to_sample_from: Path,
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

    Notes
    -----
    For downsampling N band turn of use of flagging. For L band have it on.
    """
    save_dir.mkdir(exist_ok=True, parents=True)
    downsampled_fits = save_dir / f"{fits_to_downsample.stem}.fits"
    shutil.copy(fits_to_downsample, downsampled_fits)

    keys = []
    for card in cards:
        if "flux" in card:
            keys.append([["fluxdata", "fluxerr"]])
        if card == "oi_vis2":
            keys.append([["vis2data", "vis2err"]])
        if card == "oi_vis":
            keys.append([["visamp", "visamperr"], ["visphi", "visphierr"]])
        if card == "oi_t3":
            keys.append([["t3amp", "t3amperr"], ["t3phi", "t3phierr"]])

    axarr = []
    if do_plot:
        axarr = plt.subplots(len(cards) + 2, 3, figsize=(10, 12))[1]

    with fits.open(fits_to_sample_from) as old_hdul:
        wavelengths = old_hdul["oi_wavelength"].data["eff_wave"]
        bands = old_hdul["oi_wavelength"].data["eff_band"]
        new_card_flags = [old_hdul[card].data["flag"] for card in cards]
        wl_hdu = fits.BinTableHDU(
            Table([wavelengths, bands], names=("EFF_WAVE", "EFF_BAND"))
        )
        for key, value in old_hdul["oi_wavelength"].header.items():
            if key not in wl_hdu.header:
                wl_hdu.header[key] = value

    with (
        fits.open(fits_to_downsample, "readonly") as hdul,
        fits.open(downsampled_fits, "update") as hdul_new,
    ):
        grid = hdul["oi_wavelength"].data["eff_wave"]
        hdul_new["oi_wavelength"] = wl_hdu

        index = 0
        for card, key, new_flags in zip(cards, keys, new_card_flags):
            value_names, err_names = [], []
            key_values, key_errors = [], []
            for k in key:
                value_names.append(k[0])
                err_names.append(k[1])
                values, errs = hdul[card].data[k[0]], hdul[card].data[k[1]]
                phases = True if "phi" in k[0] else False

                if values.shape[0] == 1:
                    values, errs = [values], [errs]

                if use_flags:
                    flags = hdul[card].data["flag"]
                    values = [
                        np.ma.masked_array(value, mask=flag)
                        for value, flag in zip(values, flags)
                    ]
                else:
                    values = [
                        np.ma.masked_array(
                            value, mask=np.zeros_like(value).astype(bool)
                        )
                        for value in values
                    ]

                interp_values, interp_errs = [], []
                for value, err, new_flag in zip(values, errs, new_flags):
                    interp_value = sampling_function(
                        wavelengths,
                        grid,
                        value.flatten(),
                        False,
                        phases,
                        **sampling_kwargs,
                    )
                    interp_err = sampling_function(
                        wavelengths,
                        grid,
                        err.flatten(),
                        True,
                        phases,
                        **sampling_kwargs,
                    )
                    interp_values.append(interp_value)
                    interp_errs.append(interp_err)

                    if do_plot:
                        if use_flags:
                            interp_value = np.ma.masked_array(
                                interp_value, mask=new_flag
                            )
                            interp_err = np.ma.masked_array(interp_err, mask=new_flag)

                        if card in ["oi_flux", "oi_vis"]:
                            ylim = (0, np.percentile(value.data, 80))
                        elif card in ["oi_vis2"]:
                            ylim = (0, 1)
                        else:
                            ylim = (
                                np.percentile(value.data, 10),
                                np.percentile(value.data, 90),
                            )

                        slices = slice(2, -2)
                        axarr[0, 0].set_title("Original vs Downsampled")
                        axarr[index, 0].plot(
                            grid[slices], value.flatten()[slices], label="Original"
                        )
                        axarr[index, 0].plot(
                            wavelengths[slices],
                            interp_value[slices],
                            label="Downsampled",
                        )
                        axarr[index, 0].set_ylim(ylim)

                        axarr[0, 1].set_title("Original")
                        line = axarr[index, 1].plot(
                            grid[slices], value.flatten()[slices], label="Original"
                        )
                        axarr[index, 1].fill_between(
                            grid[slices],
                            (value.flatten() - err.flatten())[slices],
                            (value.flatten() + err.flatten())[slices],
                            color=line[0].get_color(),
                            alpha=0.5,
                        )
                        axarr[index, 1].set_ylim(ylim)

                        axarr[0, 2].set_title("Downsampled")
                        line = axarr[index, 2].plot(
                            wavelengths[slices],
                            interp_value[slices],
                            label="Downsampled",
                        )
                        axarr[index, 2].fill_between(
                            wavelengths[slices],
                            (interp_value - interp_err)[slices],
                            (interp_value + interp_err)[slices],
                            color=line[0].get_color(),
                            alpha=0.5,
                        )
                        axarr[index, 2].set_ylim(ylim)

                key_values.append(np.array(interp_values))
                key_errors.append(np.array(interp_errs))
                index += 1

            columns = value_names + err_names + ["flag"]
            values = key_values + key_errors + [new_flags]
            hdul_new[card] = replace_columns(hdul[card], columns, values)

        hdul_new.flush()

        if do_plot:
            axarr[0, 0].set_ylabel(r"$F_{\nu}$ (Jy)")
            axarr[1, 0].set_ylabel(r"$V^2$ (a.u.)")
            axarr[2, 0].set_ylabel("Amplitude")
            axarr[3, 0].set_ylabel(r"$\phi_{\mathrm{diff.}}$ ($^\circ$)")
            axarr[4, 0].set_ylabel(r"Amplitude$_{\mathrm{cl.}}$")
            axarr[5, 0].set_ylabel(r"$\phi_{\mathrm{cl.}}$ ($^\circ$)")
            for index in range(3):
                axarr[-1, index].set_xlabel(r"$\lambda$ (m)")

            plot_dir = save_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(
                plot_dir / f"{fits_to_downsample.stem}_downsampled.png",
                format="png",
                dpi=300,
            )
            plt.close()


if __name__ == "__main__":
    matisse_dir = Path().home() / "Data" / "reduced" / "HD_142527" / "matisse"
    non_treated_dir = matisse_dir / "non_treated"
    downsample_dir = matisse_dir / "treated" / "downsampled"
    with open(downsample_dir / "config.yaml", "r") as f:
        data = yaml.safe_load(f)

    for low_res_fits, fits_files in data.items():
        for fits_file in tqdm(fits_files, desc="Downsampling..."):
            downsample(
                downsample_dir,
                non_treated_dir / fits_file,
                non_treated_dir / low_res_fits,
                use_flags=False,
                do_plot=True,
            )
