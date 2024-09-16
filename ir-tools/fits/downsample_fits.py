import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d


def sampled_interpolation(wavelengths, wavelength_grid, values):
    sampling_factor = wavelength_grid.size // wavelengths.size
    oversampled_wavelengths = np.linspace(wavelengths.min(), wavelengths.max(), wavelengths.size * sampling_factor)
    new_values = interp1d(wavelength_grid, values, kind="cubic", fill_value="extrapolate")(oversampled_wavelengths)

    downsampled_values = np.zeros_like(wavelengths)
    window = np.diff(oversampled_wavelengths)[0]
    for index, wl in enumerate(wavelengths):
        indices = np.where(
                (oversampled_wavelengths >= (wl - window))
                & (oversampled_wavelengths <= (wl + window)))
        downsampled_values[index] = new_values[indices].mean()

    downsampled_values[np.where((wavelengths <= 8e-6))] = np.nan
    return downsampled_values


def downsample(fits_to_downsample: Path, fits_to_sample_from: np.ndarray):
    downsampled_fits = fits_to_downsample.parent / f"{fits_to_downsample.stem}_downsampled.fits"
    shutil.copy(fits_to_sample_from, downsampled_fits)
    keys = ["oi_flux", "oi_vis2", "oi_vis", "oi_t3"]
    sub_keys = [["fluxdata", "fluxerr"], ["vis2data", "vis2err"],
                ["visamp", "visamperr"], ["t3phi", "t3phierr"]]
    _, axarr = plt.subplots(len(keys), 1, figsize=(7, 12))

    with fits.open(fits_to_sample_from) as hdul:
        wavelengths = hdul["oi_wavelength"].data["eff_wave"]

    with fits.open(fits_to_downsample, "readonly") as hdul, \
        fits.open(downsampled_fits, "update") as hdul_new:
        wavelength_grid = hdul["oi_wavelength"].data["eff_wave"]

        for ax, key, sub_key in zip(axarr.flatten(), keys, sub_keys):
            values = hdul[key].data[sub_key[0]]
            errs = hdul[key].data[sub_key[1]]
            if values.shape[0] == 1:
                values, errs = [values], [errs]

            interp_values, interp_errs = [], []
            for value, err in zip(values, errs):
                interp_value = sampled_interpolation(
                    wavelengths, wavelength_grid, value.flatten())
                interp_err = sampled_interpolation(
                    wavelengths, wavelength_grid, err.flatten())
                interp_values.append(interp_value)
                interp_errs.append(interp_err)

                ax.plot(wavelengths, interp_value, label="Downsampled")
                ax.plot(wavelength_grid, value.flatten(), label="Original")

            interp_values, interp_errs = np.array(interp_values), np.array(interp_errs)
            hdul_new[key].data[sub_key[0]] = interp_values
            hdul_new[key].data[sub_key[1]] = interp_errs

            # ax.legend()

        # plt.show()
        plt.savefig("downsampled.pdf", format="pdf")
        hdul_new.flush()
        breakpoint()


if __name__ == "__main__":
    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    low_res_fits = data_dir / "hd_142527_2023-07-01T00_57_49:2023-07-01T00_33_15_AQUARIUS_FINAL_TARGET_INT_flux_avg_vis.fits"
    high_res_fits = data_dir / "hd_142527_2021-03-27T05_29_42:2021-03-27T04_58_38_AQUARIUS_FINAL_TARGET_INT_flux_avg_vis.fits"
    downsample(high_res_fits, low_res_fits)

