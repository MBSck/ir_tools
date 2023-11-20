from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def read_file(file: Path) -> List:
    with fits.open(file) as hdul:
        wl = hdul["oi_wavelength"].data["eff_wave"]
        vis = hdul["oi_vis"].data["visamp"]
        viserr = hdul["oi_vis"].data["visamperr"]
        vis2 = hdul["oi_vis2"].data["vis2data"]
        vis2err = hdul["oi_vis2"].data["vis2err"]
    return [wl, [vis, viserr], [vis2, vis2err]]


def get_y_limits(ylim: List[int], data: np.ndarray,
                 multiplier: float) -> List[int]:
    lower, upper = np.percentile(data, 75), np.percentile(data, 25)
    interquartile = upper-lower
    lower_bound = lower-interquartile*multiplier
    upper_bound = upper+interquartile*multiplier
    filtered_data = [x for x in data if -lower_bound <= x <= -upper_bound]
    return min(ylim[0], np.min(filtered_data)), max(ylim[1], np.max(filtered_data))


def compare_and_plot(file: Path, file_for_compare: Path,
                     cutoff_multiplier: Optional[int] = 1.5) -> None:
    """Compares the data of two files."""
    data, data_other = read_file(file), read_file(file_for_compare)
    labels = ["Correlated fluxes (Jy)", "Visibilities (a.u.)"]

    diff = []
    for index, (dataset, dataset_other)\
            in enumerate(zip(data[1:], data_other[1:])):
        value = (dataset[0] - dataset_other[0])
        error = (dataset[1] - dataset_other[1])
        diff.append((value, error))

    wl = data[0]
    fig, axarr = plt.subplots(2, 2)
    for index, (values, errors) in enumerate(diff):
        axarr[0, 0].set_title("Value Difference")
        axarr[0, 1].set_title("Error Difference")

        if index == len(diff)-1:
            axarr[index, 0].set_xlabel(r"Wavelength ($\mu$m)")
            axarr[index, 1].set_xlabel(r"Wavelength ($\mu$m)")

        ylim, ylim_err = [0, 0], [0, 0]
        for value, error in zip(values, errors):
            ylim = get_y_limits(ylim, value, cutoff_multiplier)
            ylim_err = get_y_limits(ylim_err, error, cutoff_multiplier)
            axarr[index, 0].plot(wl, value)
            axarr[index, 1].plot(wl, error)
            axarr[index, 0].set_ylabel(labels[index])
        axarr[index, 0].set_ylim(ylim)
        axarr[index, 1].set_ylim(ylim_err)
    fig.tight_layout()
    plt.savefig("test.pdf", format="pdf")


if __name__ == "__main__":
    path_avg_oifits = Path("avg_oifits/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits")
    path_mat_tools = Path("mat_tools/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits")
    compare_and_plot(path_mat_tools, path_avg_oifits, cutoff_multiplier=10)
