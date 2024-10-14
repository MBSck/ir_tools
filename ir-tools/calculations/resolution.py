from pathlib import Path
from typing import List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import ppdmod
from ppdmod.utils import compute_detector_grid
from ppdmod.data import set_data



def get_wl_grid(fits_files: List[Path], band: str) -> np.ndarray:
    """Gets the wavelenght grids for the different bands."""
    data = set_data(fits_files, wavelengths="all", fit_data=["flux", "vis"])
    return np.array(list(map(lambda x: x.wavelength, np.array(data.readouts)[np.where(np.array(data.bands) == band)[0]]))[0])


if __name__ == "__main__":
    wavelengths = {"hband": [1.7] * u.um, "kband": [2.15] * u.um,
                   "lband": np.linspace(3.3, 3.8, 5) * u.um,
                   "mband": np.linspace(4.6, 4.9, 3) * u.um,
                   "nband": np.linspace(8, 15, 35) * u.um,
                   }

    fits_files = list(Path("/Users/scheuck/Data/fitting_data/hd142527").glob("*fits"))
    wavelength = np.concatenate((wavelengths["hband"], wavelengths["kband"],
                                wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))
    wl_hband = get_wl_grid(fits_files, "hband")
    wl_kband = get_wl_grid(fits_files, "kband")
    breakpoint()
    np.save("low_hband.npy", wl_hband)
    np.save("low_kband.npy", wl_kband)

    wl = np.load(Path(ppdmod.__file__).parent.parent / "data" / "wl_grids" / "fits" / "matisse" / "nband_low.npy")
    delta_wl = np.diff(wl)
    res_diff = (10.5 / delta_wl) / (10.5 / delta_wl).mean()

    wl_detector, delta_wl_detector = compute_detector_grid(wl * u.um, 8 * u.um, 2 * u.one)
    np.save("nband_low.npy", wl_detector.value)

    res_detector = 10.5 * u.um / np.diff(wl_detector)
    res_diff_detector = res_detector / res_detector.mean()

    plt.title("Differences in resolution over the grids")
    plt.plot(wl[1:], res_diff, label="Oversampled ($ n = 121$, from .fits)")
    plt.plot(wl_detector[1:], res_diff_detector, label="Detector ($ n = 16 $, calculated)")

    np.save("nband_low.npy", wl[1:], res_diff)
    plt.legend()
    plt.savefig("resolution_differences.pdf", format="pdf")
    # plt.show()
