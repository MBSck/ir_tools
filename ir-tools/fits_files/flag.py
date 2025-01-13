from pathlib import Path

from astropy.io import fits
import numpy as np


def flag_gravity(fits_file: Path):
    """Flags the first index for the GRAVITY data, which usually has detector issues."""


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting_data" / "hd142527"
    with fits.open(path / "HD_142527_2021-03-11T06_47_07_K0G2D0J3_L_TARGET_CHOPPED_FINALCAL_INT.fits", "readonly") as hdul:
        breakpoint()
