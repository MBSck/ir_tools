import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from tqdm import tqdm


def quadratic_error_propagation(data: np.ndarray, error: np.ndarray):
    """Calculates the standard deviation when taking the square root
    of the squared visibilities."""
    return np.sqrt((data**2/data)**2*error**2)


def read_gravity_data(file: Path, index: Optional[int] = 10):
    """Reads the GRAVITY data.

    Notes
    -----
    index 10 = SC ; 20 = FT ; 11,12 = SC_POL ; 21,22 = FT_POL.
    """
    with fits.open(file) as hdul:
        wave = hdul['oi_wavelength', index].data['eff_wave']*1e6
        spectre = np.mean(hdul['oi_flux', index].data['flux'], 0)
        visamp = hdul['oi_vis', index].data['visamp']
        visphi = hdul['oi_vis', index].data['visphi']
        closure = hdul['oi_t3', index].data['t3phi'][:, :]
        ucoord = hdul['oi_vis', index].data['ucoord']
        vcoord = hdul['oi_vis', index].data['vcoord']

        # NOTE: Basename
        dicname = {i: n for i, n in zip(hdul['oi_array'].data['sta_index'],
                                        hdul['oi_array'].data['sta_name'])}
        base = [dicname[i]+'-'+dicname[j] for i, j in hdul['oi_vis', index].data['sta_index']]
        triplet = [dicname[i] + '-'+dicname[j]+'-'+dicname[k] for i, j, k in hdul['oi_t3', index].data['sta_index']]
    return wave, spectre, visamp, visphi, closure, ucoord, vcoord, base, triplet


def calibrate_gravity_flux(target: Path, calibrator: Path,
                           flux_model: Path, index: Optional[int] = 10) -> None:
    """Calibrates the flux of the GRAVITY data."""
    pass



def make_vis_gravity_files(directory: Path) -> None:
    """Makes a set of gravity files where the vis is set to be the vis2
    to make it easier to handle for the fitting."""
    vis_dir = directory / "vis"
    if not vis_dir.exists():
        vis_dir.mkdir(parents=True, exist_ok=True)
    for fits_file in tqdm(list(directory.glob("*.fits"))):
        new_file = vis_dir / f"{fits_file.stem}_vis.fits"
        shutil.copy(fits_file, new_file)
        with fits.open(new_file, "update") as hdul:
            hdul["oi_vis2"].data = hdul["oi_vis"].data
            hdul["oi_vis2"].columns["visamp"].name = "vis2data".upper()
            hdul["oi_vis2"].columns["visamperr"].name = "vis2err".upper()
            hdul.flush()


if __name__ == "__main__":
    path = Path("fits") / "GRAVI.2018-06-16T03%3A22%3A27.798_singlescivis_singlesciviscalibrated.fits"
    # read_gravity_data(path)
    # make_vis_gravity_files(Path("fits"))
