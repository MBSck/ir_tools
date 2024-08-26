import pickle
from pathlib import Path

import  numpy as np
from astropy.io import fits
from astropy.table import Table
from ppdmod.utils import load_data, qval_to_opacity

if __name__ == "__main__":
    flux_dir = Path("/Users/scheuck/Data/flux_data/hd142527/")
    opacity_dir = Path("/Users/scheuck/Data/opacities")
    with open(flux_dir / "hd142527_dust_temperatures.pkl", "rb") as f:
        temperatures = pickle.load(f)

    wl_op, silicate_opacity = np.load(opacity_dir / "hd142527_boekel_qval_silicates.npy")
    wl_cont, cont_opacity = load_data(
        opacity_dir / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity)

    primary_hdu = fits.PrimaryHDU(temperatures.values)
    grid_hdu = fits.BinTableHDU(Table(data={"weights": [temperatures.weights],
                                            "radii": [temperatures.radii]}),
                                name="grid".upper())
    opacity_hdu = fits.BinTableHDU(Table(data={"silicates": [wl_op, silicate_opacity],
                                               "continuum": [wl_cont, cont_opacity]}),
                                   name="opacity".upper())
    hdul = fits.HDUList([primary_hdu, grid_hdu, opacity_hdu])
    hdul.writeto("temperatures.fits", overwrite=True)
