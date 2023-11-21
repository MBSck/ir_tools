import shutil
from pathlib import Path

import astropy.units as u
import numpy as np
from matadrs.utils.plot import Plotter
from astropy.table import QTable
from astropy.io import fits
from tqdm import tqdm


def delete_add_ins(file: Path) -> None:
    """Deletes the oi_flux and oi_vis extensions."""
    with fits.open(file, mode="update") as hdul:
        if "oi_vis" in hdul or "oi_flux" in hdul:
            del hdul["oi_vis"]
            del hdul["oi_flux"]


def quadratic_error_propagation(data: np.ndarray, error: np.ndarray):
    """Calculates the standard deviation when taking the square root
    of the squared visibilities."""
    return np.abs(0.5*error/np.sqrt(data))


def calculate_vis(file: Path, wavelength: u.um,
                  total_flux: u.Jy, total_flux_err: u.Jy,
                  **kwargs) -> None:
    """Calculates the correlated fluxes from the
    squared visibilities and a total flux.

    Also add the total flux to the file.
    """
    delete_add_ins(file)
    if not (dir := file.parent / "vis").exists():
        dir.mkdir(parents=True)

    if not (bad_data_dir := file.parent / "bad_data").exists():
        bad_data_dir.mkdir(parents=True)

    with fits.open(file, mode="readonly") as hdul:
        vis = hdul["oi_vis2"].copy()
        vis_header = vis.header.copy()

    vis_header["EXTNAME"] = "oi_vis".upper()
    vis_header["TTYPE5"] = "visamp".upper()
    vis_header["TTYPE6"] = "visamperr".upper()
    vis.name = "oi_vis".upper()
    vis.columns[4].name = "visamp".upper()
    vis.columns[5].name = "visamperr".upper()
    vis.data["visamp"] = np.sqrt(vis.data["visamp"])*total_flux*u.Jy
    vis.data["visamperr"] = quadratic_error_propagation(
            vis.data["visamp"], vis.data["visamperr"])

    shutil.copy(file, (new_file := dir / f"{file.stem}_vis.fits"))
    with fits.open(new_file, mode="update") as hdul:
        if np.max(hdul["oi_vis2"].data["vis2data"] > 1)\
                or np.min(hdul["oi_vis2"].data["vis2data"] < 0):
            file.rename(bad_data_dir / file.name)
            new_file.unlink()
            return

        vis_value = np.sqrt(hdul["oi_vis2"].data["vis2data"])
        error = quadratic_error_propagation(
                vis_value, hdul["oi_vis2"].data["vis2err"])

        hdul["oi_vis2"].data["vis2data"] = vis_value
        hdul["oi_vis2"].data["vis2err"] = error
        hdul.append(vis)

        wavelengths = hdul["oi_wavelength"].data["eff_wave"]
        flux_header = vis_header.copy()
        flux_header["EXTNAME"] = "oi_flux".upper()
        flux = fits.BinTableHDU(
                QTable({"wavelength": [wavelength*u.um],
                        "fluxdata": [[total_flux for _ in wavelengths]*u.Jy],
                        "fluxerr": [[total_flux_err for _ in wavelengths]*u.Jy]}),
                header=flux_header)
        hdul.append(flux)
        hdul.flush()

    if not (plot_dir := file.parent / "plots").exists():
        plot_dir.mkdir(parents=True)
    if not (new_plot_dir := new_file.parent / "plots").exists():
        new_plot_dir.mkdir(parents=True)
    original_plot = Plotter(file, save_path=plot_dir)
    new_plot = Plotter(new_file, save_path=new_plot_dir)
    original_plot.add_uv().add_vis2().add_cphases().plot(**kwargs)
    new_plot.add_uv().add_vis2().add_cphases().plot(**kwargs)


if __name__ == "__main__":
    pionier_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/pionier")
    directory = pionier_dir / "nChannels6" / "non_kraus"
    for index, file in enumerate(tqdm(list(directory.glob("*.fits")))):
        delete_add_ins(file)
        calculate_vis(file, 1.662, 2.06, 0.05, margin=0.3, error=True, save=True)
