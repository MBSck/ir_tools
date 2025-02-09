import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from matadrs.utils.plot import Plotter
from tqdm import tqdm
from uncertainties import unumpy

from .pionier import calculate_vis


def read_gravity_data(file: Path, index: Optional[int] = 10):
    """Reads the GRAVITY data.

    Notes
    -----
    index 10 = SC ; 20 = FT ; 11,12 = SC_POL ; 21,22 = FT_POL.
    """
    with fits.open(file) as hdul:
        header = hdul[0].header
        print([header[f"hierarch eso det{i} seq1 dit"] for i in range(1, 4)])
        wave = hdul["oi_wavelength", index].data["eff_wave"] * 1e6
        spectre = np.mean(hdul["oi_flux", index].data["flux"], 0)
        visamp = hdul["oi_vis", index].data["visamp"]
        visphi = hdul["oi_vis", index].data["visphi"]
        closure = hdul["oi_t3", index].data["t3phi"][:, :]
        ucoord = hdul["oi_vis", index].data["ucoord"]
        vcoord = hdul["oi_vis", index].data["vcoord"]

        # NOTE: Basename
        dicname = {
            i: n
            for i, n in zip(
                hdul["oi_array"].data["sta_index"], hdul["oi_array"].data["sta_name"]
            )
        }
        base = [
            dicname[i] + "-" + dicname[j]
            for i, j in hdul["oi_vis", index].data["sta_index"]
        ]
        triplet = [
            dicname[i] + "-" + dicname[j] + "-" + dicname[k]
            for i, j, k in hdul["oi_t3", index].data["sta_index"]
        ]
    return wave, spectre, visamp, visphi, closure, ucoord, vcoord, base, triplet


def calibrate_gravity_flux(
    target: Path, calibrator: Path, flux_file: Path, output_dir: Optional[Path] = None
) -> None:
    """Calibrates the flux of the GRAVITY data."""
    output_dir = Path("calibrated") if output_dir is None else output_dir
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    new_file = output_dir / f"{target.stem}_flux_calibrated.fits"
    shutil.copy(target, new_file)
    with fits.open(target, "readonly") as hdul:
        wave_sc = hdul["oi_wavelength", 10].data["eff_wave"] * 1e6
        wave_ft = hdul["oi_wavelength", 20].data["eff_wave"] * 1e6

        flux_target_sc = hdul["oi_flux", 10].data["flux"]
        flux_target_sc_err = hdul["oi_flux", 10].data["fluxerr"]
        sta_index_target_sc = hdul["oi_flux", 10].data["sta_index"]

        flux_target_ft = hdul["oi_flux", 20].data["flux"]
        flux_target_ft_err = hdul["oi_flux", 20].data["fluxerr"]
        sta_index_target_ft = hdul["oi_flux", 20].data["sta_index"]

    with fits.open(calibrator, "readonly") as hdul:
        flux_cal_sc = hdul["oi_flux", 10].data["flux"]
        flux_cal_sc_err = hdul["oi_flux", 10].data["fluxerr"]
        sta_index_cal_sc = hdul["oi_flux", 10].data["sta_index"]

        flux_cal_ft = hdul["oi_flux", 20].data["flux"]
        flux_cal_ft_err = hdul["oi_flux", 20].data["fluxerr"]
        sta_index_cal_ft = hdul["oi_flux", 20].data["sta_index"]

    flux_model_sc = get_model_flux(wave_sc, flux_file)
    flux_model_ft = get_model_flux(wave_ft, flux_file)
    flux_sc = unumpy.umatrix(flux_target_sc, flux_target_sc_err) / unumpy.umatrix(
        flux_cal_sc, flux_cal_sc_err
    )
    flux_ft = unumpy.umatrix(flux_target_ft, flux_target_ft_err) / unumpy.umatrix(
        flux_cal_ft, flux_cal_ft_err
    )
    cal_flux_sc, cal_flux_sc_err = map(
        lambda x: np.array(x.tolist()) * flux_model_sc,
        (unumpy.nominal_values(flux_sc), unumpy.std_devs(flux_sc)),
    )
    cal_flux_ft, cal_flux_ft_err = map(
        lambda x: np.array(x.tolist()) * flux_model_ft,
        (unumpy.nominal_values(flux_ft), unumpy.std_devs(flux_ft)),
    )

    with fits.open(new_file, "update") as hdul:
        hdul["oi_flux", 10].data["flux"] = cal_flux_sc
        hdul["oi_flux", 10].data["fluxerr"] = cal_flux_sc_err
        hdul["oi_flux", 20].data["flux"] = cal_flux_ft
        hdul["oi_flux", 20].data["fluxerr"] = cal_flux_ft_err
        hdul["oi_flux", 10].columns[4].name = "fluxdata".upper()
        hdul["oi_flux", 10].columns[4].unit = "Jy"
        hdul["oi_flux", 10].columns[5].unit = "Jy"
        hdul["oi_flux", 20].columns[4].name = "fluxdata".upper()
        hdul["oi_flux", 20].columns[4].unit = "Jy"
        hdul["oi_flux", 20].columns[5].unit = "Jy"
        hdul.flush()


# TODO: Add the flags for the flux
def make_vis_gravity_files(directory: Path) -> None:
    """Makes a set of gravity files where the vis is set to be the vis2
    to make it easier to handle for the fitting."""
    plot_dir = directory / "vis" / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)
    for fits_file in tqdm(list(directory.glob("*.fits"))):
        new_file = plot_dir.parent / f"{fits_file.stem}_vis.fits"
        shutil.copy(fits_file, new_file)
        with fits.open(new_file, "update") as hdul:
            hdul["oi_vis2"].data = hdul["oi_vis"].data.copy()
            hdul["oi_vis2"].columns["visamp"].name = "vis2data".upper()
            hdul["oi_vis2"].columns["visamperr"].name = "vis2err".upper()
            hdul.flush()
        plot = Plotter(new_file, save_path=plot_dir)
        plot.add_mosaic().plot(margin=0.3, error=True, save=True)


if __name__ == "__main__":
    directory = Path("/Users/scheuck/Data/reduced_data/hd142527/gravity")
    flux_file = Path("/Users/scheuck/Data/flux_data/hd142527/hd142527_sed_fit.npy")
    for fits_file in tqdm(list(directory.glob("*fits"))):
        calculate_vis(fits_file, flux_file, index=20)
