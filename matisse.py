import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from matadrs.reduction.avg_oifits import calc_vis_from_corrflux
from matadrs.utils.plot import Plotter
from tqdm import tqdm



def replace_data(file: Path,
                 file_for_replace: Path,
                 header: str, sub_headers: List[str]) -> None:
    """Replaces some data of a specific header and sub headers
    of file with the same of a different file.

    Useful when one part of data is corrupted, but all needs to be fitted.
    """
    with fits.open(file_for_replace) as hdul:
        data_to_inject = hdul[header].data

    with fits.open(file, "update") as hdul:
        data = hdul[header].data
        header = hdul[header].header
        if "INJECT" in header:
            print("[SKIPPING]: There is already data"
                  f"injected from another file for the '{header}' header.")
            return
        for sub_header in sub_headers:
            data[sub_header] = data_to_inject[sub_header]
        header["INJECT"] = f"Data injected from {file_for_replace.name}"
        hdul.flush()


def flip_phases(directory: Optional[Path] = None,
                excluded: Optional[List[Path]] = None) -> None:
    """Flips the phase of the t3phi and writes it in the header.

    Required for old versions (<=1.7.8) of the MATISSE data reduction pipeline
    which had flipped the N-band phase only by 180 degrees.
    """
    directory = Path.cwd() if directory is None else Path(directory)
    excluded = list(map(lambda x: x.name, map(Path, excluded)))\
            if excluded is not None else None
    for fits_file in tqdm(list(directory.glob("*AQUARIUS*"))):
        if excluded is not None:
            if fits_file.name in excluded:
                continue
        with fits.open(fits_file, "update") as hdul:
            header = hdul["oi_t3"].header
            if "PFLIP" in header:
                continue
            t3phi = hdul["oi_t3"].data["t3phi"]
            hdul["oi_t3"].data["t3phi"] = -t3phi
            header["PFLIP"] = True
            hdul.flush()


def combine_chopped_non_chopped(directory: Optional[Path] = None) -> None:
    """Gets the total flux from the chopped (.fits)-files
    and puts it in the non-chopped (.fits)-files.
    """
    directory = Path.cwd() if directory is None else Path(directory)
    combined_dir = directory / "combined"
    if not combined_dir.exists():
        combined_dir.mkdir(parents=True)

    for fits_file in tqdm(list(directory.glob("*HAWAII*INT.fits"))):
        chopped_file = fits_file.parent / fits_file.name.replace(".fits", "_CHOPPED.fits")
        new_file = combined_dir / f"{fits_file.stem}_combined.fits"
        shutil.copy(fits_file, new_file)
        if not chopped_file.exists():
            continue
        with fits.open(chopped_file, "readonly") as hdul:
            oi_flux = hdul["oi_flux"].data
        with fits.open(new_file, "update") as hdul:
            header = hdul["oi_flux"].header
            if "FLXCHOP" in header:
                continue
            header["FLXCHOP"] = f"Chopped flux from the file {fits_file.name}"\
                "added as the total flux"
            hdul["oi_flux"].data = oi_flux
            hdul.flush()


def average_total_flux(directory: Optional[Path] = None) -> None:
    """Averages toegether the fluxes for the 'mat_tools' reduction."""
    directory = Path.cwd() if directory is None else Path(directory)
    flux_dir = directory / "flux"
    if not flux_dir.exists():
        flux_dir.mkdir(parents=True)
    for fits_file in list(directory.glob("*.fits")):
        new_file = flux_dir / f"{fits_file.stem}_avg.fits"
        shutil.copy(fits_file, new_file)
        with fits.open(new_file, "update") as hdul:
            oi_flux = hdul["oi_flux"].data
            flux, fluxerr = oi_flux["fluxdata"], oi_flux["fluxerr"]
            avg_flux = np.mean(flux, axis=0)
            avg_fluxerr = np.sqrt(np.hypot(np.nanstd(flux, axis=0),
                                           np.nanmean(fluxerr, axis=0)))
            hdul["oi_flux"].data = hdul["oi_flux"].data[:1]
            hdul["oi_flux"].data["fluxdata"] = avg_flux
            hdul["oi_flux"].data["fluxerr"] = avg_fluxerr
            hdul.flush()


def calculate_vis(directory: Optional[Path] = None,
                  propagate_fluxerr: Optional[bool] = True,
                  **kwargs) -> None:
    """Calculates vis from corrflux."""
    directory = Path.cwd() if directory is None else Path(directory)
    vis_dir = directory / "vis"
    if not vis_dir.exists():
        vis_dir.mkdir(parents=True)

    for fits_file in list(directory.glob("*.fits")):
        new_file = vis_dir / f"{fits_file.stem}_vis.fits"
        calc_vis_from_corrflux(fits_file, fits_file, new_file,
                               propagate_fluxerr=propagate_fluxerr)
        plot = Plotter(new_file, save_path=directory / "vis")
        unwrap = True if "AQUARIUS" in fits_file.name else False
        plot.add_uv().add_flux().add_vis2().add_cphases(unwrap=unwrap).plot(**kwargs)


# TODO: Make a routine that does all these calculations immediately and makes a finished files folder, too.


if __name__ == "__main__":
    # file = Path("mat_tools/nband/HD_142666_2019-05-14T05_28_03_N_TARGET_FINALCAL_INT.fits")
    # file_for_flux = Path("mat_tools/nband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits")
    # replace_data(file, file_for_flux, "oi_flux", ["fluxdata", "fluxerr"])
    # flip_phases(directory="mat_tools/nband")
    # get_chopped_flux(excluded=excluded)
    # average_total_flux("mat_tools/nband")
    directory = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/test/2022-04-21")
    matisse_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    mat_tools_path = matisse_path / "mat_tools"
    # combine_chopped_non_chopped(mat_tools_path / "lband")
    average_total_flux(directory)
    calculate_vis(directory / "flux", margin=0.3, error=True,
                  save=True, propagate_fluxerr=False)
