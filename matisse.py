import shutil
from pathlib import Path
from typing import List, Optional

from astropy.io import fits
from matadrs.reduction.avg_oifits import calc_vis_from_corrflux
from matadrs.utils.plot import Plotter
from tqdm import tqdm

from utils import remove_flawed_telescope, average_total_flux, replace_data


# TODO: Make a routine that does all these calculations immediately and makes a finished files folder, too.


def combine_chopped_non_chopped(
        directory: Path, dates: Optional[List[str]] = None, **kwargs) -> None:
    """Gets the total flux from the chopped (.fits)-files
    and puts it in the non-chopped (.fits)-files.

    Parameters
    ----------
    directory : pathlib.Path
        The directory to look in.
    dates : list of str, optional
        The dates of the files to combine.
    """
    directory = Path.cwd() if directory is None else Path(directory)
    plot_dir = directory / "combined" / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    tag = "INT*.fits"
    if dates is not None:
        fits_files = [list(directory.glob(f"*{date}*{tag}")) for date in dates]
        fits_files = [elem for sublist in fits_files for elem in sublist]
    else:
        fits_files = list(directory.glob(f"*{tag}"))
    fits_files = [fits_file for fits_file in fits_files
                  if "CHOPPED" not in fits_file.name]

    for fits_file in tqdm(fits_files):
        chopped_file = fits_file.parent / fits_file.name.replace("INT", "INT_CHOPPED")
        new_file = plot_dir.parent / f"{fits_file.stem}_combined.fits"
        if not chopped_file.exists():
            continue

        shutil.copy(fits_file, new_file)
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
        plot = Plotter(new_file, save_path=plot_dir)
        plot.add_mosaic().plot(**kwargs)


def calculate_vis(directory: Optional[Path] = None,
                  propagate_fluxerr: Optional[bool] = True,
                  **kwargs) -> None:
    """Calculates vis from corrflux."""
    directory = Path.cwd() if directory is None else Path(directory)
    plot_dir = directory / "vis" / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)

    for fits_file in list(directory.glob("*.fits")):
        new_file = plot_dir.parent / f"{fits_file.stem}_vis.fits"
        calc_vis_from_corrflux(fits_file, fits_file, new_file,
                               propagate_fluxerr=propagate_fluxerr)
        plot = Plotter(new_file, save_path=plot_dir)
        unwrap = True if "AQUARIUS" in fits_file.name else False
        plot.add_mosaic(unwrap=unwrap).plot(**kwargs)


if __name__ == "__main__":
    directory = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/test/2022-04-21")
    matisse_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    mat_tools_path = matisse_path / "mat_tools"
    nband_dir = mat_tools_path / "nband"
    lband_dir = mat_tools_path / "lband"
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/nband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits")
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits")
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT_CHOPPED.fits")
    # remove_flawed_telescope(flawed_file, ["UT2"], error=True, save=True)
    # average_total_flux(nband_dir, error=True, save=True)
    # calculate_vis(nband_dir / "flux", propagate_fluxerr=False, error=True, save=True)
    # average_total_flux(lband_dir, error=True, save=True)

    combine_chopped_non_chopped(lband_dir / "flux", ["2023-06-17", "2023-04-30"],
                                error=True, save=True)
    # calculate_vis(lband_dir / "flux",
    #               propagate_fluxerr=False, error=True, save=True)
    calculate_vis(lband_dir / "flux" / "combined",
                  propagate_fluxerr=False, error=True, save=True)
