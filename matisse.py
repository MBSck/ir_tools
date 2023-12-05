import shutil
from pathlib import Path
from typing import Optional

from astropy.io import fits
from matadrs.reduction.avg_oifits import calc_vis_from_corrflux
from matadrs.utils.plot import Plotter
from tqdm import tqdm

from utils import remove_flawed_telescope, average_total_flux, replace_data


# TODO: Make a routine that does all these calculations immediately and makes a finished files folder, too.


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


if __name__ == "__main__":
    directory = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/test/2022-04-21")
    matisse_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    mat_tools_path = matisse_path / "mat_tools"
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/nband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits")
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT.fits")
    flawed_file = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/mat_tools/lband/hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_HAWAII-2RG_FINAL_TARGET_INT_CHOPPED.fits")
    remove_flawed_telescope(flawed_file, ["UT2"], error=True, save=True)
    average_total_flux(mat_tools_path / "nband", error=True, save=True)
    # average_total_flux(mat_tools_path / "lband", error=True, save=True)
