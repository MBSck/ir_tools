import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
from astropy.io import fits
from matadrs.utils.plot import Plotter
from tqdm import tqdm
from scipy.interpolate import interp1d

from ..utils import average_total_flux


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


def calc_vis_from_corrflux(input_corrflux_file,input_totalflux_file,outfile_path,propagate_fluxerr=True):
    """Taken from Varga code"""
    shutil.copyfile(input_corrflux_file, outfile_path)
    outhdul  = fits.open(outfile_path, mode='update')

    inhdul_corr = fits.open(input_corrflux_file, mode='readonly')
    inhdul_tot = fits.open(input_totalflux_file, mode='readonly')

    wl_flux = inhdul_tot['OI_WAVELENGTH'].data['EFF_WAVE']
    flux = inhdul_tot['OI_FLUX'].data['FLUXDATA'][0]
    fluxerr = inhdul_tot['OI_FLUX'].data['FLUXERR'][0]

    # NOTE: Read correlated spectra
    corrflux = inhdul_corr['OI_VIS'].data['VISAMP']
    corrfluxerr = inhdul_corr['OI_VIS'].data['VISAMPERR']
    wl_corrflux = inhdul_corr['OI_WAVELENGTH'].data['EFF_WAVE']

    if not len(outhdul['OI_VIS'].data['VISAMP'][0]) == len(flux):
        # NOTE: Interpolate the flux data to the wavelengths of the correlated flux
        f = interp1d(wl_flux, flux,kind='cubic')
        flux_resamp = f(wl_corrflux)
        f = interp1d(wl_flux, fluxerr,kind='cubic')
        fluxerr_resamp = f(wl_corrflux)
        flux = flux_resamp
        fluxerr = fluxerr_resamp
        outhdul['OI_FLUX'].data['FLUXDATA'] = flux
        outhdul['OI_FLUX'].data['FLUXERR'] = fluxerr
        
    for k in range(len(outhdul['OI_VIS'].data['VISAMP'])):
        # NOTE: Collect and average matching vis2 data
        vis = (corrflux[k]/flux)
        if propagate_fluxerr:
            viserr = vis*np.sqrt((corrfluxerr[k]/corrflux[k])**2 + (fluxerr/flux)**2)
        else:
            viserr = corrfluxerr[k]/flux[k] #WARNING: this is not how errors should be calculated
        vis2, vis2err = vis**2.0, 2.0*vis*viserr

        outhdul['OI_VIS2'].data['VIS2DATA'][k] = vis2
        outhdul['OI_VIS2'].data['VIS2ERR'][k] = vis2err 
        outhdul['OI_VIS2'].data['STA_INDEX'][k] = inhdul_corr['OI_VIS'].data['STA_INDEX'][k]
        outhdul['OI_VIS2'].data['UCOORD'][k] = inhdul_corr['OI_VIS'].data['UCOORD'][k]
        outhdul['OI_VIS2'].data['VCOORD'][k] = inhdul_corr['OI_VIS'].data['VCOORD'][k]
    outhdul.flush()
    outhdul.close()
    inhdul_corr.close()
    inhdul_tot.close()


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
    matisse_path = Path("/Users/scheuck/Data/reduced_data/hd142527/matisse")
    # matisse_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/old/1.7.6/mat_tools")
    path = matisse_path / "lband"
    average_total_flux(path, error=True, save=True)
    calculate_vis(path / "flux", propagate_fluxerr=False, error=True, save=True)
