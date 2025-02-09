import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.io import fits


def get_band(
    name_or_limits: str | List[float] | Tuple[float],
) -> str | Tuple[float, float]:
    """Gets either the limits of a band in near or mid-infrared.

    Supported are the H, K, L, M, and N bands.

    Parameters
    ----------
    name_or_limits : str or list of list or tuple of float
        The name of the band or the limits of the band.

    Returns
    -------
    str or tuple of float
        Either the name of the band or the limits of the band.
    """
    if isinstance(name_or_limits, str):
        match name_or_limits:
            case "hband":
                return 1.5, 1.8
            case "kband":
                return 1.9, 2.5
            case "lband":
                return 2.6, 3.99
            case "mband":
                return 4.0, 6.0
            case "lmband":
                return 2.6, 6.0
            case "nband":
                return 7.5, 16.0
            case _:
                raise ValueError(
                    "Band not recognised. Supported inputs can be 'hband', 'kband', 'lband',"
                    "'mband', or 'nband'."
                )
    elif isinstance(name_or_limits, (list, tuple, np.ndarray)):
        match name_or_limits:
            case (wl_min, wl_max) if 1.5 < wl_min < 1.8 and 1.5 < wl_max < 1.8:
                return "hband"
            case (wl_min, wl_max) if 1.9 < wl_min < 2.5 and 1.9 < wl_max < 2.5:
                return "kband"
            case (wl_min, wl_max) if 2.6 < wl_min < 4.0 and 2.6 < wl_max < 4.0:
                return "lband"
            case (wl_min, wl_max) if 4.0 <= wl_min < 6.0 and 4.0 <= wl_max < 6.0:
                return "mband"
            case (wl_min, wl_max) if 2.6 < wl_min < 6.0 and 2.6 < wl_max < 6.0:
                return "lmband"
            case (wl_min, wl_max) if 7.5 < wl_min < 16.0 and 7.5 < wl_max < 16.0:
                return "nband"
            case _:
                raise ValueError(
                    "Band not recognised. Supported ranges are between (1.5, 1.8)"
                    " for 'hband',\n (1.9, 2.5) for 'kband', (2.6, 3.99) for 'lband',"
                    " (4.0, 6.0) for 'mband', and (7.5, 16.0) for 'nband'."
                )
    else:
        raise ValueError(
            "Input of wrong type. Needs to be a string or a list or tuple"
            " or a numpy array of floats."
        )


def compute_stellar_radius(luminosity: u.Lsun, temperature: u.K) -> u.Rsun:
    """Calculates the stellar radius from the luminosity and temperature."""
    luminosity, temperature = (
        u.Quantity(luminosity, u.Lsun),
        u.Quantity(temperature, u.K),
    )
    return np.sqrt(
        luminosity.to(u.W) / (4 * np.pi * const.sigma_sb * temperature**4)
    ).to(u.Rsun)


def angular_to_distance(angular_diameter: u.mas, distance: u.pc) -> u.m:
    """Converts an angular diameter of an object at a certain distance
    from the observer from mas to meters.

    Parameters
    ----------
    angular_diameter : astropy.units.mas
        The angular diameter of an object.
    distance : astropy.units.pc
        The distance to the object.

    Returns
    -------
    diameter : astropy.units.m
        The diameter of the object.

    Notes
    -----
    The formula for the angular diameter small angle approximation is

    .. math:: d = \\delta*D

    where 'd' is the diameter of the object and 'D' is the distance from the
    observer to the object and ..math::`\\delta` is the angular diameter.
    """
    return angular_diameter.to(u.rad).value * distance.to(u.m)


def distance_to_angular(diameter: u.au, distance: u.pc) -> u.mas:
    """Converts an angular diameter of an object at a certain distance
    from the observer from mas to meters.

    Parameters
    ----------
    diameter : astropy.units.au
        The diameter of an object.
    distance : astropy.units.pc
        The distance to the object.

    Returns
    -------
    diameter : astropy.units.mas
        The diameter of the object.

    Notes
    -----
    The formula for the angular diameter small angle approximation is

    .. math:: \\delta = \\frac{d}{D}

    where 'd' is the diameter of the object and 'D' is the distance from the
    observer to the object and ..math::`\\delta` is the angular diameter.
    """
    return ((diameter.to(u.m) / distance.to(u.m)) * u.rad).to(u.mas)


def average_total_flux(directory: Optional[Path] = None, **kwargs) -> None:
    """Averages toegether the fluxes for the 'mat_tools' reduction."""
    directory = Path.cwd() if directory is None else Path(directory)
    plot_dir = directory / "flux" / "plots"
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)
    for fits_file in list(directory.glob("*.fits")):
        with fits.open(fits_file, "readonly") as hdul:
            if "OI_FLUX" not in hdul:
                continue

        new_file = plot_dir.parent / f"{fits_file.stem}_flux_avg.fits"
        shutil.copy(fits_file, new_file)
        with fits.open(new_file, "update") as hdul:
            oi_flux = hdul["oi_flux"].data
            flux, fluxerr = oi_flux["fluxdata"], oi_flux["fluxerr"]
            avg_flux = np.mean(flux, axis=0)
            avg_fluxerr = np.sqrt(
                np.hypot(np.nanstd(flux, axis=0), np.nanmean(fluxerr, axis=0))
            )
            hdul["oi_flux"].data = hdul["oi_flux"].data[:1]
            hdul["oi_flux"].data["fluxdata"] = avg_flux
            hdul["oi_flux"].data["fluxerr"] = avg_fluxerr
            hdul.flush()

        # TODO: Exchange this with new plot functionality (from oifits)
        # or separete it entirely (from this function).
        # plot = Plotter(new_file, save_dir=plot_dir)
        # unwrap = True if "AQUARIUS" in fits_file.name else False
        # plot.add_mosaic(unwrap=unwrap).plot(**kwargs)
