import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.io import fits


def get_plot_layout(nplots: int) -> Tuple[int, int]:
    """Gets the best plot arrangement for a given number of plots."""
    sqrt_nplots = np.sqrt(nplots)
    rows, cols = int(np.floor(sqrt_nplots)), int(np.ceil(sqrt_nplots))

    while rows * cols < nplots:
        if cols < rows:
            cols += 1
        else:
            rows += 1

    while (rows - 1) * cols >= nplots:
        rows -= 1

    return rows, cols


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
                return 1.5e-6, 1.8e-6
            case "kband":
                return 1.9e-6, 2.5e-6
            case "lband":
                return 2.6e-6, 3.99e-6
            case "mband":
                return 4e-6, 6e-6
            case "nband":
                return 7.5e-6, 16e-6
            case _:
                raise ValueError(
                    "Band not recognised. Supported inputs can be 'hband', 'kband', 'lband',"
                    "'mband', or 'nband'."
                )
    elif isinstance(name_or_limits, (list, tuple, np.ndarray)):
        match name_or_limits:
            case (wl_min, wl_max) if 1.5e-6 <= wl_min and wl_max <= 1.8e-6:
                return "hband"
            case (wl_min, wl_max) if 1.9e-6 <= wl_min and wl_max <= 2.5e-6:
                return "kband"
            case (wl_min, wl_max) if 2.6e-6 <= wl_min and wl_max <= 4e-6:
                return "lband"
            case (wl_min, wl_max) if 4e-6 < wl_min and wl_max < 6e-6:
                return "mband"
            case (wl_min, wl_max) if 7.5e-6 < wl_min and wl_max < 16e-6:
                return "nband"
            case _:
                raise ValueError(
                    "Band not recognised. Supported ranges are between (1.5e-6, 1.8e-6)"
                    " for 'hband',\n (1.9e-6, 2.5e-6) for 'kband', (2.6e-6, 4e-6) for 'lband',"
                    " (4e-6, 6e-6) for 'mband', and (7.5e-6, 16e-6) for 'nband'."
                )
    else:
        raise ValueError(
            "Input of wrong type. Needs to be a string or a list or tuple"
            " or a numpy array of floats."
        )


def convert_coords_to_polar(
    x: float | np.ndarray,
    y: float | np.ndarray,
    cinc: float | None = None,
    pa: float | None = None,
    deg: bool = False,
) -> Tuple[float | np.ndarray, float | np.ndarray]:
    """Calculates the effective baselines from the projected baselines
    in mega lambda.

    Parameters
    ----------
    x: float or numpy.ndarray or astropy.units.Quantity
        The x-coordinate.
    y: float or numpy.ndarray or astropy.units.Quantity
        The y-coordinate.
    cinc: float, optional
        The cosine of the inclination.
    pa: float, optional
        The positional angle of the object (in degree).
    deg : bool, optional
        If True, the angle will be returned in degree.

    Returns
    -------
    distance : float or numpy.ndarray
        Returns the distance to the point.
    angle : float or numpy.ndarray
        Returns the angle of the point (radians or degree
        if "deg=True").
    """
    if pa is not None:
        pa = pa / 180 * np.pi
        xr = x * np.cos(pa) - y * np.sin(pa)
        yr = x * np.sin(pa) + y * np.cos(pa)
    else:
        xr, yr = x, y

    if cinc is not None:
        xr *= cinc

    theta = np.arctan2(xr, yr)
    if deg:
        theta = theta / np.pi * 180
    return np.hypot(xr, yr), theta


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
