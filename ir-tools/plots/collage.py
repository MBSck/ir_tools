import sys
from pathlib import Path
from typing import List, Tuple

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import EarthLocation
from astropy.io import fits
from matplotlib.axes import Axes
from tqdm import tqdm



def sort(fits_files: List[Path], by: str) -> List[Path]:
    """Filters the object's data by the given key.

    The key can be any of the properties of the ReadoutFits class

    Parameters
    ----------
    by : str
        The data to sort by.
    """
    return sorted(fits_files, key=lambda x: getattr(x, by.lower()))


def filter(fits_files: List[Path], by: List[str], contains: List[str]):
    """Filters the object's data by the given key.

    The key can be any of the properties of the ReadoutFits class

    Parameters
    ----------
    fits_files : list of path
    by : str
        The key to filter the data by.
    contains : str
        The string that the key should contain.
    """
    by = [by] if not isinstance(by, List) else by
    contains = [contains] if not isinstance(contains, List) else contains

    # TODO: Switch this with fits_files and open them here?
    for key, contain in zip(by, contains):
        readouts = [readout for readout in readouts if contain.lower() \
            in getattr(readout, key.lower()).lower()]

    return readouts


def get_unit(self, header: str, sub_header: str) -> str:
    """Fetches the unit of a header by the sub header's name."""
    unit = getattr(self, header)[sub_header.upper()].unit
    return str(unit) if unit is not None else "a.u."


def get_baseline_name(hdu, verbose) -> None:
    # TODO: Also pass the array here
    # baselines = np.around(hdu.data["BASELINE"], 2)
    ucoord, vcoord = hdu.data["ucoord"], hdu.data["vcoord"]
    # TODO: Check convention here
    pas = np.around(
        (np.degrees(np.arctan2(vcoord, ucoord)) - 90) * -1, 2)
    breakpoint()


# TODO: Reimplement both of the following functions in a better way
def transform_uv_points(
        baselines: List[float], hour_angle: np.ndarray,
        latitude: u.rad, declination: u.rad) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the earth rotation (synthesis) for the uv-point
    corresponding to the baselines for the input hour angle(s)

    Parameters
    -----------
    baselines : list of float
        The baselines in the following order: Baselines east, -north, -longest.
    hour_angle : numpy.ndarray of float
    latitude : astropy.units.rad
        The latitude of the site
    declination : astropy.units.rad

    Returns
    -------
    u_coords : numpy.ndarray
    v_coords : numpy.ndarray
    """
    baseline_east, baseline_north, baseline_longest = baselines
    ucoord = baseline_east * np.cos(hour_angle) - baseline_north * np.sin(latitude)\
        * np.sin(hour_angle) + baseline_longest * np.cos(latitude) * np.sin(hour_angle)
    vcoord = baseline_east * np.sin(declination) * np.sin(hour_angle)\
        + baseline_north * (np.sin(latitude) * np.sin(declination) * np.cos(hour_angle)
                            + np.cos(latitude) * np.cos(declination)) - baseline_longest * \
        (np.cos(latitude) * np.sin(declination) * np.cos(hour_angle)
         - np.sin(latitude) * np.cos(declination))
    return ucoord, vcoord


def calculate_uv_tracks(baselines: List[np.ndarray],
                        declination: float, airmass_lim: float):
    """Calculates the tracks that uv-coordinates create from the earth rotation
    synthesis at the latitude of paranal.

    Parameters
    ----------
    baselines : list of numpy.ndarray
        The baselines in the following order: Baselines east, -north, -longest.
    declination : float
        The declination of the target.
    airmass_lim : float
        The airmass limit of the target.
    """
    latitude_paranal = EarthLocation.of_site("paranal").geodetic.lat.to(u.rad)
    hamax = np.arccos(abs((1 / airmass_lim-np.sin(latitude_paranal)
                           * np.sin(declination)) / (np.cos(latitude_paranal)
                                                   * np.cos(declination))))
    return transform_uv_points(baselines, np.linspace(-hamax, hamax, 1000),
                               latitude_paranal, declination)

def plot_uv(ax: Axes, hdul, symbol, show_text,
            make_tracks, show_legend, legend_location, legend_size) -> Axes:
    instruments, handles, uv_max = [], [], 0
    data, array = hdul["oi_vis2"].data, hdul["oi_array"].data
    # colors = get_colorlist(OPTIONS.plot.color.colormap, len(readouts) * 6 + 1)
    ucoord, vcoord = data["ucoord"], data["vcoord"]
    sta_indices, sta_index = data["sta_index"], array["sta_index"]
    sta_name, sta_xyz = array["sta_name"], array["staxyz"]
    # TODO: Set the max of the uv-coords in a good way.

    baselines, sta_labels = [], []
    for uv_index, _ in enumerate(ucoord):
        try:
            baseline = sta_xyz[sta_indices[uv_index, 0] == sta_index][0]\
                - sta_xyz[sta_indices[uv_index, 1] == sta_index][0]
            sta_label = sta_name[sta_indices[uv_index, 0] == sta_index][0] + '-'\
                + sta_name[sta_indices[uv_index, 1] == sta_index][0]
        except IndexError:
            baseline, sta_label = [np.nan, np.nan, np.nan], ""
        baselines.append(baseline)
        sta_labels.append(sta_label)

    # if color_by == "file":
    #     color = colors[index]
    #     handles.append(mlines.Line2D(
    #         [], [], color=color, marker="X",
    #         linestyle="None", label=readout.date[:-8]))
    # elif color_by == "instrument":
    #     if readout.instrument not in instruments:
    #         instruments.append(readout.instrument)
    #     color = colors[instruments.index(readout.instrument)]

    for uv_index, (ucoord, vcoord) in enumerate(zip(ucoord, vcoord)):
        ax.plot(ucoord, vcoord, symbol, color=color,
                markersize=10, markeredgewidth=3)
        ax.plot(-ucoord, -vcoord, symbol,
                color=color, markersize=10, markeredgewidth=3)

        if show_text:
            ax.text(-ucoord-3.5, -vcoord-1.5, sta_labels[uv_index],
                    fontsize="small", color='0', alpha=0.8)

        # if make_tracks:
        #     ucoord_track, vcoord_track = calculate_uv_tracks(
        #         baselines[uv_index], readout.dec*np.pi/180, airmass_lim)
        #     ax.plot(ucoord_track, vcoord_track, '-', color='grey', alpha=0.5)
        #     ax.plot(-ucoord_track, -vcoord_track, '-', color='grey', alpha=0.5)

    ax.plot([0.], [0.], '+k', markersize=5, markeredgewidth=2, alpha=0.5)

    # TODO: Implement check or calculation for the orientations
    xlabel, ylabel = "$u$ (m) - South", "$v$ (m) - East"
    # uv_extent = int(uv_max + uv_max*0.25) if uv_extent is None else uv_extent

    # if color_by == "instrument":
    #     handles = []
    #     for index, instrument in enumerate(instruments):
    #         color = colors[index]
    #         handles.append(mlines.Line2D(
    #             [], [], color=color, marker="X",
    #             linestyle="None", label=instrument.upper()))

    if show_legend:
        ax.legend(handles=handles,
                    loc=legend_location, fontsize=legend_size)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_data(ax: Axes, x: np.ndarray, y: np.ndarray,
              yerr: np.ndarray, flag: np.ndarray) -> Axes:
    """Plots some data with errors."""
    ys = np.ma.masked_array(y, mask=flag)
    yerrs = np.ma.masked_array(yerr, mask=flag)
    for y, yerr in zip(ys, yerrs):
        ax.plot(x.value, y)
        ax.fill_between(x.value, y + yerr, y - yerr, alpha=0.2)
    return ax


def plot_flux(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the flux."""
    card = hdul["oi_flux", index]
    key = "fluxdata" if "FLUXDATA" in card.columns.names else "flux"
    ax = plot_data(ax, x, card.data[key], card.data["fluxerr"], card.data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_vis(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the visibility or the correlated flux."""
    data = hdul["oi_vis", index].data
    plot_data(ax, x, data["visamp"], data["visamperr"], data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_vis2(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the squared visibility."""
    data = hdul["oi_vis2", index].data
    plot_data(ax, x, data["vis2data"], data["vis2err"], data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim([0, 1])
    return ax


def plot_visphi(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the differential phases."""
    data = hdul["oi_vis", index].data
    plot_data(ax, x, data["visphi"], data["visphierr"], data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def plot_t3(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the closure phases."""
    data = hdul["oi_t3", index].data
    plot_data(ax, x, data["t3phi"], data["t3phierr"], data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{cp}}$ $\left(^\circ\right)$")
    return ax


# TODO: Add the GRAVITY indices here
# TODO: Add a check for what datasets are contained (PIONIER & GRAVITY) -> usually oi_vis2 and oi_t3
def plot_collage(fits_file: Path, plots: List[str] | str = "all", cols: int = 3,
                 cell_width: int = 4, save_dir: Path | None = None) -> None:
    """Plots all the specified observables in a collage."""
    module = sys.modules[__name__]
    if plots == "all":
        plots = ["uv", "flux", "t3", "vis2", "vis", "visphi"]

    rows = int(np.ceil(len(plots) / cols))
    figsize = (cols * cell_width, rows * cell_width)
    _, axarr = plt.subplots(rows, cols, figsize=figsize,
                            sharex=True, constrained_layout=True)

    index = 20 if "GRAV" in fits_file.stem else None
    with fits.open(fits_file) as hdul:
        x = (hdul["oi_wavelength", index].data["eff_wave"] * u.m).to(u.um)
        for ax, plot in zip(axarr.flatten(), plots):
            ax = getattr(module, f"plot_{plot}")(ax, hdul, x, index)

    [ax.remove() for index, ax in enumerate(axarr.flatten()) if index >= len(plots)]

    if save_dir is not None:
        plt.savefig(save_dir / f"{fits_file.stem}.png", format="png", dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting" / "hd142527" / "flagged"
    for fits_file in tqdm(list(path.glob("*.fits")), desc="Plotting files..."):
        plot_collage(fits_file, ["flux", "t3", "vis2", "vis"], cols=4, save_dir=path / "plots")
