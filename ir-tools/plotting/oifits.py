import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import astropy.units as u
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colormaps as mcm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

from ..utils import get_band


@dataclass
class Data:
    nentry: int = 0
    header: List[fits.Header] = field(default_factory=list)
    wl: List[fits.BinTableHDU] = field(default_factory=list)
    array: List[fits.BinTableHDU] = field(default_factory=list)
    vis: List[fits.BinTableHDU] = field(default_factory=list)
    vis2: List[fits.BinTableHDU] = field(default_factory=list)
    t3: List[fits.BinTableHDU] = field(default_factory=list)


def read_data(fits_files: List[Path] | Path | fits.HDUList) -> Data:
    """Reads the data from the fits files."""
    data = Data()
    try:
        fits_files = fits_files if isinstance(fits_files, list) else [fits_files]
        hduls = [fits.open(fits_file) for fits_file in fits_files]
    except TypeError:
        hduls = fits_files

    for hdul in hduls:
        data.nentry += 1
        data.header.append(hdul[0].header.copy())
        index = 20 if "grav" in data.header[-1]["INSTRUME"].lower() else None
        data.wl.append(
            (hdul["oi_wavelength", index].data["eff_wave"].copy() * u.m).to(u.um)
        )
        for key in ["array", "vis", "vis2", "t3"]:
            if key == "array":
                card = hdul[f"oi_{key}"]
            else:
                card = hdul[f"oi_{key}", index]
            getattr(data, key).append(card.copy())

    [hdul.close() for hdul in hduls]
    return data


def get_unit(self, header: str, sub_header: str) -> str:
    """Fetches the unit of a header by the sub header's name."""
    unit = getattr(self, header)[sub_header.upper()].unit
    return str(unit) if unit is not None else "a.u."


def convert_style_to_colormap(style: str) -> ListedColormap:
    """Converts a style into a colormap."""
    plt.style.use(style)
    colormap = ListedColormap(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    plt.style.use("default")
    return colormap


def get_colormap(colormap: str) -> ListedColormap:
    """Gets the colormap as the matplotlib colormaps or styles."""
    try:
        return mcm.get_cmap(colormap)
    except ValueError:
        return convert_style_to_colormap(colormap)


def get_colorlist(colormap: str, ncolors: int | None) -> List[str]:
    """Gets the colormap as a list from the matplotlib colormaps."""
    return [get_colormap(colormap)(i) for i in range(ncolors)]


# TODO: Rewrite this function to make it easier to read and change -> Also for the readin of the data
# maybe move it to the plot section
# TODO: Make it so that there is always the max amount if the max is reached
# TODO: Rewrite this with just hduls and not the rest
def plot_baselines(
    hduls: List[fits.HDUList],
    band: str,
    observable: str = "vis",
    nplots: int = 12,
    number: bool = False,
    save_dir: Path | None = None,
) -> None:
    """Plots the observables of the model.

    Parameters
    ----------
    hduls : list of fits.HDUList
        A list containing the read in fits files.
    band : str
        The band to plot the data for.
    observable : str, optional
        The observable to plot. "vis", "visphi", "t3" and "vis2" are available.
    nplots : int, optional
        The number of plots to show.
    number : bool, optional
        If the plots should be numbered.
    save_dir : Path, optional
        The save directory for the plots.
    """
    overplot_model = observable != "visphi"
    save_dir = Path.cwd() if save_dir is None else save_dir
    bands = np.array(list(map(get_band, wavelength_range)))
    if band in ["lband", "mband"]:
        band_ind = np.where((bands == "lband") | (bands == "mband"))[0]
    else:
        band_ind = np.where(bands == band)[0]

    data = getattr(OPTIONS.data, "vis" if observable in ["vis", "visphi"] else observable)
    baseline_ind = np.where(~np.any(data.value[band_ind].mask, axis=0))[0]
    if observable in ["vis", "visphi"]:
        baselines, psi = compute_effective_baselines(data.ucoord, data.vcoord)
        names = data.baselines
    else:
        baselines, psi = compute_effective_baselines(
            data.u123coord, data.v123coord, longest=True
        )
        names = data.triangles

    baselines = baselines[1:][baseline_ind]
    psi = psi.to(u.deg)[1:][baseline_ind]

    names = [names[i] for i in baseline_ind]
    wl_data = [data.raw_wavelengths[i] for i in baseline_ind]

    if observable == "visphi":
        raw_value = [data.raw_visphi[i] for i in baseline_ind]
        raw_err = [data.raw_visphierr[i] for i in baseline_ind]
    else:
        raw_value = [data.raw_value[i] for i in baseline_ind]
        raw_err = [data.raw_err[i] for i in baseline_ind]

    # TODO: Switch the ranges here
    wavelength_range = wavelength_range[band_ind]
    wavelength = np.linspace(wl_data[0][0], wl_data[0][-1], OPTIONS.plot.dim)

    if overplot_model:
        _, vis_model, t3_model = compute_observables(
            components, wavelength=wavelength * u.um
        )
        model_data = vis_model if observable == "vis" else t3_model
        model_data = model_data[:, baseline_ind]

    percentiles = np.linspace(0, 100, nplots)
    percentile_ind = percentile_indices(baselines, percentiles)
    baselines, psi = baselines[percentile_ind], psi[percentile_ind]
    wl_data = [wl_data[i] for i in percentile_ind]
    raw_value = [raw_value[i] for i in percentile_ind]
    raw_err = [raw_err[i] for i in percentile_ind]
    names = [names[i] for i in percentile_ind]

    fig, axarr = plt.subplots(
        *get_best_plot_arrangement(nplots) * 4,
        figsize=figsize,
        facecolor=OPTIONS.plot.color.background,
        sharex=True,
        constrained_layout=True,
    )
    axarr = axarr.flatten()
    if observable == "vis":
        y_label = r"$F_{\nu,\,\mathrm{corr.}}$ (Jy)"
        ylims = [0, None]
    elif observable == "visphi":
        y_label = r"$\phi_{\mathrm{diff.}}$ ($^\circ$)"
        ylims = None
    elif observable == "t3":
        y_label = r"$\phi_{\mathrm{cl.}}$ ($^\circ$)"
        ylims = None
    else:
        y_label = "$V^2$ (a.u.)"
        ylims = [0, 1]

    for index, (baseline, baseline_angle) in enumerate(zip(baselines, psi)):
        ax = axarr[index]
        set_axes_color(ax, OPTIONS.plot.color.background)
        if overplot_model:
            ax.plot(
                wavelength,
                model_data[:, percentile_ind[index]],
                label="Model",
            )
        label = rf"{names[index]}, B={baseline.value:.2f} m, $\phi$={baseline_angle.value:.2f}$^\circ$"
        line = ax.plot(
            wl_data[index],
            raw_value[index],
            label=label,
        )
        ax.fill_between(
            wl_data[index],
            raw_value[index] + raw_err[index],
            raw_value[index] - raw_err[index],
            color=line[0].get_color(),
            alpha=0.5,
        )
        ax.set_ylim(ylims)
        if number:
            ax.text(
                0.05,
                0.95,
                str(index + 1),
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )
        ax.legend()

    [ax.remove() for index, ax in enumerate(axarr.flatten()) if index >= nplots]
    fig.subplots_adjust(left=0.2, bottom=0.2)
    # TODO: Reimplement this
    # fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
    # fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
    plt.savefig(save_dir / f"{observable}_{band}.pdf", format="pdf")
    plt.close()


def plot_uv(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the uv coverage."""
    handles = []
    colors = get_colorlist("tab20", data.nentry)
    if index is not None:
        ucoords = data.vis2[index].data["ucoord"]
        vcoords = data.vis2[index].data["vcoord"]
        color = colors[index]
        ax.plot(ucoords, vcoords, "x", markersize=6, markeredgewidth=2, color=color)
        ax.plot(-ucoords, -vcoords, "x", markersize=6, markeredgewidth=2, color=color)
    else:
        ucoords, vcoords = [], []
        for file_idx, (header, vis2) in enumerate(zip(data.header, data.vis2)):
            color = colors[file_idx]
            file_ucoords = vis2.data["ucoord"]
            file_vcoords = vis2.data["vcoord"]
            ucoords.extend(file_ucoords)
            vcoords.extend(file_vcoords)

            ax.plot(
                file_ucoords,
                file_vcoords,
                "x",
                markersize=6,
                markeredgewidth=2,
                color=color,
            )
            ax.plot(
                -file_ucoords,
                -file_vcoords,
                "x",
                markersize=6,
                markeredgewidth=2,
                color=color,
            )
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="X",
                    linestyle="None",
                    label=header["date-obs"].split("T")[0],
                )
            )

        ucoords, vcoords = np.array(ucoords), np.array(vcoords)

    sorted_idx = np.argsort(np.hypot(ucoords, vcoords))
    for idx, (ucoord, vcoord) in enumerate(
        zip(ucoords[sorted_idx], vcoords[sorted_idx]), start=1
    ):
        ax.text(
            ucoord + 3.5,
            vcoord + 3.5,
            idx,
            fontsize="small",
            color="0",
            alpha=0.8,
        )

    ax.plot([0.0], [0.0], "+k", markersize=5, markeredgewidth=2, alpha=0.5)
    xlabel, ylabel = "$u$ (m) - South", "$v$ (m) - East"
    ax.legend(handles=handles, fontsize="small")

    plt.gca().invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_data(
    ax: Axes, x: np.ndarray, y: np.ndarray, yerr: np.ndarray, flag: np.ndarray
) -> Axes:
    """Plots some data with errors."""
    ys = np.ma.masked_array(y, mask=flag)
    yerrs = np.ma.masked_array(yerr, mask=flag)
    for y, yerr in zip(ys, yerrs):
        ax.plot(x.value, y)
        ax.fill_between(x.value, y + yerr, y - yerr, alpha=0.2)
    return ax


# TODO: Finish this here
# TODO: Move all this here to paper
def plot_flux(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the flux."""
    card = hdul["oi_flux", index]
    key = "fluxdata" if "FLUXDATA" in card.columns.names else "flux"
    ax = plot_data(ax, x, card.data[key], card.data["fluxerr"], card.data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_vis(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the visibility or the correlated flux."""
    x, dataset = data.wl[index], data.vis[index].data
    plot_data(ax, x, dataset["visamp"], dataset["visamperr"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_visphi(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the differential phases."""
    x, dataset = data.wl[index], data.vis[index].data
    plot_data(ax, x, dataset["visphi"], dataset["visphierr"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def plot_vis2(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the squared visibility."""
    x, dataset = data.wl[index], data.vis2[index].data
    plot_data(ax, x, dataset["vis2data"], dataset["vis2err"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim((0, None) if ax.get_ylim()[1] < 1 else (0, 1))
    return ax


def plot_t3(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the closure phases."""
    x, dataset = data.wl[index], data.t3[index].data
    plot_data(ax, x, dataset["t3phi"], dataset["t3phierr"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{cp}}$ $\left(^\circ\right)$")
    return ax


def plot(
    fits_files: List[Path] | Path,
    plots: List[str] | str = "all",
    kind: str = "collage",
    cell_width: int = 4,
    save_dir: Path | None = None,
) -> None:
    """Plots all the specified observables in a collage.

    Parameters
    ----------
    kind : str
        The kinds of plots. "individual", "combined" and "collage" are available.
    """
    data = read_data(fits_files)
    module = sys.modules[__name__]
    if plots == "all":
        plots = ["uv", "flux", "t3", "vis2", "vis", "visphi"]

    if kind == "collage":
        cols, rows = len(plots), int(np.ceil(len(data.vis) / len(plots)))
        figsize = (cols * cell_width, rows * cell_width)
        _, axarr = plt.subplots(
            rows, cols, figsize=figsize, sharex=True, constrained_layout=True
        )
        for row_index, ax_row in enumerate(axarr):
            for ax_col, plot in zip(ax_row, plots):
                getattr(module, f"plot_{plot}")(ax_col, data, row_index)

    elif kind == "combined":
        cols, rows = len(plots), 1
        figsize = (cols * cell_width, rows * cell_width)
        _, axarr = plt.subplots(
            rows, cols, figsize=figsize, sharex=True, constrained_layout=True
        )
        axarr = [axarr] if not isinstance(axarr, np.ndarray) else axarr
        for ax, plot in zip(axarr, plots):
            getattr(module, f"plot_{plot}")(ax, data)

    # TODO: Implement the individual plots again
    else:
        ...

    # TODO: Reimplement different plots from this (as in saved in different files)
    if save_dir is not None:
        plt.savefig(save_dir, format=save_dir.suffix[1:], dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting" / "hd142527"
    plot_dir = path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    breakpoint()

