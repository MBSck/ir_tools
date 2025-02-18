import sys
from pathlib import Path
from typing import Any, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colormaps as mcm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray

# from tqdm import tqdm
from ..utils import get_plot_layout, transform_coordinates
from . import io

CARD_KEYS = {
    "flux": ["fluxdata", "fluxerr"],
    "vis": ["visamp", "visamperr"],
    "vis2": ["vis2data", "vis2err"],
    "visphi": ["visphi", "visphierr"],
    "t3": ["t3phi", "t3phierr"],
}


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


# TODO: Reimplement individual uv plots (for a certain index)
def plot_uv(
    ax: Axes, hduls: List[fits.HDUList], color_by: str = "file", number: bool = False, **kwargs
) -> Axes:
    """Plots the uv coverage.

    Parameters
    ----------
    hduls : list of astropy.io.fits.HDUList

    Returns
    -------
    ax : Axes
    """
    if color_by == "file":
        colors = get_colorlist("tab20", len(hduls))
        labels = [io.get_header_entry(hdul, "date") for hdul in hduls]
    else:
        instruments = [io.get_header_entry(hdul, "instrument") for hdul in hduls]
        labels = []
        for entry in instruments:
            if entry not in labels:
                labels.append(entry)
        colors = dict(zip(labels, get_colorlist("tab20", len(labels))))

    file_letters = io.get_labels(hduls)
    handles, names, ucoords, vcoords = [], [], [], []
    for index, hdul in enumerate(hduls):
        x = io.get_column(hdul, "oi_vis2", "ucoord")
        y = io.get_column(hdul, "oi_vis2", "vcoord")
        names.extend([f"{file_letters[index]}.{i + 1}" for i in np.arange(x.size)])
        ucoords.extend(x)
        vcoords.extend(y)

        if isinstance(colors, dict):
            instrument = io.get_header_entry(hdul, "instrument")
            color = colors[instrument]
        else:
            color = colors[index]
            label = io.get_header_entry(hdul, "date")

        line = ax.plot(
            x,
            y,
            "x",
            markersize=5,
            markeredgewidth=1,
            color=color,
        )
        ax.plot(
            -x,
            -y,
            "x",
            markersize=5,
            markeredgewidth=1,
            color=line[0].get_color(),
        )

    for label, color in zip(
        labels, colors.values() if isinstance(colors, dict) else colors
    ):
        handles.append(
            mlines.Line2D(
                [],
                [],
                color=color,
                marker="X",
                linestyle="None",
                label=label,
                markersize=4,
                markeredgewidth=1,
            )
        )

    labels, ucoords, vcoords = np.array(names), np.array(ucoords), np.array(vcoords)
    sorted_idx = np.argsort(np.hypot(ucoords, vcoords))
    if number:
        for name, ucoord, vcoord in zip(
            labels[sorted_idx], ucoords[sorted_idx], vcoords[sorted_idx]
        ):
            ax.text(
                ucoord + 3.5,
                vcoord + 3.5,
                name,
                fontsize=4,
                color="0",
                alpha=0.8,
            )

    ax.plot([0.0], [0.0], "+k", markersize=4, markeredgewidth=1, alpha=0.5)
    xlabel, ylabel = "$u$ (m) - South", "$v$ (m) - East"
    ax.legend(handles=handles, fontsize="small")

    plt.gca().invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-150, 150])
    ax.set_xlim([150, -150])


# TODO: Re-implement model overplotting for this function. Shouldn't be too hard
# TODO: Include this in the plot class of this module
def plot_vs_spf(
    files_or_hduls: Path | fits.HDUList | List[Path] | List[fits.HDUList],
    band: str,
    observable: str = "vis",
    ylims: List[float] | None = None,
    max_plots: int = 20,
    number: bool = False,
    save_dir: Path | None = None,
    **kwargs,
) -> None:
    """Plots the observables of the model.

    Parameters
    ----------
    fits_files : list of pathlib.Path or pathlib.Path
        A list of fits files or a fits file to read the data from.
    bands : str
        The band to plot the data for.
    observable : str, optional
        The observable to plot. "vis", "visphi", "t3" and "vis2" are available.
    ylims : list of float, optional
        The y-axis limits for the plots.
    max_plots : int, optional
        The maximal number of plots to show.
    number : bool, optional
        If the plots should be numbered.
    save_dir : Path, optional
        The save directory for the plots.
    """
    save_dir = Path.cwd() if save_dir is None else save_dir
    hduls = io.read(files_or_hduls)
    file_letter = io.get_labels(hduls)

    wls, stations, labels = [], [], []
    vals, errs, spfs, psis = [], [], [], []
    for index, hdul in enumerate(hduls):
        val_key, err_key = CARD_KEYS[observable]
        card_key = observable if observable != "visphi" else "vis"
        val = io.get_column(hdul, f"oi_{card_key}", val_key, masked=True)
        err = io.get_column(hdul, f"oi_{card_key}", err_key, masked=True)
        label = [f"{file_letter[index]}.{i + 1}" for i in range(val.shape[0])]
        sta_index = io.get_column(hdul, f"oi_{card_key}", "sta_index")
        sta_index_to_name = dict(
            zip(
                io.get_column(hdul, "oi_array", "sta_index").tolist(),
                io.get_column(hdul, "oi_array", "sta_name"),
            )
        )
        station = list(
            map(lambda x: "-".join(x), np.vectorize(sta_index_to_name.get)(sta_index))
        )
        wls.extend(
            [io.get_column(hdul, "oi_wavelength", "eff_wave") for _ in range(len(val))]
        )
        if observable in ["vis", "visphi"]:
            x = io.get_column(hdul, "oi_vis", "ucoord")
            y = io.get_column(hdul, "oi_vis", "vcoord")
        if observable == "t3":
            x1, x2 = map(
                lambda x: io.get_column(hdul, "oi_t3", x), ["u1coord", "u2coord"]
            )
            y1, y2 = map(
                lambda x: io.get_column(hdul, "oi_t3", x), ["v1coord", "v2coord"]
            )
            x123, y123 = np.array([x1, x2, x1 + x2]), np.array([y1, y2, y1 + y2])
            spf = np.hypot(x123, y123)
            longest_ind = (
                np.arange(spf.T.shape[0]),
                np.argmax(spf.T, axis=1),
            )
            x, y = x123.T[longest_ind], y123.T[longest_ind]

        ut, vt = transform_coordinates(x, y)
        spf, psi = np.hypot(ut, vt), np.rad2deg(np.arctan2(ut, vt))

        vals.extend(val)
        errs.extend(err)
        spfs.extend(spf)
        psis.extend(psi)
        labels.extend(label)
        stations.extend(station)

    nplots = max_plots if len(vals) > max_plots else len(vals)
    baseline_ind = np.argsort(spfs)
    wls, labels, vals, errs, spfs, psis, stations = (
        [wls[i] for i in baseline_ind],
        [labels[i] for i in baseline_ind],
        [vals[i] for i in baseline_ind],
        [errs[i] for i in baseline_ind],
        np.array(spfs)[baseline_ind],
        np.array(psis)[baseline_ind],
        np.array(stations)[baseline_ind],
    )

    percentile_ind = np.percentile(
        np.arange(len(vals)), np.linspace(0, 100, nplots)
    ).astype(int)
    wls, labels, vals, errs, spfs, psis, stations = (
        [wls[i] for i in percentile_ind],
        [labels[i] for i in percentile_ind],
        [vals[i] for i in percentile_ind],
        [errs[i] for i in percentile_ind],
        spfs[percentile_ind],
        psis[percentile_ind],
        stations[percentile_ind],
    )
    label_colors = np.unique([label.split(".")[0] for label in labels])
    label_colors = dict(zip(label_colors, get_colorlist("tab20", len(label_colors))))

    rows, cols = get_plot_layout(nplots)
    fig, axarr = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4, rows * 4),
        sharex=True,
        constrained_layout=True,
    )

    ymin, ymax = 0, 0
    for index, (ax, b, psi) in enumerate(zip(axarr.flat, spfs, psis)):
        ymax = np.max(vals[index]) if np.max(vals[index]) > ymax else ymax
        ymin = np.min(vals[index]) if np.min(vals[index]) < ymin else ymin
        line = ax.plot(
            wls[index],
            vals[index],
            color=label_colors[labels[index].split(".")[0]],
            label=rf"{stations[index]}, B={b:.2f} m, $\psi$={psi:.2f}$^\circ$",
        )
        ax.fill_between(
            wls[index],
            vals[index] + errs[index],
            vals[index] - errs[index],
            color=line[0].get_color(),
            alpha=0.5,
        )
        if number:
            ax.text(
                0.05,
                0.95,
                labels[index],
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
                va="top",
                ha="left",
            )
        ax.legend()

    # TODO: Reimplement this
    # fig.text(0.5, 0.04, r"$\lambda$ ($\mathrm{\mu}$m)", ha="center", fontsize=16)
    # fig.text(0.04, 0.5, y_label, va="center", rotation="vertical", fontsize=16)
    if ylims is None:
        ylims = [ymin - ymin * 0.25, ymax + ymax * 0.25]

    if observable == "vis":
        y_label = r"$F_{\nu,\,\mathrm{corr.}}$ (Jy)"
        ylims[0] = 0
    elif observable == "visphi":
        y_label = r"$\phi_{\mathrm{diff.}}$ ($^\circ$)"
    elif observable == "t3":
        y_label = r"$\phi_{\mathrm{cl.}}$ ($^\circ$)"

    [ax.set_ylim(ylims) for ax in axarr.flat]
    [ax.remove() for index, ax in enumerate(axarr.flat) if index >= nplots]
    # fig.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(save_dir / f"{observable}_{band}.pdf", format="pdf", transparent=True)
    plt.close()


def plot_data(
    ax: Axes,
    x: NDArray[Any],
    ys: NDArray[Any],
    yerrs: NDArray[Any],
    label: str | None = None,
    **kwargs,
) -> Axes:
    """Plots some data with errors."""
    for y, yerr in zip(ys, yerrs):
        ax.plot(x, y, label=label)
        ax.fill_between(x, y + yerr, y - yerr, alpha=0.2)
    return ax


def plot_flux(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the flux."""
    ax = plot_data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    ax.set_ylim(bottom=0)
    return ax


def plot_vis(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the visibilites/correlated fluxes."""
    """Plots the visibility or the correlated flux."""
    ax = plot_data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    ax.set_ylim(bottom=0)
    return ax


def plot_visphi(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the differential phases."""
    ax = plot_data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def plot_vis2(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the squared visibilities."""
    ax = plot_data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim(bottom=0, top=1)
    return ax


def plot_t3(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the closure phases."""
    ax = plot_data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{cp}}$ $\left(^\circ\right)$")
    return ax


def plot(
    files_or_hduls: Path | fits.HDUList | List[Path] | List[fits.HDUList],
    plots: List[str] | str = "all",
    bands: str = "all",
    kind: str = "collage",
    cell_width: int = 4,
    save_dir: Path | None = None,
    transparent: bool = False,
    **kwargs,
) -> None:
    """Plots all the specified observables in a collage.

    Parameters
    ----------
    kind : str
        The kinds of plots. "individual", "combined" and "collage" are available.
    """
    hduls = io.read(files_or_hduls)
    module = sys.modules[__name__]
    if plots == "all":
        plots = ["uv", "flux", "t3", "vis2", "vis", "visphi"]

    if kind == "collage":
        cols, rows = len(plots), len(hduls)
        figsize = (cols * cell_width, rows * cell_width)
        _, axarr = plt.subplots(
            rows, cols, figsize=figsize, sharex=True, constrained_layout=True
        )
        for row_index, row in enumerate(axarr):
            hdul = hduls[row_index]
            x = io.get_column(hdul, "oi_wavelength", "eff_wave")
            for ax, key in zip(row, plots):
                if key == "uv":
                    args = (ax, [hdul])
                else:
                    val_key, err_key = CARD_KEYS[key]
                    card_key = key if key != "visphi" else "vis"
                    y = io.get_column(hdul, f"oi_{card_key}", val_key, masked=True)
                    yerr = io.get_column(hdul, f"oi_{card_key}", err_key, masked=True)
                    sta_index = io.get_column(hdul, f"oi_{card_key}", "sta_index")
                    sta_index_to_name = dict(
                        zip(
                            io.get_column(hdul, "oi_array", "sta_index").tolist(),
                            io.get_column(hdul, "oi_array", "sta_name"),
                        )
                    )
                    label = list(
                        map(
                            lambda x: "-".join(x),
                            np.vectorize(sta_index_to_name.get)(sta_index),
                        )
                    )
                    args = (ax, x, y, yerr, label)

                getattr(module, f"plot_{key}")(*args, **kwargs)

    elif kind == "combined":
        cols, rows = len(plots), 1
        figsize = (cols * cell_width, rows * cell_width)
        _, axarr = plt.subplots(
            rows, cols, figsize=figsize, sharex=True, constrained_layout=True
        )
        axarr = [axarr] if not isinstance(axarr, np.ndarray) else axarr
        for ax, key in zip(axarr, plots):
            if key == "uv":
                args = (ax, hduls)
            # TODO: Implement this
            else:
                ...

            getattr(module, f"plot_{key}")(*args, **kwargs)

    # TODO: Implement the individual plots again
    else:
        ...

    # TODO: Reimplement different plots from this (as in saved in different files)
    if save_dir is not None:
        plt.savefig(
            save_dir, format=save_dir.suffix[1:], dpi=300, transparent=transparent
        )
    else:
        plt.show()

    plt.close()
