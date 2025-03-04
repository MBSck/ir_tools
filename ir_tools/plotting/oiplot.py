import sys
from pathlib import Path
from typing import Any, Callable, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colormaps as mcm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from numpy.typing import NDArray
from tqdm import tqdm

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


def _uv(
    ax: Axes,
    hduls: List[fits.HDUList],
    color_by: str = "file",
    number: bool = False,
    legend: bool = True,
    **kwargs,
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
        labels = [io._get_header_entry(hdul, "date") for hdul in hduls]
    else:
        instruments = [io._get_header_entry(hdul, "instrument") for hdul in hduls]
        labels = []
        for entry in instruments:
            if entry not in labels:
                labels.append(entry)
        colors = dict(zip(labels, get_colorlist("tab20", len(labels))))

    file_letters = io.get_labels(hduls)
    handles, names, ucoords, vcoords = [], [], [], []
    for index, hdul in enumerate(hduls):
        x = io._get_column(hdul, "oi_vis2", "ucoord")
        y = io._get_column(hdul, "oi_vis2", "vcoord")
        names.extend([f"{file_letters[index]}.{i + 1}" for i in np.arange(x.size)])
        ucoords.extend(x)
        vcoords.extend(y)

        if isinstance(colors, dict):
            instrument = io._get_header_entry(hdul, "instrument")
            color = colors[instrument]
        else:
            color = colors[index]
            label = io._get_header_entry(hdul, "date")

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
    if legend:
        ax.legend(handles=handles, fontsize="small")

    plt.gca().invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([-150, 150])
    ax.set_xlim([150, -150])


# TODO: Include this in the plot class of this module
def vs_spf(
    files_or_hduls: Path | fits.HDUList | List[Path] | List[fits.HDUList],
    band: str,
    observable: str = "vis",
    ylims: List[float] | None = None,
    max_plots: int = 20,
    number: bool = False,
    legend: bool = True,
    transparent: bool = False,
    model_func: Callable | None = None,
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
    transparent : bool, optional
        If the plots should be transparent.
    model_func : callable, optional
        The model function to calculate the observable for.
        Needs to takes as arguments the u and v coordinates and the wavelength.
        It needs to return the combined visibilities/correlated fluxes, the individual
        component names as well as the complex visibilities of the individual components.
        #TODO: Also relay the shapes here the arrays have to be in.
    save_dir : Path, optional
        The save directory for the plots.
    """
    hduls = io.read(files_or_hduls)
    file_letter = io.get_labels(hduls)

    wls, stations, labels = [], [], []
    vals, errs, spfs, psis = [], [], [], []
    model_vals, model_comp_vals, model_comp_labels = [], [], []
    for index, hdul in enumerate(hduls):
        val_key, err_key = CARD_KEYS[observable]
        card_key = observable if observable != "visphi" else "vis"
        val = io._get_column(hdul, f"oi_{card_key}", val_key, masked=True)
        err = io._get_column(hdul, f"oi_{card_key}", err_key, masked=True)
        label = [f"{file_letter[index]}.{i + 1}" for i in range(val.shape[0])]
        sta_index = io._get_column(hdul, f"oi_{card_key}", "sta_index")
        sta_index_to_name = dict(
            zip(
                io._get_column(hdul, "oi_array", "sta_index").tolist(),
                io._get_column(hdul, "oi_array", "sta_name"),
            )
        )
        station = list(
            map(lambda x: "-".join(x), np.vectorize(sta_index_to_name.get)(sta_index))
        )
        wls.extend(
            [io._get_column(hdul, "oi_wavelength", "eff_wave") for _ in range(len(val))]
        )
        if observable in ["vis", "vis2", "visphi"]:
            x = io._get_column(hdul, f"oi_{card_key}", "ucoord")
            y = io._get_column(hdul, f"oi_{card_key}", "vcoord")
            if model_func is not None:
                model_val, model_comp_val, model_comp_label = model_func(x, y, wls[-1])
                model_vals.extend(model_val)
                model_comp_vals.extend(model_comp_val)
                model_comp_labels.extend(model_comp_label)

        if observable == "t3":
            x1, x2 = map(
                lambda x: io._get_column(hdul, "oi_t3", x), ["u1coord", "u2coord"]
            )
            y1, y2 = map(
                lambda x: io._get_column(hdul, "oi_t3", x), ["v1coord", "v2coord"]
            )
            x123, y123 = np.array([x1, x2, x1 + x2]), np.array([y1, y2, y1 + y2])
            if model_func is not None:
                model_val, model_comp_val, model_comp_label = model_func(x123, y123, wls[-1])
                model_vals.extend(model_val)
                model_comp_vals.extend(model_comp_val)
                model_comp_labels.extend(model_comp_label)

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
    if model_vals:
        model_vals = [model_vals[i] for i in baseline_ind]
        model_comp_vals, model_comp_labels = (
            [model_comp_vals[i] for i in baseline_ind],
            [model_comp_labels[i] for i in baseline_ind],
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
    if model_vals:
        model_vals = [model_vals[i] for i in percentile_ind]
        model_comp_vals, model_comp_labels = (
            [model_comp_vals[i] for i in percentile_ind],
            [model_comp_labels[i] for i in percentile_ind],
        )

    label_colors = np.unique([label.split(".")[0] for label in labels])
    label_colors = dict(zip(label_colors, get_colorlist("tab20", len(label_colors))))

    rows, cols = get_plot_layout(nplots)
    fig, axarr = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4, rows * 4),
        sharex=True,
        sharey=True,
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
        if model_vals:
            line = ax.plot(
                wls[index],
                model_vals[index],
                color="k",
                label="Model",
            )

            if observable != "t3":
                linestyles = ["--", "-.", ":"]
                for comp_val, comp_label, line_style in zip(
                    model_comp_vals[index], model_comp_labels[index], linestyles
                ):
                    ax.plot(
                        wls[index],
                        comp_val,
                        color="k",
                        label=comp_label,
                        linestyle=line_style,
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
        if legend:
            ax.legend()

    if ylims is None:
        ylims = [ymin - ymin * 0.25, ymax + ymax * 0.25]

    if observable == "vis":
        y_label = r"$F_{\nu,\,\mathrm{corr.}}$ (Jy)"
        ylims[0] = None
    elif observable == "vis2":
        y_label = r"$V^2$ (a.u.)"
        ylims = [None, 1]
    elif observable == "visphi":
        y_label = r"$\phi_{\mathrm{diff.}}$ ($^\circ$)"
    elif observable == "t3":
        y_label = r"$\phi_{\mathrm{cl.}}$ ($^\circ$)"

    for index, ax in enumerate(axarr.flat):
        if index >= nplots:
            ax.remove()
            continue

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(ylims)

    fig.supxlabel(r"$\lambda$ (m)", fontsize=16)
    fig.supylabel(y_label, fontsize=16)
    if save_dir is not None:
        plt.savefig(
            save_dir, format=save_dir.suffix[1:], dpi=300, transparent=transparent
        )
    else:
        plt.show()
    plt.close()


def _data(
    ax: Axes,
    x: NDArray[Any],
    ys: NDArray[Any],
    yerrs: NDArray[Any],
    label: str | None = None,
    linestyle: str | None = None,
    **kwargs,
) -> Axes:
    """Plots some data with errors."""
    for y, yerr in zip(ys, yerrs):
        ax.plot(x, y, label=label, linestyle=linestyle)
        ax.fill_between(x, y + yerr, y - yerr, alpha=0.2)
    return ax


def _flux(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the flux."""
    ax = _data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    ax.set_ylim(bottom=0)
    return ax


def _vis(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the visibilites/correlated fluxes."""
    """Plots the visibility or the correlated flux."""
    ax = _data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    ax.set_ylim(bottom=0)
    return ax


def _visphi(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the differential phases."""
    ax = _data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def _vis2(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the squared visibilities."""
    ax = _data(*args, **kwargs)
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim(bottom=0, top=1)
    return ax


def _t3(*args, **kwargs) -> Axes:
    """Macro of ".plot_data" for the closure phases."""
    ax = _data(*args, **kwargs)
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
            x = io._get_column(hdul, "oi_wavelength", "eff_wave")
            for ax, key in zip(row, plots):
                if key == "uv":
                    args = (ax, [hdul])
                else:
                    val_key, err_key = CARD_KEYS[key]
                    card_key = key if key != "visphi" else "vis"
                    y = io._get_column(hdul, f"oi_{card_key}", val_key, masked=True)
                    yerr = io._get_column(hdul, f"oi_{card_key}", err_key, masked=True)
                    sta_index = io._get_column(hdul, f"oi_{card_key}", "sta_index")
                    sta_index_to_name = dict(
                        zip(
                            io._get_column(hdul, "oi_array", "sta_index").tolist(),
                            io._get_column(hdul, "oi_array", "sta_name"),
                        )
                    )
                    label = list(
                        map(
                            lambda x: "-".join(x),
                            np.vectorize(sta_index_to_name.get)(sta_index),
                        )
                    )
                    args = (ax, x, y, yerr, label)

                getattr(module, f"_{key}")(*args, **kwargs)

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
                getattr(module, f"_{key}")(*args, **kwargs)
            else:
                # TODO: Make more linestyles so it never cuts the list of hduls
                linestyles = ["-", "--", "-.", ":"]
                for linestyle, hdul in zip(linestyles, hduls):
                    val_key, err_key = CARD_KEYS[key]
                    card_key = key if key != "visphi" else "vis"
                    x = io._get_column(hdul, "oi_wavelength", "eff_wave")
                    y = io._get_column(hdul, f"oi_{card_key}", val_key, masked=True)
                    yerr = io._get_column(hdul, f"oi_{card_key}", err_key, masked=True)
                    sta_index = io._get_column(hdul, f"oi_{card_key}", "sta_index")
                    sta_index_to_name = dict(
                        zip(
                            io._get_column(hdul, "oi_array", "sta_index").tolist(),
                            io._get_column(hdul, "oi_array", "sta_name"),
                        )
                    )
                    label = list(
                        map(
                            lambda x: "-".join(x),
                            np.vectorize(sta_index_to_name.get)(sta_index),
                        )
                    )
                    kwargs["linestyle"] = linestyle
                    args = (ax, x, y, yerr, label)
                    getattr(module, f"_{key}")(*args, **kwargs)

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


# TODO: Do the overplotting here
if __name__ == "__main__":
    reduced_dir = Path().home() / "Data" / "reduced"
    target = "M8E-IR"
    data_dir = reduced_dir / target / "matisse" / "treated" / "flagged"
    plot_dir = data_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)

    fits_files = list(data_dir.glob("*.fits"))
    plots = ["flux", "vis2", "t3"]
    for fits_file in tqdm(fits_files, desc="Plotting..."):
        plot(
            fits_file,
            kind="combined",
            plots=plots,
            save_dir=plot_dir / f"{fits_file.stem}.png"
        )
