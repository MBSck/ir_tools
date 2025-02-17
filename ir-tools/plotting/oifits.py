import sys
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Iterator, List

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib import colormaps as mcm
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap

# from tqdm import tqdm
from ..utils import convert_coords_to_polar, get_band, get_plot_layout


@dataclass
class Dataset:
    header: List[fits.Header] = field(default_factory=list)
    val: List[np.ma.MaskedArray] = field(default_factory=list)
    err: List[np.ma.MaskedArray] = field(default_factory=list)
    x: List[np.ndarray] = field(default_factory=list)
    y: List[np.ndarray] = field(default_factory=list)
    station: List[List[str]] = field(default_factory=list)

    def __len__(self):
        return len(self.header)

    def __getitem__(self, index: int | List[int]):
        if isinstance(index, int):
            index = [index]

        if index[0] < 0 or index[-1] >= self.__len__():
            raise IndexError("Index out of range.")

        dataset = Dataset()
        for content in fields(self):
            setattr(
                dataset, content.name, [getattr(self, content.name)[i] for i in index]
            )

        return dataset

    def __iter__(self) -> Iterator:
        for i in range(len(self.header)):
            yield self.header[i], self.val[i], self.err[i], self.x[i], self.y[
                i
            ], self.station[i]


@dataclass
class Data:
    hduls: List[fits.HDUList] = field(default_factory=list)
    header: List[fits.Header] = field(default_factory=list)
    label: List[str] = field(default_factory=list)
    wl: List[np.ndarray] = field(default_factory=list)
    array: List[fits.BinTableHDU] = field(default_factory=list)
    flux: Dataset = field(default_factory=Dataset)
    vis2: Dataset = field(default_factory=Dataset)
    t3: Dataset = field(default_factory=Dataset)
    vis: Dataset = field(default_factory=Dataset)
    visphi: Dataset = field(default_factory=Dataset)

    def __len__(self):
        return len(self.hduls)

    def __getitem__(self, index: int | List[int]):
        if isinstance(index, int):
            index = [index]

        # FIXME: Gives an error if data is filtered by band that only has that one band
        if index[0] < 0 or index[-1] >= self.__len__():
            raise IndexError("Index out of range.")

        data = Data()
        for content in fields(self):
            if isinstance(getattr(self, content.name), Dataset):
                setattr(data, content.name, getattr(self, content.name)[index])
            else:
                setattr(
                    data, content.name, [getattr(self, content.name)[i] for i in index]
                )

        return data


def get_station_names(
    sta_index_pairs: np.ndarray, sta_index: np.ndarray, tel_name: np.ndarray
):
    """Gets the station names from the "oi_array" sta indices and telescope names."""
    sta_to_tel = dict(zip(sta_index, tel_name))
    return np.array(
        [
            "-".join(arr)
            for arr in np.vectorize(lambda x: sta_to_tel.get(x))(sta_index_pairs)
        ]
    )


def read_data(fits_files: Path | List[Path]) -> Data:
    """Reads the data from the fits files."""
    data = Data()
    fits_files = [fits_files] if isinstance(fits_files, Path) else fits_files
    for index, fits_file in enumerate(fits_files):
        with fits.open(fits_file) as hdul:
            data.hduls.append(hdul.copy())
            data.header.append(hdul[0].header)
            sci_index = 20 if "grav" in data.header[-1]["INSTRUME"].lower() else None
            data.label.append(f"{chr(ord('A') + index // 9)}{(index % 9) + 1}")
            data.wl.append(hdul["oi_wavelength", sci_index].data["eff_wave"])
            for key in ["array", "flux", "vis2", "vis", "t3"]:
                if key == "array":
                    data.array.append(hdul[f"oi_{key}"].copy())
                else:
                    # TODO: Finish this similar to the way it done for visphi
                    if key == "vis" and f"oi_{key}" not in hdul:
                        ...

                    card = hdul[f"oi_{key}", sci_index]
                    if key == "flux":
                        val_key = (
                            "fluxdata" if "FLUXDATA" in card.columns.names else "flux"
                        )
                        err_key = "fluxerr"
                        xcoord, ycoord = [], []
                    elif key == "vis":
                        val_key, err_key = "visamp", "visamperr"
                        xcoord, ycoord = card.data["ucoord"], card.data["vcoord"]
                        if "VISPHI" in card.columns.names:
                            visphi_val = np.ma.masked_array(
                                card.data["visphi"], mask=card.data["flag"]
                            )
                            visphi_err = np.ma.masked_array(
                                card.data["visphierr"], mask=card.data["flag"]
                            )
                            visphi_name = get_station_names(
                                card.data["sta_index"],
                                data.array[-1].data["sta_index"],
                                data.array[-1].data["sta_name"],
                            )
                        else:
                            visphi_val = np.ma.masked_invalid(
                                np.full(data.vis2.val[-1].shape, np.nan)
                            )
                            visphi_err = np.ma.masked_invalid(
                                np.full(data.vis2.val[-1].shape, np.nan)
                            )
                            visphi_name = data.vis2.station[-1].copy()

                        data.visphi.header.append(card.header)
                        data.visphi.val.append(visphi_val)
                        data.visphi.err.append(visphi_err)
                        data.visphi.x.append(xcoord)
                        data.visphi.y.append(ycoord)
                        data.visphi.station.append(visphi_name)
                    elif key == "vis2":
                        val_key, err_key = "vis2data", "vis2err"
                        xcoord, ycoord = card.data["ucoord"], card.data["vcoord"]
                    elif key == "t3":
                        val_key, err_key = "t3phi", "t3phierr"
                        x1coord, x2coord = card.data["u1coord"], card.data["u2coord"]
                        y1coord, y2coord = card.data["v1coord"], card.data["v2coord"]
                        xcoord = np.array([x1coord, x2coord, x1coord + x2coord])
                        ycoord = np.array([y1coord, y2coord, y1coord + y2coord])

                    getattr(data, key).header.append(card.header)
                    getattr(data, key).val.append(
                        np.ma.masked_array(card.data[val_key], mask=card.data["flag"])
                    )
                    getattr(data, key).err.append(
                        np.ma.masked_array(card.data[err_key], mask=card.data["flag"])
                    )
                    getattr(data, key).x.append(xcoord)
                    getattr(data, key).y.append(ycoord)
                    getattr(data, key).station.append(
                        get_station_names(
                            card.data["sta_index"],
                            data.array[-1].data["sta_index"],
                            data.array[-1].data["sta_name"],
                        )
                    )
    return data


def filter_data(data: Data, **kwargs) -> Data:
    """Filters the data based on the kwargs.

    Parameters
    ----------
    bands : list of str or str
    """
    for key, value in kwargs.items():
        if "band" in key:
            if value == "all":
                continue

            if not isinstance(value, list):
                value = [value]

            indices = []
            for index, wl in enumerate(data.wl):
                band = get_band((wl[0], wl[-1]))
                if band in value:
                    indices.append(index)

            data = data[indices]

    return data


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


def plot_data(ax: Axes, x: np.ndarray, ys: np.ndarray, yerrs: np.ndarray) -> Axes:
    """Plots some data with errors."""
    for y, yerr in zip(ys, yerrs):
        ax.plot(x, y)
        ax.fill_between(x, y + yerr, y - yerr, alpha=0.2)
    return ax


# TODO: Finish this here
# TODO: Move all this here to paper
def plot_flux(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the flux."""
    ax = plot_data(ax, data.wl[index], data.flux.val[index], data.flux.err[index])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_vis(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the visibility or the correlated flux."""
    plot_data(ax, data.wl[index], data.vis.val[index], data.vis.err[index])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_visphi(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the differential phases."""
    plot_data(ax, data.wl[index], data.visphi.val[index], data.visphi.err[index])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def plot_vis2(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the squared visibility."""
    plot_data(ax, data.wl[index], data.vis2.val[index], data.vis2.err[index])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim((0, None) if ax.get_ylim()[1] < 1 else (0, 1))
    return ax


def plot_t3(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the closure phases."""
    plot_data(ax, data.wl[index], data.t3.val[index], data.t3.err[index])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{cp}}$ $\left(^\circ\right)$")
    return ax


# TODO: Re-implement model overplotting for this function. Shouldn't be too hard
# TODO: Include this in the plot class of this module
def plot_baselines(
    fits_files: List[Path] | Path,
    band: str,
    observable: str = "vis",
    max_plots: int = 20,
    number: bool = False,
    save_dir: Path | None = None,
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
    max_plots : int, optional
        The maximal number of plots to show.
    number : bool, optional
        If the plots should be numbered.
    save_dir : Path, optional
        The save directory for the plots.
    """
    save_dir = Path.cwd() if save_dir is None else save_dir
    data = filter_data(read_data(fits_files), bands=band)

    wls, stations, labels = [], [], []
    values, errors, baselines, psis = [], [], [], []
    for index, (_, val, err, x, y, name) in enumerate(getattr(data, observable)):
        wls.extend([data.wl[index] for _ in range(len(val))])
        baseline, psi = convert_coords_to_polar(x, y, deg=True)
        if observable == "t3":
            longest_ind = (
                np.arange(baseline.T.shape[0]),
                np.argmax(baseline.T, axis=1),
            )
            baseline, psi = baseline.T[longest_ind], psi.T[longest_ind]

        labels.extend([f"{data.label[index]}.{i+1}" for i in np.arange(baseline.size)])
        values.extend(val)
        errors.extend(err)
        baselines.extend(baseline)
        psis.extend(psi)
        stations.extend(name)

    nplots = max_plots if len(values) > max_plots else len(values)
    baseline_ind = np.argsort(baselines)
    wls, labels, values, errors, baselines, psis, stations = (
        [wls[i] for i in baseline_ind],
        [labels[i] for i in baseline_ind],
        [values[i] for i in baseline_ind],
        [errors[i] for i in baseline_ind],
        np.array(baselines)[baseline_ind],
        np.array(psis)[baseline_ind],
        np.array(stations)[baseline_ind],
    )

    percentile_ind = np.percentile(
        np.arange(len(values)), np.linspace(0, 100, nplots)
    ).astype(int)
    wls, labels, values, errors, baselines, psis, stations = (
        [wls[i] for i in percentile_ind],
        [labels[i] for i in percentile_ind],
        [values[i] for i in percentile_ind],
        [errors[i] for i in percentile_ind],
        baselines[percentile_ind],
        psis[percentile_ind],
        stations[percentile_ind],
    )

    rows, cols = get_plot_layout(nplots)
    fig, axarr = plt.subplots(
        rows,
        cols,
        figsize=(cols * 4, rows * 4),
        sharex=True,
        constrained_layout=True,
    )

    ymin, ymax = 0, 0
    for index, (ax, b, psi) in enumerate(zip(axarr.flat, baselines, psis)):
        ymax = np.max(values[index]) if np.max(values[index]) > ymax else ymax
        ymin = np.min(values[index]) if np.min(values[index]) < ymin else ymin
        line = ax.plot(
            wls[index],
            values[index],
            label=rf"{stations[index]}, B={b:.2f} m, $\psi$={psi:.2f}$^\circ$",
        )
        ax.fill_between(
            wls[index],
            values[index] + errors[index],
            values[index] - errors[index],
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
    if observable == "vis":
        y_label = r"$F_{\nu,\,\mathrm{corr.}}$ (Jy)"
        ylims = [0, ymax + ymax * 0.15]
    elif observable == "visphi":
        y_label = r"$\phi_{\mathrm{diff.}}$ ($^\circ$)"
        ylims = [ymin - ymin * 0.25, ymax + ymax * 0.15]
    elif observable == "t3":
        y_label = r"$\phi_{\mathrm{cl.}}$ ($^\circ$)"
        ylims = [ymin - ymin * 0.25, ymax + ymax * 0.15]
    else:
        y_label = "$V^2$ (a.u.)"
        ylims = [0, 1]
    [ax.set_ylim(ylims) for ax in axarr.flat]
    [ax.remove() for index, ax in enumerate(axarr.flat) if index >= nplots]
    fig.subplots_adjust(left=0.2, bottom=0.2)
    plt.savefig(save_dir / f"{observable}_{band}.pdf", format="pdf")
    plt.close()


def plot_uv(ax: Axes, data: Data, index: int | None = None) -> Axes:
    """Plots the uv coverage.

    Parameters
    ----------
    ax :
    data : Data
    index : int, optional

    Returns
    -------
    ax : Axes
    """
    handles = []
    colors = get_colorlist("tab20", len(data))
    # TODO: Reimplement individual uv plots (for a certain index)
    labels, ucoords, vcoords = [], [], []
    for file_idx, (*_, x, y, _) in enumerate(data.vis2):
        color = colors[file_idx]
        labels.extend([f"{data.label[file_idx]}.{i + 1}" for i in np.arange(x.size)])
        ucoords.extend(x)
        vcoords.extend(y)

        ax.plot(
            x,
            y,
            "x",
            markersize=6,
            markeredgewidth=2,
            color=color,
        )
        ax.plot(
            -x,
            -y,
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
                label=data.header[file_idx]["date-obs"].split("T")[0],
            )
        )

    labels, ucoords, vcoords = np.array(labels), np.array(ucoords), np.array(vcoords)
    sorted_idx = np.argsort(np.hypot(ucoords, vcoords))
    for label, ucoord, vcoord in zip(
        labels[sorted_idx], ucoords[sorted_idx], vcoords[sorted_idx]
    ):
        ax.text(
            ucoord + 3.5,
            vcoord + 3.5,
            label,
            fontsize="xx-small",
            color="0",
            alpha=0.8,
        )

    ax.plot([0.0], [0.0], "+k", markersize=5, markeredgewidth=2, alpha=0.5)
    xlabel, ylabel = "$u$ (m) - South", "$v$ (m) - East"
    ax.legend(handles=handles, fontsize="small")

    plt.gca().invert_xaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot(
    fits_files: List[Path] | Path,
    plots: List[str] | str = "all",
    bands: str = "all",
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
    data = filter_data(read_data(fits_files), bands=bands)
    module = sys.modules[__name__]
    if plots == "all":
        plots = ["uv", "flux", "t3", "vis2", "vis", "visphi"]

    if kind == "collage":
        cols, rows = len(plots), len(data)
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
            # FIXME: This only works for one file with the index at -1
            getattr(module, f"plot_{plot}")(ax, data, -1)

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
    # plot(
    #     fits_files,
    #     kind="collage",
    #     plots=["flux", "vis", "visphi", "t3"],
    #     save_dir=plot_dir / "reflagged_data_nband.png",
    # )

    # fits_files = list((path).glob("HD_*.fits"))
    # for fits_file in tqdm(fits_files, desc="Plotting oifits..."):
    #     plot(fits_file, kind="combined", plots=["flux", "vis", "visphi", "t3"], save_dir=plot_dir / f"{fits_file.stem}.png")

    # plot_baselines(list(path.glob("*.fits")), "nband", number=True)
    # plot_baselines(list(path.glob("*.fits")), "nband", observable="visphi", number=True)
