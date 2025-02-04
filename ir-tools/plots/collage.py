import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.axes import Axes
from tqdm import tqdm


@dataclass
class Data:
    header: List[fits.Header] = field(default_factory=list)
    wl: List[fits.BinTableHDU] = field(default_factory=list)
    array: List[fits.BinTableHDU] = field(default_factory=list)
    vis: List[fits.BinTableHDU] = field(default_factory=list)
    vis2: List[fits.BinTableHDU] = field(default_factory=list)
    t3: List[fits.BinTableHDU] = field(default_factory=list)


def read_data(fits_files: List[Path] | Path) -> Data:
    """Reads the data from the fits files."""
    data = Data()
    for fits_file in fits_files if isinstance(fits_files, list) else [fits_files]:
        index = 20 if "GRAV" in fits_file.stem else None
        with fits.open(fits_file) as hdul:
            for key in ["header", "wl", "array", "vis", "vis2", "t3"]:
                if key == "header":
                    data.header.append(hdul[0].header)
                elif key == "wl":
                    data.wl.append(
                        (hdul["oi_wavelength", index].data["eff_wave"] * u.m).to(u.um)
                    )
                else:
                    getattr(data, key).append(hdul[f"oi_{key}", index])
    return data


def get_unit(self, header: str, sub_header: str) -> str:
    """Fetches the unit of a header by the sub header's name."""
    unit = getattr(self, header)[sub_header.upper()].unit
    return str(unit) if unit is not None else "a.u."


def plot_uv(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the uv coverage."""
    ucoords, vcoords = data.vis2.data["ucoord"], data.vis2.data["vcoord"]
    # if color_by == "file":
    #     color = colors[index]
    #     handles.append(mlines.Line2D(
    #         [], [], color=color, marker="X",
    #         linestyle="None", label=readout.date[:-8]))
    # elif color_by == "instrument":
    #     if readout.instrument not in instruments:
    #         instruments.append(readout.instrument)
    #     color = colors[instruments.index(readout.instrument)]

    for uv_index, (ucoord, vcoord) in enumerate(zip(ucoords, vcoords)):
        ax.plot(ucoord, vcoord, "x", color=color, markersize=10, markeredgewidth=3)
        ax.plot(-ucoord, -vcoord, "x", color=color, markersize=10, markeredgewidth=3)

        # if show_text:
        #     ax.text(
        #         -ucoord - 3.5,
        #         -vcoord - 1.5,
        #         uv_index
        #         fontsize="small",
        #         color="0",
        #         alpha=0.8,
        #     )

    ax.plot([0.0], [0.0], "+k", markersize=5, markeredgewidth=2, alpha=0.5)
    # TODO: Implement check or calculation for the orientations
    xlabel, ylabel = "$u$ (m) - South", "$v$ (m) - East"
    # uv_extent = int(uv_max + uv_max*0.25) if uv_extent is None else uv_extent

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


def plot_flux(ax: Axes, hdul, x: np.ndarray, index: int) -> Axes:
    """Plots the flux."""
    card = hdul["oi_flux", index]
    key = "fluxdata" if "FLUXDATA" in card.columns.names else "flux"
    ax = plot_data(ax, x, card.data[key], card.data["fluxerr"], card.data["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_vis(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the visibility or the correlated flux."""
    x, dataset = data.wl[index], data.vis[index].data
    plot_data(ax, x, dataset["visamp"], dataset["visamperr"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$F_{\nu,\,\mathrm{corr}}$ $\left(\mathrm{Jy}\right)$")
    return ax


def plot_visphi(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the differential phases."""
    x, dataset = data.wl[index], data.vis[index].data
    plot_data(ax, x, dataset["visphi"], dataset["visphierr"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$\phi_{\mathrm{diff}}$ $\left(^\circ\right)$")
    return ax


def plot_vis2(ax: Axes, data: Data, index: int) -> Axes:
    """Plots the squared visibility."""
    x, dataset = data.wl[index], data.vis2[index].data
    plot_data(ax, x, dataset["vis2data"], dataset["vis2err"], dataset["flag"])
    ax.set_xlabel(r"$\lambda$ $\left(\mathrm{\mu}m\right)$")
    ax.set_ylabel(r"$V^{2}$ (a.u.)")
    ax.set_ylim((0, None) if ax.get_ylim()[1] < 1 else (0, 1))
    return ax


def plot_t3(ax: Axes, data: Data, index: int) -> Axes:
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

    cols, rows = len(plots), int(np.ceil(len(data.vis) / len(plots)))
    figsize = (cols * cell_width, rows * cell_width)
    _, axarr = plt.subplots(
        rows, cols, figsize=figsize, sharex=True, constrained_layout=True
    )

    if kind == "collage":
        for row_index, ax_row in enumerate(axarr):
            for plot, ax_col in zip(plots, ax_row):
                getattr(module, f"plot_{plot}")(ax_col, data, row_index)
    elif kind == "combined":
        ...
    # TODO: Implement the individual plots again
    else:
        ...

    # TODO: Reimplement different plots from this (as in saved in different files)
    if save_dir is not None:
        plt.savefig(save_dir, format="png", dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    path = Path().home() / "Data" / "fitting" / "hd142527"
    plot_dir = path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot(
        list(path.glob("*_N_*.fits")),
        # ["flux", "t3", "vis2", "vis"],
        ["uv"],
        save_dir=plot_dir,
    )
