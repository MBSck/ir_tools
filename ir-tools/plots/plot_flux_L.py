from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matadrs.utils.readout import ReadoutFits


def map_condition(wl, value, error=None):
    indices = np.where((wl >= 3.) & (wl < 5.))
    args = [wl, value] if error is None else [wl, value, error]
    return list(map(lambda x: x[indices].squeeze(), args))


def get_err_fill(value, error):
    return (value-error, value+error)


def plot_w_err_fill(ax, data, label, color=None):
    if isinstance(data, list):
        values = data
    elif isinstance(data, Path):
        values = map_condition(*np.loadtxt(data, unpack=True))
    ax.plot(*values[:2], color=color, label=label)
    if len(values) == 3:
        ax.fill_between(values[0], *get_err_fill(*values[1:]),
                        alpha=0.2, color=color)


if __name__ == "__main__":
    data_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
    files = list(data_path.glob("*HAWAII*"))

    flux_data = Path("/Users/scheuck/Data/flux_data/hd142666")
    spitzer = flux_data / "HD_142666_sws.txt"
    iso = flux_data / "10402847.ss"

    readout_april_21 = ReadoutFits(files[0])
    readout_april_23 = ReadoutFits(files[1])
    print((readout_april_21.oi_flux["FLUXERR"].data.squeeze() /
           readout_april_21.oi_flux["FLUXDATA"].data.squeeze()).mean())
    print((readout_april_23.oi_flux["FLUXERR"].data.squeeze() /
           readout_april_23.oi_flux["FLUXDATA"].data.squeeze()).mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_w_err_fill(ax, [readout_april_21.oi_wl["EFF_WAVE"].data.squeeze(),
                         readout_april_21.oi_flux["FLUXDATA"].data.squeeze()],
                    "MATISSE/2022-4-21", "green")
    ax.errorbar(readout_april_21.oi_wl["EFF_WAVE"].data.squeeze()[25],
                readout_april_21.oi_flux["FLUXDATA"].data.squeeze()[25],
                yerr=readout_april_21.oi_flux["FLUXERR"].data.squeeze()[25],
                color="green", capsize=3)
    plot_w_err_fill(ax, [readout_april_23.oi_wl["EFF_WAVE"].data.squeeze(),
                         readout_april_23.oi_flux["FLUXDATA"].data.squeeze()],
                    "MATISSE/2022-4-23", "blue")
    ax.errorbar(readout_april_23.oi_wl["EFF_WAVE"].data.squeeze()[35],
                readout_april_23.oi_flux["FLUXDATA"].data.squeeze()[35],
                yerr=readout_april_23.oi_flux["FLUXERR"].data.squeeze()[35],
                color="blue", capsize=3)
    ax.errorbar(readout_april_23.oi_wl["EFF_WAVE"].data.squeeze()[10],
                readout_april_23.oi_flux["FLUXERR"].data.squeeze()[10])
    plot_w_err_fill(ax, spitzer, "SPITZER", "red")
    plot_w_err_fill(ax, iso, "ISO/1996", "orange")
    ax.set_xlim([3, 5])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Total Flux (Jy)")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    plt.legend(loc="best", fontsize="small")
    # plt.show()
    plt.savefig("flux_plot_L.pdf", format="pdf")
