from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from ppdmod import utils
from ppdmod.data import ReadoutFits


def plot_combined_grains(file_dir: Path,
                         weights: np.ndarray,
                         names: List[str], method: str,
                         sizes: List[float],
                         fmaxs: Optional[List[float]] = None,
                         wavelength_range: Optional[List[float]] = [1.6, 14],
                         save_dir: Optional[Path] = None) -> None:
    save_dir = Path(save_dir)
    wls, data = utils.get_opacity(file_dir, weights, sizes, names, method, fmaxs=fmaxs)
    indices = np.where((wls[0] > wavelength_range[0])
                       & (wls[0] < wavelength_range[1]))

    plt.plot(wls[0][indices], data[indices])
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    plt.savefig(save_dir, format=save_dir.suffix[1:], dpi=300)
    plt.close()


def plot_individual_grains(file_dir: Path, continuum_file: Path,
                           weights: np.ndarray,
                           names: List[str], method: str,
                           sizes: List[float],
                           fmaxs: Optional[List[float]] = None,
                           wavelength_range: Optional[List[float]] = [1.6, 14],
                           save_dir: Optional[Path] = None) -> None:
    """Plot individual opacities."""
    save_dir = Path(save_dir)
    linestyles = ["solid", "dashed"]
    wl_cont, op_cont = map(
            lambda x: x.value, utils.qval_to_opacity(continuum_file))
    cont_ind = np.where((wl_cont > wavelength_range[0])
                         & (wl_cont < wavelength_range[1]))

    wls, data = utils.get_opacity(file_dir, weights, sizes, names,
                                  method, fmaxs=fmaxs, individual=True)
    indices = np.where((wls[0] > wavelength_range[0])
                       & (wls[0] < wavelength_range[1]))

    _, axarr = plt.subplots(len(names)+1, 1,
                            figsize=(12, 10), sharex=True)
    lower_ind = 0
    for ax, size, label in zip(axarr.flatten(), sizes, names):
        tmp_wls = wls[lower_ind:lower_ind+len(size)]
        tmp_data = data[lower_ind:lower_ind+len(size)]
        for i, dat in enumerate(tmp_data):
            ax.plot(tmp_wls.flatten()[indices], dat.flatten()[indices],
                    label=rf"{size[i]} $\mu$m", ls=linestyles[i])
            ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
            ax.text(0.5, 0.8, label.title(), ha="center",
                    va="center", transform=ax.transAxes)
            ax.legend()
        lower_ind += len(size)

    axarr[-1].plot(wl_cont[cont_ind], op_cont[cont_ind], label=r"0.1 $\mu$m")
    axarr[-1].set_xlabel(r"$\lambda$ ($\mu$m)")
    axarr[-1].set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    axarr[-1].text(0.5, 0.8, "Carbon", ha="center",
                   va="center", transform=axarr[-1].transAxes)
    axarr[-1].legend()
    plt.savefig(save_dir, format=save_dir.suffix[1:], dpi=300)
    plt.close()


def plot_all_combined_grains(qval_paths, grf_paths,
                             continuum_file, weights,
                             wavelength_range: Optional[List[float]] = [1, 14]):
    wavelength, opacity = map(
            lambda x: x.value, utils.qval_to_opacity(continuum_file))
    indices = np.where((wavelength > wavelength_range[0])
                       & (wavelength < wavelength_range[1]))
    _, ax = plt.subplots()
    plot_combined_grains(qval_paths, weights,
                         load_func=utils.qval_to_opacity,
                         ax=ax, label="DHS")
    plot_combined_grains(grf_paths, weights, ax=ax, label="grf")
    ax.plot(wavelength[indices], opacity[indices], label="Continuum (DHS)")
    ax.set_xlabel(r"$\lambda$ ($\mu$m)")
    ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    plt.legend()
    plt.savefig("combined_grains.pdf", format="pdf")


if __name__ == "__main__":
    path = Path("/Users/scheuck/Code/modelling/ppdmod/tests/data")
    readouts = list(map(ReadoutFits, (path / "fits").glob("*fits")))
    # wavelength_axes = list(map(lambda x: x.wavelength, readouts))
    # wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

    dhs_continuum_file = path / "qval" / "Q_amorph_c_rv0.1.dat"
    # dhs_continuum_file = path / "qval" / "Q_iron_0.10um_dhs_0.99.dat"

    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0])/100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]
    fmaxs = [1.0, 1.0, 1.0, 0.7]

    plot_individual_grains(path, dhs_continuum_file,
                           weights, names, "qval", sizes, fmaxs,
                           save_dir="individual_grains.png")

    plot_combined_grains(path, weights, names, "qval", sizes, fmaxs,
                         save_dir="combined_grains.png")
    # plot_all_combined_grains(qval_paths, grf_paths,
    #                          dhs_continuum_file, weights)
