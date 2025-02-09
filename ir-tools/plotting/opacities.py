from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from ppdmod import utils
from ppdmod.data import ReadoutFits


def plot_combined_grains(
    file_dir: Path,
    weights: np.ndarray,
    names: List[str],
    method: str,
    sizes: List[float],
    fmaxs: Optional[List[float]] = None,
    wavelength_range: Optional[List[float]] = [1.6, 14],
    save_dir: Optional[Path] = None,
) -> None:
    """Plot combined opacities."""
    save_dir = Path(save_dir)
    wl, data = utils.get_opacity(file_dir, weights, sizes, names, method, fmaxs=fmaxs)
    indices = np.where((wl > wavelength_range[0]) & (wl < wavelength_range[1]))

    plt.plot(wl[indices], data[indices])
    plt.xlabel(r"$\lambda$ ($\mu$m)")
    plt.ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
    plt.savefig(save_dir, format=save_dir.suffix[1:], dpi=300)
    plt.close()


def plot_individual_grains(
    file_dir: Path,
    continuum_file: Path,
    weights: np.ndarray,
    names: List[str],
    method: str,
    sizes: List[float],
    fmaxs: Optional[List[float]] = None,
    wavelength_range: Optional[List[float]] = [1.6, 14],
    save_dir: Optional[Path] = None,
    cmap: Optional[str] = "tab20",
) -> None:
    """Plot individual opacities."""
    save_dir = Path(save_dir)
    linestyles = ["solid", "dashed"]
    wl_cont, op_cont = map(lambda x: x.value, utils.qval_to_opacity(continuum_file))
    cont_ind = np.where(
        (wl_cont > wavelength_range[0]) & (wl_cont < wavelength_range[1])
    )

    wl, data = utils.get_opacity(
        file_dir, weights, sizes, names, method, fmaxs=fmaxs, individual=True
    )
    indices = np.where((wl > wavelength_range[0]) & (wl < wavelength_range[1]))

    text_kwargs = {"fontsize": 14, "va": "center"}
    _, axarr = plt.subplots(len(names) + 1, 1, figsize=(12, 10), sharex=True)
    # fig.text(0.04, 0.5, r"$\kappa$ ($cm^{2}g^{-1}$)",
    #          rotation="vertical", va="center")

    lower_ind = 0
    for index, (ax, size, label) in enumerate(zip(axarr.flatten(), sizes, names)):
        tmp_data = data[lower_ind : lower_ind + len(size)]
        for i, dat in enumerate(tmp_data):
            ax.plot(
                wl.flatten()[indices],
                dat.flatten()[indices],
                label=rf"{size[i]} $\mu$m",
                ls=linestyles[i],
                c=plt.get_cmap(cmap)(i),
            )
        if index == 2:
            ax.set_ylabel(r"$\kappa$ ($cm^{2}g^{-1}$)")
        ax.text(
            0.5, 0.8, label.title(), ha="center", transform=ax.transAxes, **text_kwargs
        )
        ax.legend()
        lower_ind += len(size)

    axarr[-1].plot(wl_cont[cont_ind], op_cont[cont_ind], label=r"0.1 $\mu$m")
    axarr[-1].set_xlabel(r"$\lambda$ ($\mu$m)")
    axarr[-1].text(
        0.5, 0.8, "Carbon", ha="center", transform=axarr[-1].transAxes, **text_kwargs
    )
    plt.tight_layout()
    plt.savefig(save_dir, format=save_dir.suffix[1:], dpi=300)
    plt.close()


def plot_all_combined_grains(
    qval_paths,
    grf_paths,
    continuum_file,
    weights,
    wavelength_range: Optional[List[float]] = [1, 14],
):
    wavelength, opacity = map(lambda x: x.value, utils.qval_to_opacity(continuum_file))
    indices = np.where(
        (wavelength > wavelength_range[0]) & (wavelength < wavelength_range[1])
    )
    _, ax = plt.subplots()
    plot_combined_grains(
        qval_paths, weights, load_func=utils.qval_to_opacity, ax=ax, label="DHS"
    )
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

    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0]) / 100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]
    fmaxs = [1.0, 1.0, 1.0, None]

    plot_individual_grains(
        path,
        dhs_continuum_file,
        weights,
        names,
        "qval",
        sizes,
        fmaxs,
        save_dir="dust_species.pdf",
    )

    plot_combined_grains(
        path, weights, names, "qval", sizes, fmaxs, save_dir="combined_grains.pdf"
    )
    # plot_all_combined_grains(qval_paths, grf_paths,
    #                          dhs_continuum_file, weights)
