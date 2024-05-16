from pathlib import Path
from typing import Optional, List

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table

# import oimodeler as oim
from fluxcal import transform_spectrum_to_real_spectral_resolution


def make_overview_plot(grains, f_maxs, gs_sizes, qval_files):
    qval_folder = Path("/Users/scheuck/Data/opacities/QVAL")
    fig, ax = plt.subplots(int(np.ceil(len(grains)/2)), 2, figsize=(10, 10))
    axarr = ax.flatten()
    for index, (grain, f_max, gs_size, qval_file)\
            in enumerate(zip(grains, f_maxs, gs_sizes, qval_files)):
        plot_title = f"{grain}_a{gs_size}_dhs_fmax{f_max}"
        qval_file = qval_folder / qval_file
        qval_wl, qval = np.loadtxt(qval_file, unpack=True, skiprows=1)
        particle = oim.oimOptoolBackend(grains=grain, f_max=f_max,
                                        wavelength_solution=qval_wl,
                                        gs_sizes=gs_size, wgrid=True,
                                        cache_dir="/Users/scheuck/Data/opacities/optool")
        q = 4/3*(particle.kabs[0]*particle.rho*particle.a1*(u.um.to(u.cm)))
        indices_lam = np.where(np.logical_and(particle.lam >= 7, particle.lam <= 15))
        axarr[index].plot(qval_wl[indices_lam], q[indices_lam], label="optool")
        axarr[index].plot(qval_wl[indices_lam], qval[indices_lam], label="SpecFit")
        axarr[index].set_title(plot_title)
        axarr[index].legend()

    fig.suptitle("Overview: Optical depths")
    penultimate_plot = axarr.shape[0]-2
    axarr[penultimate_plot].set_xlabel(r"Wavelength ($\mu$m)")
    axarr[penultimate_plot].set_ylabel("Absorption efficiency Q")
    plt.savefig("overview_opacities.pdf", format="pdf")


def calculate_and_plot(grains, author, f_max, gs_size, qval_file):
    plot_title = f"{grains}_{author}_a{gs_size}_dhs_fmax{f_max}_nwl800"
    path = Path("/Users/scheuck/Data/reduced_data/jozsef_reductions/")
    file = path / "hd142666/HD_142666_2019-03-24T09_01_46_N_TARGET_FINALCAL_INT.fits"
    file_high = path / "HD_144432_2020-03-13T08_41_26_N_TARGET_FINALCAL_INT.fits"

    qval_folder = Path("/Users/scheuck/Data/opacities/QVAL")
    qval_file = qval_folder / qval_file
    qval_wl, qval = np.loadtxt(qval_file, unpack=True, skiprows=1)
    indices = np.where(np.logical_and(qval_wl >= 7, qval_wl <= 15))
    qval_wl, qval = map(lambda x: x[indices], [qval_wl, qval])

    low_dl_coeffs = [0.10600484,  0.01502548,  0.00294806, -0.00021434]
    high_dl_coeffs = [-8.02282965e-05,  3.83260266e-03,
                      7.60090459e-05, -4.30753848e-07]

    with fits.open(file) as hdul:
        wl = Table.read(hdul["OI_WAVELENGTH"])["EFF_WAVE"].data*1e6

    with fits.open(file_high) as hdul:
        wl_high = Table.read(hdul["OI_WAVELENGTH"])["EFF_WAVE"].data*1e6

    fig, axarr = plt.subplots(2, 2, figsize=(10, 7))
    ax, bx = axarr[0].flatten()
    cx, dx = axarr[1].flatten()
    fig.suptitle(plot_title)

    particle = oim.oimOptoolBackend(grains=grains, f_max=f_max,
                                    wl_min=7, wl_max=15, wl_num=800,
                                    porosity=0.7,
                                    gs_sizes=gs_size, wgrid=True,
                                    cache_dir="/Users/scheuck/Data/opacities/optool",
                                    storage_dir=plot_title)

    indices_lam = np.where(np.logical_and(particle.lam >= 7, particle.lam <= 15))
    ax.plot(particle.lam[indices_lam], q[indices_lam])
    ax.set_title("Optool opacities")
    opacity_low = transform_spectrum_to_real_spectral_resolution(particle.lam,
                                                                 particle.kabs[0],
                                                                 low_dl_coeffs,
                                                                 10, wl, 7)
    opacity_high = transform_spectrum_to_real_spectral_resolution(particle.lam,
                                                                  particle.kabs[0],
                                                                  high_dl_coeffs,
                                                                  10, wl_high, 7)

    q_interp_low = 4/3*(opacity_low*particle.rho*particle.a1*(u.um.to(u.cm)))
    q_interp_high = 4/3*(opacity_high*particle.rho*particle.a1*(u.um.to(u.cm)))
    bx.plot(wl, q_interp_low)
    bx.plot(wl_high, q_interp_high)
    bx.legend(["LOW", "HIGH"])
    bx.set_title("Optool MATISSE interpolated opacities")

    cx.plot(qval_wl, qval)
    cx.set_xlabel(r"Wavelength ($\mu$m)")
    cx.set_ylabel("Absorption efficiency Q")
    cx.set_title("Min+Rens opacities")

    qval_low = transform_spectrum_to_real_spectral_resolution(qval_wl, qval,
                                                              low_dl_coeffs,
                                                              10, wl, 7)
    qval_high = transform_spectrum_to_real_spectral_resolution(qval_wl, qval,
                                                               high_dl_coeffs,
                                                               10, wl_high, 7)
    dx.plot(wl, qval_low)
    dx.plot(wl_high, qval_high)
    dx.legend(["LOW", "HIGH"])
    dx.set_title("Min+Rens MATISSE interpolated opacities")
    plt.savefig(particle.storage_dir / f"{plot_title}.png", format="png", dpi=300)
    plt.close()


def qval_to_opacity(qval_file: Path) -> u.cm**2/u.g:
    """Reads the qval file and returns the opacity.

    Parameters
    ----------
    qval_file : pathlib.Path

    Notes
    -----
    The qval-files give the grain size in microns and the
    density in g/cm^3.
    """
    with open(qval_file, "r+", encoding="utf8") as file:
        _, grain_size, density = map(float, file.readline().strip().split())
    wavelength_grid, qval = np.loadtxt(qval_file, skiprows=1, unpack=True)
    return wavelength_grid*u.um,\
        3*qval/(4*(grain_size*u.um).to(u.cm)*(density*u.g/u.cm**3))


def plot_used_opacities(directory: Path,
                        opacity_files: List[Path],
                        labels: List[str],
                        linestyles: List[str],
                        band: Optional[str] = "nband",
                        filetype: Optional[str] = "qval",
                        wavelength_span: Optional[List[float]] = None):
    """Plots the used opacities.

    Can either take an optool/GRF or an Qval file.

    Parameters
    ----------
    band : str, optional
        The band to plot. The default is "nband".
    wavelength_span : list, optional
        The wavelength span to plot. The default is [7.9, 13.9] for the nband.
    """
    if wavelength_span is None:
        wavelength_span = [7.9, 13.9] if band == "nband" else [3.5, 4.5]

    fig, axarr = plt.subplots(len(opacity_files), 1, figsize=(7, 7))
    axarr = axarr.flatten()
    for ax_index, (name, files) in enumerate(opacity_files.items()):
        for file_index, (file, used) in enumerate(files):
            color = "k" if not used else "r"
            if filetype in ["optool", "grf"]:
                skiprows = 2 if filetype == "optool" else 0
                wavelength_grid, opacity, _, _ = np.loadtxt(
                        directory / file, unpack=True,
                        comments="#", skiprows=skiprows)
                wavelength_grid *= u.um
                opacity *= u.cm**2/u.g
            else:
                wavelength_grid, opacity = qval_to_opacity(directory / file)
            indices = np.where(
                np.logical_and(wavelength_grid > wavelength_span[0]*u.um,
                               wavelength_grid < wavelength_span[1]*u.um))[0]
            axarr[ax_index].plot(wavelength_grid[indices], opacity[indices],
                                 c=color, label=labels[file_index],
                                 linestyle=linestyles[file_index])

            axarr[ax_index].set_ylim(bottom=0)
            if file_index == 1:
                axarr[ax_index].text(0.85, 0.85, name, size=15,
                                     ha="center", va="center",
                                     transform=axarr[ax_index].transAxes)
                if ax_index == 0:
                    axarr[ax_index].legend(loc="upper left")
            if ax_index == (len(qval_files)-1):
                axarr[ax_index].set_xlabel(r"$\lambda$ ($\mu$m)")
            else:
                axarr[ax_index].tick_params(axis="x", which="both", labelbottom=False)
    fig.text(0.02, 0.5, r"$\kappa_{abs}$ (cm$^2$g$^{-1}$)", va="center", rotation="vertical")
    plt.subplots_adjust(
        left=0.12, right=0.98, wspace=0.3, bottom=0.07, top=0.98)
    plt.savefig("used_opacities.pdf", format="pdf")


if __name__ == "__main__":
    # qval_files = ["Q_Am_Mgolivine_Jae_DHS_f0.7_rv2.0.dat",
    #               "Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv0.1.dat",
    #               "Q_Am_Mgpyroxene_Dor_DHS_f0.7_rv2.0.dat",
    #               "Q_Fo_Sogawa_DHS_f1.0_rv0.1.dat",
    #               "Q_En_Jaeger_DHS_f1.0_rv5.0.dat"]
    # grains = ["ol", "pyr", "pyr", "for", "ens"]
    # authors = ["dorschner", "dorschner", "dorschner", "suto", "jaeger"]
    # f_maxs = [0.7, 0.7, 0.7, 0.9999, 0.9999]
    # gs_sizes = [2.0, 0.1, 2.0, 0.1, 5.0]
    #
    # make_overview_plot(grains, f_maxs, gs_sizes, qval_files)

    # for grain, author, f_max, gs_size, qval_file\
    #         in zip(grains, authors, f_maxs, gs_sizes, qval_files):
    #     calculate_and_plot(grain, author, f_max, gs_size, qval_file)
    qval_dir = Path("/Users/scheuck/Data/opacities/QVAL")
    qval_files = {"Olivine": [["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat", True],
                              ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat", True]],
                  "Pyroxene": [["Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv0.1.dat", False],
                               ["Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat", True]],
                  "Forsterite": [["Q_Fo_Suto_DHS_f1.0_rv0.1.dat", True],
                                 ["Q_Fo_Suto_DHS_f1.0_rv1.5.dat", True]],
                  "Enstatite": [["Q_En_Jaeger_DHS_f1.0_rv0.1.dat", False],
                                ["Q_En_Jaeger_DHS_f1.0_rv1.5.dat", True]],
                  "Carbon": [["Q_amorph_c_rv0.1.dat", True],
                             ["Q_amorph_c_rv1.5.dat", False]]}

    grf_dir = Path("/Users/scheuck/Data/opacities/GRF")
    grf_files = {"Olivine": [["MgOlivine0.1.Combined.Kappa", True],
                             ["MgOlivine2.0.Combined.Kappa", True]],
                 "Pyroxene": [["MgPyroxene0.1.Combined.Kappa", False],
                              ["MgPyroxene2.0.Combined.Kappa", True]],
                 "Forsterite": [["Forsterite0.1.Combined.Kappa", True],
                                ["Forsterite2.0.Combined.Kappa", True]],
                 "Enstatite": [["Enstatite0.1.Combined.Kappa", False],
                               ["Enstatite2.0.Combined.Kappa", True]],
                 "Carbon": [["SiC0.1.Combined.Kappa", True],
                            ["SiC2.0.Combined.Kappa", False]]}
    labels = [r"0.1 $ \mu $m", r"1.5 $ \mu $m"]
    linestyles = ["-", "--"]
    plot_used_opacities(qval_dir, qval_files, filetype="qval",
                        labels=labels, linestyles=linestyles)
