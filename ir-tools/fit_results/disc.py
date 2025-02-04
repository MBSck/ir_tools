import pickle
from pathlib import Path

import astropy.units as u
import emcee
import numpy as np
from dynesty import DynamicNestedSampler
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    get_labels,
    get_units,
)
from ppdmod.options import OPTIONS
from ppdmod.plot import (
    plot_baselines,
    plot_components,
    plot_corner,
    plot_fit,
    plot_intermediate_products,
    plot_overview,
)
from ppdmod.utils import (
    # create_adaptive_bins,
    windowed_linspace,
)

from ..tables import best_fit_parameters

np.seterr(over="ignore", divide="ignore")


def ptform():
    pass


# TODO: Fix the chi square here to get correct fit values
if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "results" / "disc" / "2025-01-30"
    path /= "different_asym"

    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    fits_dir = data_dir / "fitting" / "hd142527"
    # nband_wavelengths, nband_binning_windows = create_adaptive_bins(
    #     [8.6, 12.3], [9.2, 11.9], 0.2, 0.65
    # )
    wavelengths = {
        "hband": [1.7] * u.um,
        "kband": [2.15] * u.um,
        "lband": windowed_linspace(3.1, 3.8, OPTIONS.data.binning.lband.value) * u.um,
        "mband": windowed_linspace(4.65, 4.9, OPTIONS.data.binning.mband.value) * u.um,
        "nband": windowed_linspace(8.25, 12.75, OPTIONS.data.binning.nband.value) * u.um,
    }
    # OPTIONS.data.binning.nband = nband_binning_windows * u.um
    fits_files = list((fits_dir).glob("*fits"))

    OPTIONS.fit.fitter = "dynesty"
    OPTIONS.fit.condition = "sequential_radii"

    dim = 1024
    bands = ["hband", "kband", "lband", "mband", "nband"]
    wavelengths = np.concatenate([wavelengths[band] for band in bands])
    fit_data = ["flux", "vis", "t3"]
    data = set_data(
        fits_files, wavelengths=wavelengths, fit_data=fit_data,
    )
    if OPTIONS.fit.fitter == "emcee":
        sampler = emcee.backends.HDFBackend(path / "sampler.h5")
    else:
        sampler = DynamicNestedSampler.restore(path / "sampler.save")

    theta = np.load(path / "theta.npy")
    uncertainties = np.load(path / "uncertainties.npy")
    with open(path / "components.pkl", "rb") as f:
        components = OPTIONS.model.components = pickle.load(f)

    components[-1].rout.value = 2.1
    components[-1].rin.value = 1.4
    components[-1].sigma0.value = 1e-3

    labels, units = get_labels(components), get_units(components)
    OPTIONS.fit.condition_indices = list(
        map(labels.index, (filter(lambda x: "rin" in x or "rout" in x, labels)))
    )
    component_labels = [component.label for component in components]

    # TODO: Check why the chi_sq is different here from the value that it should be?
    rchi_sqs = compute_interferometric_chi_sq(
        components,
        theta.size,
        method="linear",
        reduced=True,
    )
    print(f"Total reduced chi sq: {rchi_sqs[0]:.2f}")
    print(f"Individual reduced chi_sqs: {np.round(rchi_sqs[1:], 2)}")

    plot_format = "pdf"
    # plot_corner(sampler, labels, units, savefig=(plot_dir / f"corner.{plot_format}"), discard=1000)
    # plot_overview(savefig=(plot_dir / f"overview.{plot_format}"))
    # plot_overview(
    #     bands=["nband"],
    #     savefig=(plot_dir / f"overview_nband.{plot_format}"),
    # )
    # plot_overview(
    #     bands=["hband", "kband", "lband", "mband"],
    #     savefig=(plot_dir / f"overview_hlkmband.{plot_format}"),
    # )
    plot_fit(components=components, savefig=(plot_dir / f"disc.{plot_format}"))
    plot_fit(
        components=components,
        bands=["nband"],
        savefig=(plot_dir / f"disc_nband.{plot_format}"),
    )
    # plot_fit(
    #     components=components,
    #     bands=["hband", "kband", "lband", "mband"],
    #     ylims={"t3": [-15, 15]},
    #     savefig=(plot_dir / f"disc_hklmband.{plot_format}"),
    # )
    # zoom = 5
    # plot_components(
    #     components,
    #     dim,
    #     0.1,
    #     3.5,
    #     norm=0.3,
    #     zoom=zoom,
    #     savefig=plot_dir / "image_lband.png",
    # )
    #
    # OPTIONS.data.binning.nband = (
    #     np.interp(10.5, nband_wavelengths, nband_binning_windows) * u.um
    # )
    # plot_components(
    #     components,
    #     dim,
    #     0.1,
    #     10.5,
    #     norm=0.3,
    #     zoom=zoom,
    #     savefig=plot_dir / "image_nband.png",
    # )
    # # plot_intermediate_products(
    # #     dim, wavelengths, components, component_labels, save_dir=plot_dir
    # # )
    # best_fit_parameters(
    #     labels,
    #     units,
    #     theta,
    #     uncertainties,
    #     save_as_csv=True,
    #     savefig=assets_dir / "disc.csv",
    #     fit_method=OPTIONS.fit.fitter,
    # )
    # best_fit_parameters(
    #     labels,
    #     units,
    #     theta,
    #     uncertainties,
    #     save_as_csv=False,
    #     savefig=assets_dir / "disc",
    #     fit_method=OPTIONS.fit.fitter,
    # )

    number = True
    for nplots, band in zip([20, 20, 20, 12], ["hband", "kband", "lband", "nband"]):
        plot_baselines(
            wavelengths,
            components,
            band,
            plot_dir,
            data_type="vis",
            nplots=nplots,
            number=number,
        )
        plot_baselines(
            wavelengths,
            components,
            band,
            plot_dir,
            data_type="t3",
            nplots=nplots,
            number=False,
        )
