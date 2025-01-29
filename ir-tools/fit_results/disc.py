import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
from dynesty import DynamicNestedSampler
from ppdmod.data import set_data
from ppdmod.fitting import (
    compute_interferometric_chi_sq,
    get_labels,
    get_theta,
    get_units,
)
from ppdmod.options import OPTIONS
from ppdmod.plot import (
    plot_chains,
    plot_component_mosaic,
    plot_components,
    plot_corner,
    plot_fit,
    plot_interferometric_observables,
    plot_intermediate_products,
    plot_overview,
)
from ppdmod.utils import (
    create_adaptive_bins,
    windowed_linspace,
)

from ..tables import best_fit_parameters

np.seterr(over="ignore", divide="ignore")


def ptform():
    pass


# TODO: Fix the chi square here to get correct fit values
if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "results" / "disc" / "2025-01-28"
    path /= "better_constraints_lnf"
    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    fits_dir = data_dir / "fitting" / "hd142527"
    nband_wavelengths, nband_binning_windows = create_adaptive_bins(
        [8, 13], [9.2, 11.9], 0.2, 0.65
    )
    wavelengths = {
        "hband": [1.7] * u.um,
        "kband": [2.15] * u.um,
        "lband": windowed_linspace(3.1, 3.4, OPTIONS.data.binning.lband.value) * u.um,
        "mband": windowed_linspace(4.7, 4.9, OPTIONS.data.binning.mband.value) * u.um,
        "nband": nband_wavelengths * u.um,
    }
    OPTIONS.data.binning.nband = nband_binning_windows * u.um
    fits_files = list((fits_dir).glob("*fits"))

    dim = 1024
    bands = ["hband", "kband", "lband", "mband", "nband"]
    wavelengths = np.concatenate([wavelengths[band] for band in bands])
    fit_data = ["flux", "vis", "t3"]
    data = set_data(
        fits_files, wavelengths=wavelengths, fit_data=fit_data,
    )
    uncertainties = np.load(path / "uncertainties.npy")
    with open(path / "components.pkl", "rb") as f:
        components = pickle.load(f)

    # breakpoint()
    theta = get_theta(components)
    component_labels = [component.label for component in components]

    # TODO: Check why the chi_sq is different here from the value that it should be?
    rchi_sqs = compute_interferometric_chi_sq(
        components,
        ndim=theta.size,
        method="linear",
        reduced=True,
    )
    print(f"Total reduced chi sq: {rchi_sqs[0]:.2f}")
    print(f"Individual reduced chi_sqs: {np.round(rchi_sqs[1:], 2)}")

    labels = get_labels(components)
    units = get_units(components)
    sampler = DynamicNestedSampler.restore(path / "sampler.save")

    plot_format = "pdf"
    plot_corner(sampler, labels, units, savefig=(plot_dir / f"corner.{plot_format}"))
    plot_overview(savefig=(plot_dir / f"overview.{plot_format}"))
    plot_overview(
        bands=["nband"],
        savefig=(plot_dir / f"overview_nband.{plot_format}"),
    )
    plot_overview(
        bands=["hband", "kband", "lband", "mband"],
        savefig=(plot_dir / f"overview_hlkmband.{plot_format}"),
    )
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=True,
        savefig=assets_dir / "disc.csv",
    )
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=False,
        savefig=assets_dir / "disc",
    )
    plot_fit(components=components, savefig=(plot_dir / f"disc.{plot_format}"))
    plot_fit(
        components=components,
        bands=["nband"],
        savefig=(plot_dir / f"disc_nband.{plot_format}"),
    )
    plot_fit(
        components=components,
        bands=["hband", "kband", "lband", "mband"],
        ylims={"t3": [-15, 15]},
        savefig=(plot_dir / f"disc_hklmband.{plot_format}"),
    )
    zoom = 5
    plot_components(
        components,
        dim,
        0.1,
        3.5,
        norm=0.3,
        zoom=zoom,
        savefig=plot_dir / "image_lband.png",
    )

    OPTIONS.data.binning.nband = (
        np.interp(10.5, nband_wavelengths, nband_binning_windows) * u.um
    )
    plot_components(
        components,
        dim,
        0.1,
        10.5,
        norm=0.3,
        zoom=zoom,
        savefig=plot_dir / "image_nband.png",
    )
    # plot_component_mosaic(
    #     components, dim, 0.1, norm=0.2, savefig=plot_dir / "models.pdf", zoom=8
    # )
    # plot_intermediate_products(
    #     dim, wavelengths, components, component_labels, save_dir=plot_dir
    # )
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)
