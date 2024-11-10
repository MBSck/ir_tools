import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
from dynesty import DynamicNestedSampler
from matadrs.utils.plot import Plotter
from ppdmod.data import set_data
from ppdmod.fitting import compute_observable_chi_sq, compute_observables
from ppdmod.options import OPTIONS
from ppdmod.plot import (
    get_best_fit,
    plot_chains,
    plot_components,
    plot_component_mosaic,
    plot_corner,
    plot_fit,
    plot_interferometric_observables,
    plot_intermediate_products,
    plot_overview,
    plot_sed,
)

from ..tables import best_fit_parameters


def ptform():
    pass


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = (
        data_dir
        / "model_results"
        / "disc_fits"
        / "2024-11-08"
        / "results_model_14:47:41"
    )
    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    fits_dir = data_dir / "fitting_data" / "hd142527"
    wavelengths = {
        "hband": [1.7] * u.um,
        "kband": [2.15] * u.um,
        "lband": np.linspace(3.1, 3.4, 5) * u.um,
        "mband": np.linspace(4.7, 4.9, 3) * u.um,
        "nband": np.linspace(8, 13, 35) * u.um,
    }

    OPTIONS.model.output = "non-normed"
    fits_files = list((fits_dir).glob("*fits"))
    # plot_kwargs = dict(
    #     legend_format="short",
    #     error=True,
    #     subplots=True,
    #     margin=0.3,
    #     legend_size="medium",
    #     sharex=True,
    #     share_legend=True,
    #     save=True,
    # )

    # lband_data = list(fits_dir.glob("*HAW*"))
    # plot_fits = Plotter(lband_data[:3], plot_name="lband_data.pdf", save_dir=plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(
    #     unwrap=False
    # )
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(
    #     lband_data[4:], plot_name="lband_data_continuation.pdf", save_dir=plot_dir
    # )
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(
    #     unwrap=False
    # )
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(
    #     list(fits_dir.glob("*AQU*")), plot_name="nband_data.pdf", save_dir=plot_dir
    # )
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(
    #     unwrap=True
    # )
    # plot_fits.plot(**plot_kwargs)

    dim = 1024
    wavelength = np.concatenate(
        (
            wavelengths["hband"],
            wavelengths["kband"],
            wavelengths["lband"],
            wavelengths["mband"],
            wavelengths["nband"],
        )
    )
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis"])
    sampler = DynamicNestedSampler.restore(path / "sampler.save")
    results = np.load(path / "results.npz")
    theta, uncertainties = results["theta"], results["uncertainties"]

    with open(path / "components.pkl", "rb") as f:
        components = pickle.load(f)

    component_labels = ["Star", "Inner Ring", "Outer Ring", "Last Ring"][
        : len(components)
    ]

    # TODO: Check why the chi_sq is different here from the value that it should have
    ndim = np.array(theta).size
    rchi_sq = compute_observable_chi_sq(
        *compute_observables(components),
        ndim=ndim,
        reduced=True,
        rtotal_chi_sq=True,
    )
    print(f"Total reduced chi_sq: {rchi_sq:.2f}")

    rchi_sqs = compute_observable_chi_sq(
        *compute_observables(components), ndim=ndim, reduced=True
    )
    rchi_sqs = np.round(rchi_sqs, 2)
    print(f"Individual reduced chi_sqs: {rchi_sqs}")

    labels = np.load(path / "labels.npy")
    units = np.load(path / "units.npy", allow_pickle=True)
    plot_corner(sampler, labels, units, savefig=plot_dir / "corner.pdf")
    # plot_chains(sampler, labels, units, savefig=plot_dir / "chains.pdf")

    plot_overview(savefig=plot_dir / "overview.pdf")
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=True,
        savefig=assets_dir / "disc.csv",
    )
    plot_fit(components=components, savefig=plot_dir / "disc.pdf")
    plot_components(
        components, dim, 0.1, 10, norm=0.2, zoom=8, savefig=plot_dir / "components.pdf"
    )
    plot_component_mosaic(
        components, dim, 0.1, norm=0.2, savefig=plot_dir / "models.pdf", zoom=8
    )
    plot_intermediate_products(
        dim, wavelength, components, component_labels, save_dir=plot_dir
    )
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)
