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
    plot_components,
    plot_corner,
    plot_fit,
    plot_intermediate_products,
    plot_overview,
)
from ppdmod.utils import (
    windowed_linspace,
)

from ..tables import best_fit_parameters
from . import oifits

np.seterr(over="ignore", divide="ignore")


def ptform():
    pass


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "results" / "disc" / "2025-02-13"
    path /= "matisse_and_2nband_temp_power_xy_free"

    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    fits_dir = data_dir / "fitting" / "hd142527"
    fits_files = list(map(lambda x: fits_dir / x, np.load(path / "files.npy")))

    OPTIONS.fit.fitter = "dynesty"
    OPTIONS.fit.condition = "sequential_radii"

    dim = 1024
    wavelengths = np.load(path / "wl.npy") * u.um
    data = set_data(
        fits_files,
        wavelengths=wavelengths,
        fit_data=np.load(path / "observables.npy"),
    )
    if OPTIONS.fit.fitter == "emcee":
        sampler = emcee.backends.HDFBackend(path / "sampler.h5")
    else:
        sampler = DynamicNestedSampler.restore(path / "sampler.save")

    theta = np.load(path / "theta.npy")
    uncertainties = np.load(path / "uncertainties.npy")
    with open(path / "components.pkl", "rb") as f:
        components = OPTIONS.model.components = pickle.load(f)

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

    plot_format = "png"
    plot_corner(
        sampler,
        labels,
        units,
        savefig=(plot_dir / f"corner.{plot_format}"),
    )
    plot_overview(savefig=(plot_dir / f"overview.{plot_format}"))
    plot_overview(
        bands=["nband"],
        savefig=(plot_dir / f"overview_nband.{plot_format}"),
    )
    plot_overview(
        bands=["hband", "kband", "lband", "mband"],
        savefig=(plot_dir / f"overview_hlkmband.{plot_format}"),
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
    plot_components(
        components,
        dim,
        0.1,
        10.5,
        norm=0.3,
        zoom=zoom,
        savefig=plot_dir / "image_nband.png",
    )
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=True,
        savefig=assets_dir / "disc.csv",
        fit_method=OPTIONS.fit.fitter,
    )
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=False,
        savefig=assets_dir / "disc",
        fit_method=OPTIONS.fit.fitter,
    )

    max_plots, number = 20, True
    # HACK: Remove that for now as it doesn't work to do it in the functions
    fits_files = [s for s in fits_files if "GRAV" not in s.stem]
    fits_files = [s for s in fits_files if "PION" not in s.stem]

    # oifits.plot(
    #     fits_files,
    #     bands=["nband"],
    #     kind="combined",
    #     plots=["uv"],
    #     save_dir=plot_dir / "uv.png",
    # )
    #
    # bands = ["lband", "nband"]
    # for band in bands:
    #     oifits.plot_baselines(
    #         fits_files,
    #         band,
    #         "vis",
    #         max_plots=max_plots,
    #         number=number,
    #         save_dir=plot_dir,
    #     )
    #     oifits.plot_baselines(
    #         fits_files,
    #         band,
    #         "visphi",
    #         max_plots=max_plots,
    #         number=number,
    #         save_dir=plot_dir,
    #     )
    #     oifits.plot_baselines(
    #         fits_files,
    #         band,
    #         "t3",
    #         max_plots=max_plots,
    #         number=number,
    #         save_dir=plot_dir,
    #     )
