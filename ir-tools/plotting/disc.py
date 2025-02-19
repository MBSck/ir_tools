import pickle
from functools import partial
from pathlib import Path
from typing import Any

import astropy.units as u
import emcee
import numpy as np
from dynesty import DynamicNestedSampler
from numpy.typing import NDArray
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
    compute_t3,
    compute_vis,
)

from ..tables import best_fit_parameters
from . import io, oiplot

np.seterr(over="ignore", divide="ignore")


def ptform():
    pass


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "results" / "disc" / "2025-02-13"
    path /= "all_nband_temp_power_xy_free"

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
    # plot_corner(
    #     sampler,
    #     labels,
    #     units,
    #     savefig=(plot_dir / f"corner.{plot_format}"),
    # )
    # plot_overview(savefig=(plot_dir / f"overview.{plot_format}"))
    # plot_overview(
    #     bands=["nband"],
    #     savefig=(plot_dir / f"overview_nband.{plot_format}"),
    # )
    # plot_overview(
    #     bands=["hband", "kband", "lband", "mband"],
    #     savefig=(plot_dir / f"overview_hlkmband.{plot_format}"),
    # )
    # plot_fit(components=components, savefig=(plot_dir / f"disc.{plot_format}"))
    # plot_fit(
    #     components=components,
    #     bands=["nband"],
    #     savefig=(plot_dir / f"disc_nband.{plot_format}"),
    # )
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
    #     [3.5] * u.um,
    #     norm=0.3,
    #     zoom=zoom,
    #     savefig=plot_dir / "image_lband.png",
    # )
    # plot_components(
    #     components,
    #     dim,
    #     0.1,
    #     [10.5] * u.um,
    #     norm=0.3,
    #     zoom=zoom,
    #     savefig=plot_dir / "image_nband.png",
    # )
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

    max_plots, number = 20, True
    # HACK: Remove that for now as it doesn't work to do it in the functions
    fits_files = [s for s in fits_files if "GRAV" not in s.stem]
    fits_files = [s for s in fits_files if "PION" not in s.stem]

    hduls = io.sort(io.read(fits_files), by="date")
    oiplot.plot(
        hduls,
        bands=["nband"],
        kind="combined",
        plots=["uv"],
        number=number,
        save_dir=plot_dir / "uv_uts.png",
    )

    def model_func(ucoord, vcoord, wl, components, observable) -> NDArray[Any]:
        ucoord = u.Quantity(np.insert(ucoord, 0, 0), u.m).reshape(1, -1)
        vcoord = u.Quantity(np.insert(vcoord, 0, 0), u.m).reshape(1, -1)
        wl = u.Quantity(wl, u.m).to(u.um)
        complex_vis = np.sum(
            [comp.compute_complex_vis(ucoord, vcoord, wl) for comp in components],
            axis=0,
        ).T

        flux_model = complex_vis[0]
        if OPTIONS.model.output == "normed":
            complex_vis /= flux_model

        return compute_vis(complex_vis[1:])

    bands = ["nband"]
    for band in bands:
        oiplot.plot_vs_spf(
            hduls,
            band,
            "vis",
            max_plots=max_plots,
            number=number,
            save_dir=plot_dir / "vis_vs_spf.png",
        )
        oiplot.plot_vs_spf(
            hduls,
            band,
            "vis",
            model_func=partial(model_func, components=components, observable="vis"),
            max_plots=max_plots,
            number=number,
            save_dir=plot_dir / "vis_vs_spf_model.png",
        )
        oiplot.plot_vs_spf(
            hduls,
            band,
            "visphi",
            ylims=[-20, 20],
            max_plots=max_plots,
            number=number,
            save_dir=plot_dir / "visphi_vs_spf.png",
        )
        oiplot.plot_vs_spf(
            hduls,
            band,
            "t3",
            ylims=[-20, 55],
            max_plots=max_plots,
            number=number,
            save_dir=plot_dir / "t3_vs_spf.png",
        )

    hduls = io.sort(io.read(list(fits_dir.glob("*.fits"))), by="instrument")
    oiplot.plot(
        hduls,
        kind="combined",
        plots=["uv"],
        color_by="instrument",
        save_dir=plot_dir / "uv_all.png",
    )
