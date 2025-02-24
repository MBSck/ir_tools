from functools import partial
from pathlib import Path
from typing import Any

import astropy.units as u
import emcee
import numpy as np
from dynesty import DynamicNestedSampler
from numpy.typing import NDArray
from ppdmod.components import AsymTempGrad, Point
from ppdmod.data import set_data
from ppdmod.fitting import (
    get_labels,
    get_units,
)
from ppdmod.options import OPTIONS
from ppdmod.parameter import Parameter
from ppdmod.plot import plot_components
from ppdmod.utils import compute_vis, load_data, qval_to_opacity, windowed_linspace

from ..plotting import io, oiplot

np.seterr(over="ignore", divide="ignore")


def ptform():
    pass


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "results" / "disc" / "test"
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

    def au_to_mas(r: float, dist=158.51) -> float:
        return r * 1e3 / dist

    wavelengths = {
        "hband": [1.7] * u.um,
        "kband": [2.15] * u.um,
        "lband": windowed_linspace(3.1, 3.8, 0.2) * u.um,
        "mband": windowed_linspace(4.65, 4.9, 0.2) * u.um,
        "nband": windowed_linspace(8.25, 12.75, 0.2) * u.um,
    }
    # bands = ["hband", "kband", "lband", "mband", "nband"]
    # bands = ["lband", "mband", "nband"]
    bands = ["nband"]
    wavelengths = np.concatenate([wavelengths[band] for band in bands])

    fit_data = ["flux", "vis", "t3"]
    fits_files = list((fits_dir).glob("*_N_*.fits"))
    data = set_data(
        fits_files,
        wavelengths=wavelengths,
        fit_data=fit_data,
    )

    grid, value = np.loadtxt(
        data_dir / "flux" / "hd142527" / "HD142527_stellar_model.txt",
        usecols=(0, 2),
        unpack=True,
    )
    flux_star = Parameter(grid=grid, value=value, base="f")

    SOURCE_DIR = data_dir / "results" / "hd142527"

    method = "grf"
    grid, value = np.load(SOURCE_DIR / f"silicate_{method}_opacities.npy")
    kappa_abs = Parameter(grid=grid, value=value, base="kappa_abs")

    grid, value = load_data(
        data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat",
        load_func=qval_to_opacity,
    )
    kappa_cont = Parameter(grid=grid, value=value, base="kappa_cont")

    shared_params = {
        "dim": 32,
        "dist": 158.51,
        "eff_temp": 6750,
        "eff_radius": 3.46,
        "pa": 352,
        "cinc": 0.82,
        "q": -0.4,
        "temp0": 450,
        "kappa_abs": kappa_abs,
        "kappa_cont": kappa_cont,
    }

    s = Point(label="Star", fr=flux_star, **shared_params)
    inr = AsymTempGrad(
        label="Inner Ring",
        rin=0.4,
        rout=2.5,
        p=0.21,
        sigma0=3.95e-4,
        rho1=0.82,
        theta1=-180,
        weight_cont=85,
        **shared_params,
    )
    our = AsymTempGrad(
        label="Outer Ring",
        rin=3.8,
        rout=4,
        p=-0.9,
        sigma0=2.33e-3,
        rho1=0.98,
        theta1=-154,
        weight_cont=48,
        **shared_params,
    )

    components = [s, inr, our]
    labels, units = get_labels(components), get_units(components)
    component_labels = [component.label for component in components]

    zoom = 5
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

    plot_format = "png"
    max_plots, number = 20, True
    # HACK: Remove that for now as it doesn't work to do it in the functions
    fits_files = [s for s in fits_files if "GRAV" not in s.stem]
    fits_files = [s for s in fits_files if "PION" not in s.stem]

    def model_func(ucoord, vcoord, wl, components, observable) -> NDArray[Any]:
        ucoord = u.Quantity(np.insert(ucoord, 0, 0), u.m).reshape(1, -1)
        vcoord = u.Quantity(np.insert(vcoord, 0, 0), u.m).reshape(1, -1)
        wl = u.Quantity(wl, u.m).to(u.um)
        complex_vis = np.sum(
            [comp.compute_complex_vis(ucoord, vcoord, wl) for comp in components],
            axis=0,
        ).T

        flux_model = complex_vis[0]
        vis = compute_vis(complex_vis)
        if observable == "vis2":
            complex_vis /= flux_model
            vis **= 2

        return vis

    hduls = io.sort(io.read(fits_files), by="date")
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
            "t3",
            ylims=[-20, 55],
            max_plots=max_plots,
            number=number,
            save_dir=plot_dir / "t3_vs_spf.png",
        )
