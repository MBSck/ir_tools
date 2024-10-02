from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod.analysis import save_fits
from ppdmod.basic_components import assemble_components
from ppdmod.data import get_all_wavelengths, set_data
from ppdmod.parameter import Parameter
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import get_opacity, load_data, qval_to_opacity


if __name__ == "__main__":
    path = Path("/Users/scheuck/Code/modelling/ppdmod/tests/data")

    wavelengths = {"hband": [1.6]*u.um, "kband": [2.25]*u.um,
                   "lband": [3.2]*u.um, "nband": [8., 9., 10., 11.3, 12.5]*u.um}

    fits_files = list((path / "fits" / "hd142527").glob("*fits"))
    wavelength = np.concatenate((wavelengths["lband"], wavelengths["nband"]))
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis2", "t3"])
    wavelengths = get_all_wavelengths()

    weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0])/100
    names = ["pyroxene", "forsterite", "enstatite", "silica"]
    fmaxs = [1.0, 1.0, 1.0, 0.7]
    sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

    _, opacity = get_opacity(path, weights, sizes, names,
                             "qval", wavelengths.value, fmaxs)

    cont_opacity_file = path / "qval" / "Q_amorph_c_rv0.1.dat"
    wl_cont, cont_opacity = load_data(cont_opacity_file, load_func=qval_to_opacity)
    cont_opacity = np.interp(wavelengths.value, wl_cont, cont_opacity)

    kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
    kappa_abs.value, kappa_abs.wavelength = opacity, wavelengths
    kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
    kappa_cont.value, kappa_cont.wavelength = cont_opacity, wavelengths

    pa = Parameter(**STANDARD_PARAMETERS.pa)
    inc = Parameter(**STANDARD_PARAMETERS.inc)
    cont_weight = Parameter(**STANDARD_PARAMETERS.cont_weight)

    pa.value = 163
    inc.value = 0.5
    cont_weight.value = 0.40             # Relative contribution (adds to 1). Mass fractions

    pa.set(min=0, max=180)
    inc.set(min=0.3, max=0.95)
    cont_weight.set(min=0.3, max=0.8)

    OPTIONS.model.shared_params = {"pa": pa, "inc": inc,
                                   "cont_weight": cont_weight}

    wl_flux, flux = load_data(path / "flux" / "hd142527" / "HD142527_stellar_model.txt")
    flux_interpn = np.interp(wavelengths.value, wl_flux, flux)
    star_flux = Parameter(**STANDARD_PARAMETERS.f)
    star_flux.value, star_flux.wavelength = flux_interpn, wavelengths

    rin = Parameter(**STANDARD_PARAMETERS.rin)
    rout = Parameter(**STANDARD_PARAMETERS.rout)
    p = Parameter(**STANDARD_PARAMETERS.p)
    inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)
    a = Parameter(**STANDARD_PARAMETERS.a)
    phi = Parameter(**STANDARD_PARAMETERS.phi)

    rin.value = 1.
    rout.value = 2.
    p.value = 0.5
    inner_sigma.value = 1e-3
    a.value = 0.5
    phi.value = 130

    rin.set(min=0.5, max=5)
    rout.set(min=1.5, max=6)
    p.set(min=0., max=1.)
    inner_sigma.set(min=0, max=1e-2)

    rout.free = True

    # inner_ring = {"x": x, "y": y, "rin": rin,
    #               "rout": rout, "inner_sigma": inner_sigma, "p": p}
    inner_ring = {"rin": rin, "rout": rout, "a": a, "phi": phi, "inner_sigma": inner_sigma, "p": p}
    inner_ring_labels = [f"ir_{label}" for label in inner_ring]

    rin = Parameter(**STANDARD_PARAMETERS.rin)
    p = Parameter(**STANDARD_PARAMETERS.p)
    inner_sigma = Parameter(**STANDARD_PARAMETERS.inner_sigma)
    a = Parameter(**STANDARD_PARAMETERS.a)
    phi = Parameter(**STANDARD_PARAMETERS.phi)

    rin.value = 13
    p.value = 0.5
    inner_sigma.value = 1e-3
    a.value = 0.5
    phi.value = 130

    # NOTE: Set outer radius to be constant and calculate flux once?
    rin.set(min=1, max=40)
    p.set(min=0., max=1.)
    inner_sigma.set(min=0, max=1e-2)

    outer_ring = {"rin": rin, "a": a, "phi": phi, "inner_sigma": inner_sigma, "p": p}

    OPTIONS.model.components_and_params = [
        ["Star", {}],
        ["GreyBody", inner_ring],
        ["AsymmetricGreyBody", outer_ring],
    ]

    component_labels = ["Star", "Inner Ring", "Outer Ring"]
    components = assemble_components(
            OPTIONS.model.components_and_params,
            OPTIONS.model.shared_params)

    save_fits(4096, 0.1, 157.3,
              components, component_labels,
              opacities=[kappa_abs, kappa_cont],
              savefits="model.fits",
              object_name="HD 142666")
