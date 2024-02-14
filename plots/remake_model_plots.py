from pathlib import Path

import astropy.units as u
import numpy as np

from ppdmod import custom_components
from ppdmod import data
from ppdmod import fitting
from ppdmod import plot
from ppdmod import utils
from ppdmod.parameter import STANDARD_PARAMETERS, Parameter
from ppdmod.options import OPTIONS


# NOTE: Turns off numpys automated parellelization.
# os.environ["OMP_NUM_THREADS"] = "1"

# TODO: Check wavelength axis for opacity interpolation.
OPTIONS["fit.data"] = ["flux", "vis", "t3phi"]
OPTIONS["data.binning.window"] = 0.1*u.um
# data.set_fit_wavelengths([1.6, 2.25, 3.5]*u.um)
# data.set_fit_wavelengths([3.5]*u.um)
data.set_fit_wavelengths([8., 9., 10., 11.3, 12.5]*u.um)
fits_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/1.7.6/mat_tools/nband/flux")
# fits_files = list(fits_dir.glob("*.fits"))
fits_files = [fits_dir / "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT_flux_avg.fits"]
data.set_data(fits_files)

wavelength_axes = list(
    map(lambda x: data.ReadoutFits(x).wavelength, fits_files))
wavelength_axes = np.sort(np.unique(np.concatenate(wavelength_axes)))

flux_file = Path("/Users/scheuck/Data/flux_data/hd142666/HD142666_stellar_model.txt.gz")
wavelength, flux = np.loadtxt(flux_file, comments="#", unpack=True)[:2]
matisse_flux = utils.opacity_to_matisse_opacity(
    wavelength_axes, wavelength_grid=wavelength*u.um, opacity=flux*u.Jy).value*u.Jy
star_flux = Parameter(**STANDARD_PARAMETERS["f"])
star_flux.value, star_flux.wavelength = matisse_flux, wavelength_axes

weights = np.array([42.8, 9.7, 43.5, 1.1, 2.3, 0.6])/100
qval_file_dir = Path("/Users/scheuck/Data/opacities/QVAL")
qval_files = ["Q_Am_Mgolivine_Jae_DHS_f1.0_rv0.1.dat",
              "Q_Am_Mgolivine_Jae_DHS_f1.0_rv1.5.dat",
              "Q_Am_Mgpyroxene_Dor_DHS_f1.0_rv1.5.dat",
              "Q_Fo_Suto_DHS_f1.0_rv0.1.dat",
              "Q_Fo_Suto_DHS_f1.0_rv1.5.dat",
              "Q_En_Jaeger_DHS_f1.0_rv1.5.dat"]
qval_paths = list(map(lambda x: qval_file_dir / x, qval_files))
opacity = utils.linearly_combine_opacities(
    weights, qval_paths, wavelength_axes)
continuum_opacity = utils.opacity_to_matisse_opacity(
    wavelength_axes, qval_file=qval_file_dir / "Q_amorph_c_rv0.1.dat")

kappa_abs = Parameter(**STANDARD_PARAMETERS["kappa_abs"])
kappa_abs.value, kappa_abs.wavelength = opacity, wavelength_axes
kappa_cont = Parameter(**STANDARD_PARAMETERS["kappa_cont"])
kappa_cont.value, kappa_cont.wavelength = continuum_opacity, wavelength_axes

dim, distance = 32, 148.3
OPTIONS["model.constant_params"] = {
    "dim": dim, "dist": distance, "eff_temp": 7500,
    "f": star_flux, "pixel_size": 0.1,
    "eff_radius": 1.75, "inner_temp": 1500,
    "kappa_abs": kappa_abs, "kappa_cont": kappa_cont}

# rin = Parameter(**STANDARD_PARAMETERS["rin"])
# rout = Parameter(**STANDARD_PARAMETERS["rout"])
# p = Parameter(**STANDARD_PARAMETERS["p"])
# inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

# rin.value = 1.
# rout.value = 7.
# p.value = 0.5
# inner_sigma.value = 1e-3

# rin.set(min=0.5, max=3)
# rout.set(min=0.5, max=4)
# p.set(min=0., max=1.)
# inner_sigma.set(min=0, max=1e-2)

# rout.free = True

# inner_ring = {"rin": rin, "rout": rout, "inner_sigma": inner_sigma, "p": p}
# inner_ring_labels = [f"ir_{label}" for label in inner_ring]

rin = Parameter(**STANDARD_PARAMETERS["rin"])
a = Parameter(**STANDARD_PARAMETERS["a"])
phi = Parameter(**STANDARD_PARAMETERS["phi"])
p = Parameter(**STANDARD_PARAMETERS["p"])
inner_sigma = Parameter(**STANDARD_PARAMETERS["inner_sigma"])

rin.value = 13
a.value = 0.5
phi.value = 130
p.value = 0.5
inner_sigma.value = 1e-3

# NOTE: Set outer radius to be constant and calculate flux once?
rin.set(min=1, max=30)
p.set(min=0., max=1.)
inner_sigma.set(min=0, max=1e-2)
a.set(min=0., max=1.)
phi.set(min=0, max=360)

outer_ring = {"rin": rin, "a": a, "phi": phi,
              "inner_sigma": inner_sigma, "p": p}
outer_ring_labels = [f"or_{label}" for label in outer_ring]

# q = Parameter(**STANDARD_PARAMETERS["q"])
pa = Parameter(**STANDARD_PARAMETERS["pa"])
elong = Parameter(**STANDARD_PARAMETERS["elong"])
cont_weight = Parameter(**STANDARD_PARAMETERS["cont_weight"])

# q.value = 0.5
pa.value = 145
elong.value = 0.5
cont_weight.value = 130

# q.set(min=0., max=1.)
pa.set(min=0, max=360)
elong.set(min=0, max=1)
cont_weight.set(min=0., max=1.)

OPTIONS["model.shared_params"] = {"pa": pa, "elong": elong,
                                  "cont_weight": cont_weight}
shared_params_labels = [f"sh_{label}"
                        for label in OPTIONS["model.shared_params"]]

# OPTIONS["model.components_and_params"] = [
#     ["Star", {}],
#     ["AnalyticalTempGradient", inner_ring],
#     ["AnalyticalAsymmetricTempGradient", outer_ring],
# ]

OPTIONS["model.components_and_params"] = [
    ["Star", {}],
    # ["AnalyticalGreyBody", inner_ring],
    ["AnalyticalAsymmetricGreyBody", outer_ring],
]

# component_labels = ["Star", "Inner Ring", "Outer Ring"]
# component_labels = ["Star", "Inner Ring"]
component_labels = ["Star", "Outer Ring"]

# labels = inner_ring_labels + outer_ring_labels + shared_params_labels
labels = outer_ring_labels + shared_params_labels
# labels = inner_ring_labels + shared_params_labels

OPTIONS["model.modulation.order"] = 1
OPTIONS["model.gridtype"] = "logarithmic"

OPTIONS["fit.method"] = "emcee"

result_dir = Path("/Users/scheuck/Data/model_results/2024-01-21/results_model_23:51:13")
# OPTIONS["plot.color.background"] = "black"

if __name__ == "__main__":
    sampler = np.load(result_dir / "sampler.npy", allow_pickle=True).tolist()
    theta = fitting.get_best_fit(sampler, discard=2500, method="quantile")
    components_and_params, shared_params = fitting.set_params_from_theta(theta)
    components = custom_components.assemble_components(
            components_and_params, shared_params)
    theta2 = fitting.get_best_fit(sampler, discard=2500, method="quantile")
    components_and_params2, shared_params2 = fitting.set_params_from_theta(theta2)
    components2 = custom_components.assemble_components(
            components_and_params, shared_params)
    breakpoint()

    # HACK: This is to include innermost radius for rn.
    innermost_radius = components[1].params["rin"]
    for component in components:
        component.params["rin0"] = innermost_radius

    new_params = dict(zip(labels, theta))
    # plot.save_fits(
    #         4096, 0.1, distance,
    #         OPTIONS["fit.wavelengths"], components,
    #         component_labels, opacities=[kappa_abs, kappa_cont],
    #         savefits=result_dir / "model.fits",
    #         options=OPTIONS, object_name="HD 142666")

    # plot.plot_fit(
    #         new_params["sh_elong"], new_params["sh_pa"],
    #         components=components,
    #         savefig=result_dir / "fit_results.pdf")

    plot.plot_fit(
            new_params["sh_elong"], new_params["sh_pa"],
            components=components,
            savefig=result_dir / "fit_results.pdf",
            wavelength_range=[8, 13]*u.um, ylimits={"vis2": [0, 0.2]})
