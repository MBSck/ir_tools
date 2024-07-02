from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from ppdmod.basic_components import GreyBody
from ppdmod.data import set_data
from ppdmod.parameter import Parameter
from ppdmod.plot import plot_overview
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import load_data, get_opacity, qval_to_opacity, \
    compute_stellar_radius, compute_effective_baselines

DATA_DIR = Path("/Users/scheuck/Data")
OPTIONS.model.output = "normed"
fits_files = list((DATA_DIR / "fitting_data" / "hd142527").glob("*HAW*"))
wavelength = [3.5] * u.um
data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis"])

wl_flux, flux = load_data(DATA_DIR / "flux_data" / "hd142527" / "HD142527_stellar_model.txt")
star_flux = Parameter(**STANDARD_PARAMETERS.f)
star_flux.wavelength, star_flux.value = wl_flux, flux

weights = np.array([73.2, 8.6, 0.6, 14.2, 2.4, 1.0])/100
names = ["pyroxene", "forsterite", "enstatite", "silica"]
sizes = [[1.5], [0.1], [0.1, 1.5], [0.1, 1.5]]

wl_opacity, roy_opacity = get_opacity(
    DATA_DIR / "opacities", weights, sizes, names, "boekel")

cont_opacity_file = DATA_DIR / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat"
# cont_opacity_file = DATA_DIR / "qval" / "Q_iron_0.10um_dhs_0.7.dat",
wl_cont, cont_opacity = load_data(cont_opacity_file, load_func=qval_to_opacity)

kappa_abs = Parameter(**STANDARD_PARAMETERS.kappa_abs)
kappa_abs.value, kappa_abs.wavelength = roy_opacity, wl_opacity
kappa_cont = Parameter(**STANDARD_PARAMETERS.kappa_cont)
kappa_cont.value, kappa_cont.wavelength = cont_opacity, wl_cont

dim, distance, eff_temp = 32, 158.51, 6500
eff_radius = compute_stellar_radius(10**1.35, eff_temp).value
inclination, pos_angle = 0.79, 37.18*u.deg
ucoord = np.linspace(0, 200) * u.m

inner_ring = GreyBody(dim=dim, dist=distance, rin=0.5,
                      rout = 0.7, sigma0=1e-3, p=-2, cont_weight=0.7,
                      inc=inclination, pa=pos_angle,
                      eff_temp=eff_temp, eff_radius=eff_radius,
                      kappa_abs=kappa_abs, kappa_cont=kappa_cont)

outer_ring = GreyBody(dim=dim, dist=distance, rin=3,
                      sigma0=1e-3, p=-2, cont_weight=0.7,
                      inc=inclination, pa=pos_angle,
                      eff_temp=eff_temp, eff_radius=eff_radius,
                      kappa_abs=kappa_abs, kappa_cont=kappa_cont)
vis_inner_ring = inner_ring.compute_complex_vis(ucoord, ucoord, wavelength)
vis_outer_ring = outer_ring.compute_complex_vis(ucoord, ucoord, wavelength)
effective_baselines, _ = compute_effective_baselines(ucoord, ucoord, inclination, pos_angle)

fig, axarr = plot_overview(raxis=True, pos_angle=pos_angle, inclination=inclination)
# axarr["vis"].plot(effective_baselines, np.abs(vis_inner_ring).squeeze(), label="Inner Disk")
# axarr["vis"].plot(effective_baselines, np.abs(vis_outer_ring).squeeze(), label="Inner Disk")
plt.savefig("manual_fit_vs_data.pdf", format="pdf")
