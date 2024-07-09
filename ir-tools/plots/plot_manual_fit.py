from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from ppdmod.basic_components import Ring
from ppdmod.data import set_data
from ppdmod.parameter import Parameter
from ppdmod.plot import plot_overview
from ppdmod.options import STANDARD_PARAMETERS, OPTIONS
from ppdmod.utils import load_data, compute_effective_baselines

DATA_DIR = Path("/Users/scheuck/Data")
OPTIONS.model.output = "normed"
fits_files = list((DATA_DIR / "fitting_data" / "hd142527").glob("*HAW*"))
wavelength = [3.2] * u.um
data = set_data(fits_files, wavelengths=wavelength, fit_data=["vis2"])

wl_flux, flux = load_data(DATA_DIR / "flux_data" / "hd142527" / "HD142527_stellar_model.txt")
star_flux = Parameter(**STANDARD_PARAMETERS.f)
star_flux.wavelength, star_flux.value = wl_flux, flux

inclination, pos_angle = 0.79, 37.18*u.deg
ucoord = np.linspace(0, 200, 1024) * u.m

inner_ring = Ring(rin=2, inc=inclination, pa=pos_angle)
outer_ring = Ring(rin=11, inc=inclination, pa=pos_angle)
vis_inner_ring = inner_ring.compute_complex_vis(ucoord, ucoord, wavelength)
vis_outer_ring = outer_ring.compute_complex_vis(ucoord, ucoord, wavelength)

cont_outer = 0.5
vis_combined = np.abs(cont_outer * vis_inner_ring + (1 - cont_outer) * vis_outer_ring) ** 2
vis_inner_ring, vis_outer_ring = np.abs(vis_inner_ring) ** 2, np.abs(vis_outer_ring) ** 2
effective_baselines, _ = compute_effective_baselines(ucoord, ucoord, inclination, pos_angle)

fig, axarr = plot_overview(raxis=True, pos_angle=pos_angle, inclination=inclination)
axarr["vis"].plot(effective_baselines, vis_inner_ring.squeeze(), label="Inner Ring")
axarr["vis"].plot(effective_baselines, vis_outer_ring.squeeze(), label="Outer Ring")
axarr["vis"].plot(effective_baselines, vis_combined.squeeze(), label="Combined Rings")
axarr["vis"].legend()
plt.savefig("manual_fit_vs_data.pdf", format="pdf")
