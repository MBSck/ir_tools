from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import oimodeler as oim
from ppdmod.utils import (
    load_data,
    qval_to_opacity,
    windowed_linspace,
)

from .model import oimAsymTempGrad

data_dir = Path().home() / "Data"
files = list((data_dir / "fitting" / "hd142527").glob("HD_*.fits"))

data = oim.oimData(files)
f1 = oim.oimRemoveArrayFilter(targets="all", arr=["OI_FLUX", "OI_VIS2"])
f2 = oim.oimDataTypeFilter(targets="all", dataType=["T3AMP"])
data.setFilter(oim.oimDataFilter([f1, f2]))

source_dir = data_dir / "results" / "hd142527"
grid, value = np.load(source_dir / "silicate_grf_opacities.npy")
kappa_abs = oim.oimInterp("wl", wl=grid * 1e-6, values=value)

grid, value = load_data(
    data_dir / "opacities" / "qval" / "Q_amorph_c_rv0.1.dat", load_func=qval_to_opacity
)
kappa_cont = oim.oimInterp("wl", wl=grid * 1e-6, values=value)

grid, value = np.loadtxt(
    data_dir / "flux" / "hd142527" / "HD142527_stellar_model.txt",
    usecols=(0, 2),
    unpack=True,
)

shared_input = {
    "dim": 128,
    "dist": 158.51,
    "eff_temp": 6750,
    "eff_radius": 3.46,
    "pa": 352,
    "elong": 1 / 0.8,
    "kappa_abs": kappa_abs,
    "kappa_cont": kappa_cont,
}

oim.oimOptions.model.grid.type = "linear"
itg = oimAsymTempGrad(rin=0.1, rout=1, **shared_input)
otg = oimAsymTempGrad(rin=2, rout=4, **shared_input)

test = oim.oimInterp("wl", wl=grid * 1e-6, values=value)
s = oim.oimPt()
s.params["f"].free = False
s.params["f"] = oim.oimParamInterpolatorWl(s.params["f"], wl=grid * 1e-6, values=value)

itg.params["rin"].set(min=0, max=3)
itg.params["rout"].set(min=0, max=3)
itg.params["cont_weight"].set(min=0, max=1)
itg.params["p"].set(min=-1, max=1)
itg.params["sigma0"].set(min=0, max=1e-1)
itg.params["pa"].free = False
itg.params["elong"].set(min=1, max=50)

otg.params["rin"].set(min=0.2, max=5)
otg.params["rout"].set(min=1.5, max=10)
otg.params["cont_weight"].set(min=0, max=1)
otg.params["p"].set(min=-1, max=1)
otg.params["sigma0"].set(min=0, max=1e-1)
otg.params["pa"].free = False
otg.params["elong"].set(min=1, max=50)

itg.params["pa"] = oim.oimParamLinker(otg.params["pa"])
itg.params["elong"] = oim.oimParamLinker(otg.params["elong"])

wavelengths = {
    "hband": [1.7],
    "kband": [2.15],
    "lband": windowed_linspace(3.1, 3.8, 0.2),
    "mband": windowed_linspace(4.65, 4.9, 0.2),
    "nband": windowed_linspace(8.25, 12.75, 0.2),
}
bands = ["lband", "mband", "nband"]
wavelengths = np.concatenate([wavelengths[band] for band in bands]) * 1e-6
s._t = itg._t = otg._t = 59300.23175615755
s._wl = itg._wl = otg._wl = wavelengths

model = oim.oimModel([s, itg, otg])
breakpoint()
sim = oim.oimSimulator(data=data, model=model)
sim.compute(computeChi2=True, computeSimulatedData=True)
print("Chi2r = {}".format(sim.chi2r))

fit = oim.oimFitterEmcee(data, model, nwalkers=25)
fit.prepare(init="random")
fit.run(nsteps=100, progress=True)

figWalkers, axeWalkers = fit.walkersPlot()
fig0, ax0 = sim.plot(["FLUXDATA", "VISAMP", "T3PHI"])
best, err_l, err_u, err = fit.getResults(mode="best", discard=10)

s._wl = itg._wl = otg._wl = None
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
model.showModel(
    512,
    0.35,
    wl=3.5e-6,
    swapAxes=True,
    fromFT=False,
    normPow=0.3,
    axe=ax[0],
    colorbar=False,
)
model.showModel(
    512,
    0.35,
    wl=10.5e-6,
    swapAxes=True,
    fromFT=False,
    normPow=0.3,
    axe=ax[1],
    colorbar=False,
)
ax[0].set_title(r"Direct Image (@3.5$\mathrm{\mu}$m")
ax[1].set_title(r"Direct Image (@10.5$\mathrm{\mu}$m")
ax[1].get_yaxis().set_visible(False)
plt.show()
