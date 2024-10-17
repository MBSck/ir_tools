from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import oimodeler as oim
from astropy.modeling.models import BlackBody


class oimSED(oim.oimPt):
    name = "SED"
    shortname = "SED"

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.tempc = oim.oimParam(**oim._standardParameters["amp"])
        self.tempc.description = "Characterizes the temperature."
        self.tempc.set(min=0, max=1500)
        self.tempc.unit = u.K

        self.offset = oim.oimParam(**oim._standardParameters["amp"])
        self.offset.description = "The offset for the weights"
        self.offset.set(min=16, max=20)

        self.pah_scale = oim.oimParam(**oim._standardParameters["amp"])
        self.pah_scale.description = "The weight of the PAH (flux) emission"
        self.pah_scale.set(min=0, max=10)

        self.pah = oim.oimParam(**oim._standardParameters["x"])
        self.pah.description = "The PAHs (flux) emission"

        self.materials = list(
            ["_".join(key.split("_")[1:]) for key in kwargs.keys() if "kappa" in key]
        )

        for name, param in list(filter(lambda x: "kappa" in x[0], kwargs.items())):
            setattr(self, name, param)

            weight = oim.oimParam(**oim._standardParameters["amp"])
            weight.name = f"weight_{'_'.join(name.split('_')[1:])}"
            weight.description = (
                f"The mass fraction for {' '.join(name.split('_')[1:])}"
            )
            weight.set(min=0, max=100)
            weight.unit = u.pct
            setattr(self, weight.name, weight)

        self._t = np.array([0])  # constant value <=> static model
        self._wl = None
        self._eval(**kwargs)

    def _visFunction(self, ucoord, vcoord, rho, wl, t):
        bb = BlackBody(self.tempc() * self.tempc.unit)(wl * u.m)
        pah = self.pah_scale() * self.pah(wl) * self.pah.unit

        breakpoint()
        # TODO: Get interpolator to work here
        opacity = np.sum(
            [
                getattr(self, f"weight_{material}")()
                * getattr(self, f"kappa_{material}")(wl)
                for material in self.materials
            ],
            axis=0,
        )

        # NOTE: Divide by 1e2 to fit percentages
        opacity /= 1e2 * getattr(self, f"kappa_{self.material[0]}").unit
        flux = bb * opacity * u.sr * 10.0 ** -self.factor()
        return flux.to(u.Jy) + pah


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    files = list(
        (data_dir / "fitting_data" / "hd142527" / "sed_fit" / "averaged").glob("*.fits")
    )
    data = oim.oimData(files)
    f1 = oim.oimRemoveArrayFilter(targets="all", arr=["OI_VIS", "OI_T3"])
    f2 = oim.oimWavelengthRangeFilter(targets="all", wlRange=[8e-6, 16e-6])
    data.setFilter(oim.oimDataFilter([f1, f2]))

    opacity_dir = data_dir / "opacities"
    materials = dict(
        zip(
            ["pyrox", "enst", "forst", "sil", "oliv"],
            ["MgPyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"],
        )
    )
    model_kwargs = {}

    for shortname, name in materials.items():
        for size, value in {"small": 0.1, "large": 2.0}.items():
            grid, value = np.loadtxt(
                opacity_dir / "grf" / f"{name}{value}.Combined.Kappa",
                usecols=(0, 2),
                unpack=True,
            )
            interp = oim.oimInterp(
                "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
            )
            param = oim.oimParam(**oim._standardParameters["x"])
            param.name = f"kappa_{shortname}_{size}"
            param.value, param.unit = interp, u.cm**2 / u.g
            model_kwargs[param.name] = param

    param = oim.oimParam(**oim._standardParameters["x"])
    param.name = "kappa_continuum"
    grid, value = np.load(opacity_dir / "optool" / "preibisch_amorph_c_rv0.1.npy")
    interp = oim.oimInterp(
        "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
    )
    param.value, param.unit = interp, u.cm**2 / u.g
    model_kwargs[param.name] = param

    param = oim.oimParam(**oim._standardParameters["x"])
    param.name = "pah"
    wl, value = np.loadtxt(opacity_dir / "boekel" / "PAH.kappa", unpack=True)
    interp = oim.oimInterp(
        "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
    )
    param.value, param.unit = interp, u.Jy
    model_kwargs[param.name] = param

    # NOTE: Model creation
    model = oim.oimModel(oimSED(**model_kwargs))

    # NOTE: Simulate and plot the initial model observables and compute the associated reduced Chi2
    sim = oim.oimSimulator(data=data, model=model)
    sim.compute(computeChi2=True, computeSimulatedData=True)
    fig0, ax0 = sim.plot(["VIS2DATA", "T3PHI"])
    print(f"Chi2r = {sim.chi2r}")

    # NOTE: Perfoming the model-fitting
    fit = oim.oimFitterEmcee(data, model, nwalkers=25)
    fit.prepare(init="random")
    fit.run(nsteps=1000, progress=True)

    # NOTE: Plot the walkers path and make the corner plot
    figWalkers, axeWalkers = fit.walkersPlot()
    figCorner, axeCorner = fit.cornerPlot(discard=200)
    figSim, axSim = fit.simulator.plot(["FLUXDATA"])
    plt.show()

    # NOTE: Get the best-fit reduced chi2 and best-fit values of the free parameters (+ their errors)
    best, err_l, err_u, err = fit.getResults(mode="best", discard=10)
    print(f"Chi2r = {fit.simulator.chi2r}")
