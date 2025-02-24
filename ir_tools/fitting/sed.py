from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import oimodeler as oim
from astropy.modeling.models import BlackBody


def reparametrise_sum_to_one(keys: np.ndarray, theta: np.ndarray):
    indices = list(map(keys.index, filter(lambda x: "weight" in x, keys)))
    normed_params = [theta[indices][0] / 1e2]
    for param in theta[indices][1:] / 1e2:
        normed_params.append(param * (1 - np.sum(normed_params)))

    params = theta.copy()
    params[indices] = np.array(normed_params) * 1e2
    return params


def _logProbability(self, theta: np.ndarray):
    for iparam, parami in enumerate(self.freeParams.values()):
        parami.value = theta[iparam]

    theta = reparametrise_sum_to_one(list(self.freeParams.keys()), theta)
    for i, key in enumerate(self.freeParams):
        val = theta[i]
        low, up = self.limits[key]
        if not low < val < up:
            return -np.inf

    self.simulator.compute(computeChi2=True, dataTypes=self.dataTypes)
    return -0.5 * self.simulator.chi2


class oimSED(oim.oimPt):
    name = "SED"
    shortname = "SED"

    def __init__(self, **kwargs):
        """The class's constructor."""
        super().__init__(**kwargs)
        self.params["f"].free = False

        self.params["tempc"] = oim.oimParam(**oim._standardParameters["amp"])
        self.params["tempc"].description = "Characterizes the temperature."
        self.params["tempc"].set(min=0, max=1500)
        self.params["tempc"].unit = u.K

        self.params["offset"] = oim.oimParam(**oim._standardParameters["amp"])
        self.params["offset"].description = "The offset for the weights"
        self.params["offset"].set(min=16, max=20)

        self.params["pah_scale"] = oim.oimParam(**oim._standardParameters["amp"])
        self.params["pah_scale"].description = "The weight of the PAH (flux) emission"
        self.params["pah_scale"].set(min=0, max=10)

        self.params["pah"] = oim.oimParam(**oim._standardParameters["x"])
        self.params["pah"].description = "The PAHs (flux) emission"
        self.params["pah"].unit = u.Jy

        self.materials = list(
            ["_".join(key.split("_")[1:]) for key in kwargs.keys() if "kappa" in key]
        )

        for material in self.materials:
            for prefix in ["kappa", "weight"]:
                key = "x" if prefix == "kappa" else "amp"
                param = oim.oimParam(**oim._standardParameters[key])
                param.name = f"{prefix}_{material}"
                param.description = f"The mass fraction for {param.name}"
                if prefix == "kappa":
                    param.unit = u.cm**2 / u.g
                else:
                    param.set(min=0, max=100)
                    param.unit = u.pct

                self.params[param.name] = param

        self._t = np.array([0])  # constant value <=> static model
        self._wl = None
        self._eval(**kwargs)

    def _visFunction(self, ucoord, vcoord, rho, wl, t):
        bb = BlackBody(self.params["tempc"]() * self.params["tempc"].unit)(wl * u.m)
        pah = (
            self.params["pah_scale"]()
            * self.params["pah"](wl)
            * self.params["pah"].unit
        )

        # NOTE: Divide by 1e2 to fit percentages
        opacity = np.sum(
            [
                self.params[f"weight_{material}"]()
                / 1e2
                * self.params[f"kappa_{material}"](wl)
                for material in self.materials
            ],
            axis=0,
        )

        flux = (bb * opacity * u.sr * 10.0 ** -self.params["offset"]()).to(u.Jy)
        return (flux + pah).value


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
            model_kwargs[f"kappa_{shortname}_{size}"] = oim.oimInterp(
                "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
            )

    grid, value = np.load(opacity_dir / "optool" / "preibisch_amorph_c_rv0.1.npy")
    model_kwargs["kappa_continuum"] = oim.oimInterp(
        "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
    )

    grid, value = np.loadtxt(opacity_dir / "boekel" / "PAH.kappa", unpack=True)
    model_kwargs["pah"] = oim.oimInterp(
        "wl", wl=grid * 1e-6, values=value, kind="linear", extrapolate=False
    )

    # NOTE: Model creation
    model = oim.oimModel(oimSED(**model_kwargs))

    # NOTE: Simulate and plot the initial model observables and compute the associated reduced Chi2
    sim = oim.oimSimulator(data=data, model=model)
    sim.compute(computeChi2=True, computeSimulatedData=True)
    # fig0, ax0 = sim.plot(["VIS2DATA", "T3PHI"])
    # print(f"Chi2r = {sim.chi2r}")

    # NOTE: Perfoming the model-fitting
    nsteps, ndiscard, nwalkers = int(1e3), int(2.5e2), 50
    fit = oim.oimFitterEmcee(data, model, nwalkers=nwalkers)
    fit._logProbability = lambda theta: _logProbability(fit, theta)
    fit.prepare(init="random")
    fit.run(nsteps=nsteps, progress=True)

    # # NOTE: Plot the walkers path and make the corner plot
    # figWalkers, axeWalkers = fit.walkersPlot()
    figCorner, axeCorner = fit.cornerPlot(discard=ndiscard)
    # figSim, axSim = fit.simulator.plot(["OI_FLUX"])
    # plt.show()

    # NOTE: Get the best-fit reduced chi2 and best-fit values of the free parameters (+ their errors)
    keys = list(model.getParameters(free=True).keys())
    best, err_l, err_u, err = fit.getResults(mode="best", discard=ndiscard)
    best = reparametrise_sum_to_one(keys, best)

    for iparam, parami in enumerate(model.getParameters(free=True).values()):
        parami.value = best[iparam]

    print(f"Chi2r = {fit.simulator.chi2r}")

    wavelengths = data.data[0][3].data["EFF_WAVE"]
    flux = data.data[0][5].data["FLUXDATA"][0]
    fluxerr = data.data[0][5].data["FLUXERR"][0]
    model_flux = model.getComplexCoherentFlux(0, 0, wavelengths, 0)

    fig, ax = plt.subplots()
    ax.plot(wavelengths, flux, label="Data")
    ax.fill_between(wavelengths, flux - fluxerr, flux + fluxerr, alpha=0.5)
    ax.plot(wavelengths, model_flux, label="Model", color="red")
    plt.show()
