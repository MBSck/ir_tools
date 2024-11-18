from pathlib import Path

import matplotlib.pyplot as plt
import oimodeler as oim


# NOTE: Load the simulated data from ASPRO and apply several filters to
# keep only VIS2DATA and T3PHI, and a narrower wavelength range, for model fitting
# TODO: Make a converted dataset that has all the vis in the right place
data_dir = Path().home() / "Data" / "fitting_data" / "hd142527"

fits_files = []
for key in ["_L_"]:
    fits_files.extend(list(data_dir.glob(f"*{key}*.fits")))

data = oim.oimData(fits_files)
f1 = oim.oimRemoveArrayFilter(targets="all", arr=["OI_FLUX", "OI_T3"])
f2 = oim.oimWavelengthRangeFilter(targets="all", wlRange=[3.1e-6, 3.4e-6])
data.setFilter(oim.oimDataFilter([f1, f2]))

# NOTE: Model creation
r = oim.oimEIRing(dim=128, d=2, pa=352)
model = oim.oimModel([r])

# NOTE: Simulate and plot the initial model observables and compute the associated reduced Chi2
sim = oim.oimSimulator(data=data, model=model)
sim.compute(computeChi2=True, computeSimulatedData=True)
fig0, ax0 = sim.plot(["VIS2DATA"])
print(f"Chi2r = {sim.chi2r}")

# NOTE: Specifying the parameter space
r.params["pa"].free = False
r.params["f"].free = False
r.params["d"].set(min=0, max=4)

# NOTE: Perfoming the model-fitting
fit = oim.oimFitterEmcee(data, model, nwalkers=25)
fit.prepare(init="random")
fit.run(nsteps=5000, progress=True)

discard = 1000
# NOTE: Plot the walkers path and make the corner plot
figWalkers, axeWalkers = fit.walkersPlot()
figCorner, axeCorner=fit.cornerPlot(discard=discard)
figSim, axSim=fit.simulator.plot(["VIS2DATA"])

# NOTE: Get the best-fit reduced chi2 and best-fit values of the free parameters (+ their errors)
best, err_l, err_u, err = fit.getResults(mode="best", discard=discard)
print(f"Chi2r = {fit.simulator.chi2r}")

# NOTE: Plotting images of the model
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
model.showModel(128, 0.15, swapAxes=True, fromFT=False,
                normPow=0.2, axe=ax[0], colorbar=False)
model.showModel(128, 0.15, swapAxes=True, fromFT=True,
                normPow=0.2, axe=ax[1], colorbar=False)
ax[1].get_yaxis().set_visible(False)
ax[0].set_title("Direct Image")
ax[1].set_title("From FFT")
plt.show()
