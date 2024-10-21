from pathlib import Path

import matplotlib.pyplot as plt
import oimodeler as oim

from .sort_data import get_target_list, get_dir_name


def prepare_data(
    target_dir: Path,
    remove_array: list[str] | None = None,
    wavelength_range: list[float] | None = None,
    exclude_data: list[str] | None = None,
) -> oim.oimData:
    """Prepare the data for the model fitting.

    Parameters
    ----------
    target_dir : Path
        The directory containing the data to fit.
    remove_array : list of str, optional
        The arrays to remove from the data, by default None.
    wavelength_range : list of float, optional
        The wavelength range to consider, by default None.
    exclude_data : list of str, optional
        The data to exclude from the fitting, by default None.

    Returns
    -------
    oim.oimData
        The data to fit.
    """
    fits_files = list((target_dir / "good").glob("*.fits"))
    if exclude_data is not None:
        for exclusion_condition in exclude_data:
            if "band" in exclusion_condition:
                fits_files = [
                    fits_file
                    for fits_file in fits_files
                    if exclusion_condition.split("band")[0].upper()
                    not in fits_file.name.split("_")
                ]
            else:
                fits_files = [
                    fits_file
                    for fits_file in fits_files
                    if exclusion_condition.lower() not in fits_file.name.lower()
                ]

    data = oim.oimData()

    filters = []
    if remove_array is not None:
        filters.append(oim.oimRemoveArrayFilter(targets="all", arr=remove_array))

    if wavelength_range is not None:
        filters.append(
            oim.oimWavelengthRangeFilter(targets="all", wlRange=wavelength_range)
        )

    data.setFilter(oim.oimDataFilter(filters))
    return data


def run_fit(
    data: oim.oimData,
    model: oim.oimModel,
    nsteps: int | float,
    ndiscard: int | float,
    nwalkers: int | float,
    data_to_plot: list[str] = ["VIS2DATA", "T3PHI"],
) -> None:
    """Run the model fitting.

    Parameters
    ----------
    data : oim.oimData
        The data to fit.
    model : oim.oimModel
        The model to fit.
    nsteps : int | float
        The number of steps to run the model fitting.
    ndiscard : int | float
        The number of steps to discard.
    nwalkers : int | float
        The number of walkers to use.
    data_to_plot : list of str, optional
        The data to plot, by default ["VIS2DATA", "T3PHI"].
    """
    sim = oim.oimSimulator(data=data, model=model)
    sim.compute(computeChi2=True, computeSimulatedData=True)
    fig0, ax0 = sim.plot(data_to_plot)
    print(f"Chi2r = {sim.chi2r}")

    fit = oim.oimFitterEmcee(data, model, nwalkers=int(nwalkers))
    fit.prepare(init="random")
    fit.run(nsteps=int(nsteps), progress=True)

    figWalkers, axeWalkers = fit.walkersPlot()
    figCorner, axeCorner = fit.cornerPlot(discard=int(ndiscard))
    fig0, ax0 = sim.plot(data_to_plot)

    keys = list(model.getParameters(free=True).keys())
    best, err_l, err_u, err = fit.getResults(mode="best", discard=int(ndiscard))
    plt.show()


if __name__ == "__main__":
    survey_dir = Path().home() / "Data" / "survey"

    # NOTE: Model creation
    g = oim.oimEGauss(fwhm=20)
    g.params["elong"].set(min=1, max=100)
    g.params["fwhm"].set(min=0, max=100)
    model = oim.oimModel(g)

    # NOTE: Model fit
    for target in get_target_list(survey_dir / "MATISSE data overview.xlsx"):
        data = prepare_data(
            survey_dir / get_dir_name(target),
            remove_array=["OI_FLUX", "OI_T3"],
            exclude_data=["nband", "chopped", "k2n"],
        )
        run_fit(
            data,
            model,
            nsteps=1e3,
            ndiscard=2.5e2,
            nwalkers=50,
            data_to_plot=["VIS2DATA"],
        )
        breakpoint()
