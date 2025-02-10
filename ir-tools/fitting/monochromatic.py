import signal
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import oimodeler as oim

from .sort_data import get_dir_name, get_source_list
from .tables import survey_results


def save_and_exit(signum, frame):
    print(f"Received signal {signum}. Saving data...")
    np.save(survey_dir / "results" / save_dir / "results.npy", np.array(results))
    np.save(survey_dir / "results" / save_dir / "chi_sqs.npy", np.array(chi_sqs))

    print("Data saved. Exiting...")
    sys.exit(0)


# NOTE: Register the signal handlers
signal.signal(signal.SIGINT, save_and_exit)  # Ctrl+C
signal.signal(signal.SIGTERM, save_and_exit)  # Termination signal


def prepare_data(
    source_dir: Path,
    remove_array: list[str] | None = None,
    wavelength_range: list[float] | None = None,
    exclude_data: list[str] | None = None,
    keep_data: list[str] | None = None,
) -> oim.oimData:
    """Prepare the data for the model fitting.

    Parameters
    ----------
    source_dir : Path
        The directory containing the data to fit.
    remove_array : list of str, optional
        The arrays to remove from the data, by default None.
    wavelength_range : list of float, optional
        The wavelength range to consider, by default None.
    exclude_data : list of str, optional
        The data to exclude from the fitting, by default None.
    keep_data : list of str, optional
        The data to keep in the fitting, by default None.

    Returns
    -------
    oim.oimData
        The data to fit.
    """
    fits_files = list((source_dir / "good").glob("*.fits"))
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

    data = oim.oimData(fits_files)

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
    source: str,
    nsteps: int | float,
    ndiscard: int | float,
    nwalkers: int | float,
    save_dir: Path,
    data_to_plot: list[str] = ["VIS2DATA", "T3PHI"],
) -> Tuple[oim.oimSimulator, oim.oimFitterEmcee]:
    """Run the model fitting.

    Parameters
    ----------
    data : oim.oimData
        The data to fit.
    model : oim.oimModel
        The model to fit.
    source : str
        The source to fit.
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

    fit = oim.oimFitterEmcee(data, model, nwalkers=int(nwalkers))
    fit.prepare(init="random")
    fit.run(nsteps=int(nsteps), progress=True)

    try:
        fit.cornerPlot(discard=int(ndiscard))
        plt.savefig(save_dir / f"{source}_corner.png")
        plt.close()
    except AssertionError:
        pass

    sim.plot(data_to_plot)
    plt.savefig(save_dir / f"{source}_fit.png")
    plt.close()

    print(f"Chi2r = {sim.chi2r}")
    return sim, fit


if __name__ == "__main__":
    survey_dir = Path().home() / "Data" / "survey"

    g = oim.oimEGauss(f=0.5, fwhm=10)
    g.params["fwhm"].set(min=0, max=40)
    g.params["elong"].set(min=1, max=100)
    g.params["f"].free = False
    model = oim.oimModel(g)

    save_dir = (
        "model_"
        + "_".join([component.shortname for component in model.components]).lower()
    )
    model_dir = survey_dir / "results" / save_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    parameter_labels = list(model.getParameters(free=True).keys())
    sources = get_source_list(survey_dir / "MATISSE data overview.xlsx")
    np.save(model_dir / "labels.npy", np.array(parameter_labels))
    np.save(model_dir / "sources.npy", np.array(sources))

    chi_sqs, results = [], []
    for source in sources:
        data = prepare_data(
            survey_dir / get_dir_name(source),
            wavelength_range=[3.1e-6, 3.6e-6],
            remove_array=["OI_FLUX", "OI_T3"],
            exclude_data=["nband", "chopped", "k2n"],
        )

        ndiscard = 2.5e2
        sim, fit = run_fit(
            data,
            model,
            source,
            nsteps=1e3,
            ndiscard=ndiscard,
            nwalkers=50,
            data_to_plot=["VIS2DATA"],
            save_dir=survey_dir / "results" / save_dir,
        )

        chi_sqs.append(sim.chi2r)
        results.append(
            np.array(fit.getResults(mode="best", discard=int(ndiscard))[:-1]).T
        )

    np.save(model_dir / "results.npy", np.array(results))
    np.save(model_dir / "chi_sqs.npy", np.array(chi_sqs))
    survey_results(sources, parameter_labels, results, chi_sqs, model_dir)
