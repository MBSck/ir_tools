import pickle
from pathlib import Path

import astropy.units as u
import numpy as np
from dynesty import DynamicNestedSampler
from ppdmod.data import set_data
from ppdmod.fitting import get_best_fit, compute_sed_chi_sq
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_corner, plot_overview, plot_sed

from ..tables import best_fit_parameters


def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-10-28/")
    dir_name = "averaged"

    path /= dir_name
    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    OPTIONS.model.output = "non-normed"
    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")

    # wavelength_range = None
    wavelength_range = [8.0, 13.1] * u.um
    data = set_data(
        list((data_dir / "sed_fit" / dir_name).glob("*fits")),
        wavelengths="all",
        wavelength_range=wavelength_range,
        fit_data=["flux"],
    )

    labels = np.load(path / "labels.npy").tolist()
    units = np.load(path / "units.npy", allow_pickle=True)

    sampler = DynamicNestedSampler.restore(path / "sampler.save")
    theta, uncertainties = get_best_fit(sampler)
    print(f"Best fit parameters:\n{np.array(theta)}")

    indices = list(
        map(labels.index, filter(lambda x: "weight" in x and "pah" not in x, labels))
    )
    print(f"Normed sum: {np.array(theta)[indices].sum()}")

    best_fit_params = np.save(
        assets_dir / "best_fit.npy", [np.array(labels), np.array(theta)]
    )
    silicate_weights = np.array(theta)[indices[1:]]
    np.save(assets_dir / "silicate_labels_and_weights.npy", silicate_weights)

    with open(path / "components.pkl", "rb") as f:
        components = pickle.load(f)

    rchi_sq = compute_sed_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths),
        reduced=True,
        nfree=theta.size,
    )
    print(f"rchi_sq: {rchi_sq:.2f}")

    dim = 1024
    plot_corner(sampler, labels, units, savefig=plot_dir / "corner.pdf")
    plot_overview(savefig=plot_dir / "data_overview.pdf")
    best_fit_parameters(
        labels,
        units,
        theta,
        uncertainties,
        save_as_csv=True,
        savefig=assets_dir / "sed.csv",
    )

    plot_sed([7.9, 13.1] * u.um, components, scaling="nu", save_dir=plot_dir)
    plot_sed([7.9, 13.1] * u.um, components, scaling=None, save_dir=plot_dir)
