from pathlib import Path

import astropy.units as u
import numpy as np
from ppdmod.analysis import restore_from_fits
from ppdmod.fitting import compute_sed_chi_sq, get_best_fit
from ppdmod.data import set_data
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_overview, plot_sed, plot_corner

from ..tables import best_fit_parameters


def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-10-08/")
    # dir_name = "averaged"
    # dir_name =  "only_low"
    dir_name =  "downsampled"
    # dir_name =  "only_high"

    path /= dir_name
    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    OPTIONS.model.output = "non-normed"
    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")

    # wavelength_range = None
    wavelength_range = [8., 13.1] * u.um
    data = set_data(list((data_dir / "sed_fit" / dir_name).glob("*fits")),
                    wavelengths="all", wavelength_range=wavelength_range,
                    fit_data=["flux"])

    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)

    component_labels, components, sampler = restore_from_fits(path, "sed.fits")
    labels = np.load(path / "labels.npy").tolist()
    units = np.load(path / "units.npy", allow_pickle=True)

    theta, uncertainties = get_best_fit(sampler)
    print(f"Best fit parameters:\n{np.array(theta)}")

    indices = list(map(labels.index, filter(lambda x: "weight" in x and "pah" not in x, labels)))
    print(f"Normed sum: {np.array(theta)[indices].sum()}")

    silicate_weights = np.array(theta)[indices[1:]]
    np.save(assets_dir / "silicate_labels_and_weights.npy",
            [np.array(labels)[indices[1:]], silicate_weights / silicate_weights.sum() * 1e2])

    rchi_sq = compute_sed_chi_sq(
        components[0].compute_flux(OPTIONS.fit.wavelengths), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    dim = 1024
    plot_corner(sampler, labels, units, savefig=plot_dir / "corner.pdf")
    plot_overview(savefig=plot_dir / "data_overview.pdf")
    best_fit_parameters(labels, units, theta, uncertainties,
                        save_as_csv=True, savefig=assets_dir / "sed.csv")

    plot_sed([7.9, 13.15] * u.um, components, scaling="nu", save_dir=plot_dir)
    plot_sed([7.9, 13.15] * u.um, components, scaling=None, save_dir=plot_dir)
