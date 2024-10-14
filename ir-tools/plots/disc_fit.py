from pathlib import Path

import astropy.units as u
import numpy as np
from matadrs.utils.plot import Plotter
from ppdmod.analysis import restore_from_fits
from ppdmod.data import set_data
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_component_mosaic, \
    plot_fit, plot_interferometric_observables, \
    plot_overview, plot_sed, plot_intermediate_products, \
    plot_corner, plot_chains, get_best_fit
from ppdmod.fitting import compute_observables, compute_observable_chi_sq

from ..tables import best_fit_parameters

def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/disc_fits/2024-10-07/results_model_19:19:59")
    plot_dir, assets_dir = path / "plots", path / "assets"
    plot_dir.mkdir(exist_ok=True, parents=True)
    assets_dir.mkdir(exist_ok=True, parents=True)

    fits_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    wavelengths = {"hband": [1.7] * u.um, "kband": [2.15] * u.um,
                   "lband": np.linspace(3.3, 3.8, 5) * u.um,
                   "mband": np.linspace(4.6, 4.9, 3) * u.um,
                   "nband": np.linspace(8, 13, 35) * u.um,
                   }

    OPTIONS.model.output = "non-normed"
    fits_files = list((fits_dir).glob("*fits"))
    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)

    # lband_data = list(fits_dir.glob("*HAW*"))
    # plot_fits = Plotter(lband_data[:3],
    #                     plot_name="lband_data.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=False)
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(lband_data[4:],
    #                     plot_name="lband_data_continuation.pdf", save_dir=plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=False)
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(list(fits_dir.glob("*AQU*")),
    #                     plot_name="nband_data.pdf", save_dir=plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=True)
    # plot_fits.plot(**plot_kwargs)

    dim = 1024
    wavelength = np.concatenate((wavelengths["hband"], wavelengths["kband"],
                                 wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis"])
    component_labels, components, sampler = restore_from_fits(path)
    theta, uncertainties = get_best_fit(sampler)
    rchi_sq = compute_observable_chi_sq(
            *compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    labels = np.load(path / "labels.npy")
    units = np.load(path / "units.npy", allow_pickle=True)
    plot_corner(sampler, labels, units, savefig=plot_dir / "corner.pdf")
    # plot_chains(sampler, labels, units, savefig=fit_plot_dir / "chains.pdf")

    plot_overview(savefig=plot_dir / "overview.pdf")
    # best_fit_parameters(labels, units, theta, save_as_csv=False, savefig=fit_plot_dir / "disc.pdf")
    best_fit_parameters(labels, units, theta, uncertainties,
                        save_as_csv=True, savefig=assets_dir / "disc.csv")
    plot_fit(components=components, savefig=plot_dir / "disc.pdf")

    plot_component_mosaic(components, dim, 0.1, norm=0.2, savefig=plot_dir / "models.pdf", zoom=8)
    # plot_sed([1, 13.5] * u.um, components, scaling="nu", save_dir=fit_plot_dir)
    # plot_sed([1, 13.5] * u.um, components, scaling=None, save_dir=fit_plot_dir)
    plot_intermediate_products(dim, wavelength, components,
                               component_labels, save_dir=plot_dir)
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)

