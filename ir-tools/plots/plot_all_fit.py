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
    plot_corner, plot_chains
from ppdmod.fitting import compute_observables, compute_observable_chi_sq


def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/2024-08-22/results_model_14:42:00")
    fit_plot_dir = path / "fits"
    fit_plot_dir.mkdir(exist_ok=True, parents=True)
    data_plot_dir = path / "data"
    data_plot_dir.mkdir(exist_ok=True, parents=True)

    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    wavelengths = {"hband": [1.7]*u.um, "kband": [2.15]*u.um,
                   "lband": np.linspace(3.3, 3.8, 5) * u.um,
                   "mband": np.linspace(4.6, 4.9, 3) * u.um,
                   "nband": np.linspace(8, 13, 35) * u.um,
                   }

    OPTIONS.model.output = "non-normed"
    fits_files = list((data_dir).glob("*fits"))

    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)

    # lband_data = list(data_dir.glob("*HAW*"))
    # plot_fits = Plotter(lband_data[:3],
    #                     plot_name="lband_data.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=False)
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(lband_data[4:],
    #                     plot_name="lband_data_continuation.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=False)
    # plot_fits.plot(**plot_kwargs)
    #
    # plot_fits = Plotter(list(data_dir.glob("*AQU*")),
    #                     plot_name="nband_data.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=True)
    # plot_fits.plot(**plot_kwargs)

    dim = 1024
    wavelength = np.concatenate((wavelengths["hband"], wavelengths["kband"],
                                 wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis"])
    component_labels, components, sampler = restore_from_fits(path)
    rchi_sq = compute_observable_chi_sq(
            *compute_observables(components), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    labels, units = np.load(path / "labels.npy"), np.load(path / "units.npy", allow_pickle=True)
    plot_corner(sampler, labels, units, savefig=fit_plot_dir / "corner.pdf")
    # plot_chains(sampler, labels, units, savefig=fit_plot_dir / "chains.pdf")

    plot_overview(savefig=data_plot_dir / "data_overview.pdf")
    plot_fit(components=components, savefig=fit_plot_dir / "fit_results.pdf")
    plot_component_mosaic(components, dim, 0.1, norm=0.2,
                          savefig=fit_plot_dir / "mosaic_model.pdf", zoom=8)
    plot_sed([1, 13.5] * u.um, components, scaling="nu", save_dir=fit_plot_dir)
    plot_sed([1, 13.5] * u.um, components, scaling=None, save_dir=fit_plot_dir)
    plot_intermediate_products(dim, wavelength, components,
                               component_labels, save_dir=fit_plot_dir)
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)

