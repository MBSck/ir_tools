from pathlib import Path

import astropy.units as u
import numpy as np
from matadrs.utils.plot import Plotter
from ppdmod.analysis import save_fits, restore_from_fits
from ppdmod.fitting import compute_sed_chi_sq, get_best_fit
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_component_mosaic, \
    plot_fit, plot_overview, plot_sed, plot_corner, plot_chains


def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-09-23/only_low")
    fit_plot_dir = path / "fits"
    fit_plot_dir.mkdir(exist_ok=True, parents=True)
    data_plot_dir = path / "data"
    data_plot_dir.mkdir(exist_ok=True, parents=True)

    OPTIONS.model.output = "non-normed"
    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    # fits_dir = data_dir / "sed_fit"
    # fits_dir = data_dir / "sed_fit" / "downsampled"
    # fits_dir = data_dir / "sed_fit" / "only_high"
    fits_dir = data_dir / "sed_fit" / "only_low"

    # wavelength_range = None
    wavelength_range = [8., 13.1] * u.um
    data = set_data(list(fits_dir.glob("*fits")), wavelengths="all",
                wavelength_range=wavelength_range, fit_data=["flux"])
    wavelengths = get_all_wavelengths()

    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)

    # plot_fits = Plotter(fits_file, plot_name="nband_data.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().plot(**plot_kwargs)

    labels, units = np.load(path / "labels.npy").tolist(), np.load(path / "units.npy", allow_pickle=True)
    component_labels, components, sampler = restore_from_fits(path)

    theta, uncertainties = get_best_fit(sampler)
    print(f"Best fit parameters:\n{np.array(theta)}")

    indices = list(map(labels.index, filter(lambda x: "weight" in x and "pah" not in x, labels)))
    print(f"Normed sum: {np.array(theta)[indices].sum()}")

    rchi_sq = compute_sed_chi_sq(
        components[0].compute_flux(get_all_wavelengths()), reduced=True)
    print(f"rchi_sq: {rchi_sq:.2f}")

    # TODO: Save these as well in the fits file? Maybe even the sampler?
    # plot_corner(sampler, labels, units, savefig=fit_plot_dir / "corner.pdf")
    # plot_chains(sampler, labels, units, savefig=fit_plot_dir / "chains.pdf")

    dim = 1024
    plot_overview(savefig=data_plot_dir / "data_overview.pdf")
    # plot_fit(components=components, savefig=fit_plot_dir / "fit_results.pdf")
    # plot_component_mosaic(components, dim, 0.1, norm=0.2,
    #                       savefig=fit_plot_dir / "mosaic_model.pdf", zoom=8)
    plot_sed([7.9, 13.15] * u.um, components, scaling="nu", save_dir=fit_plot_dir)
    plot_sed([7.9, 13.15] * u.um, components, scaling=None, save_dir=fit_plot_dir)
    # plot_intermediate_products(dim, wavelength, components,
    #                            component_labels, save_dir=fit_plot_dir)
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)

