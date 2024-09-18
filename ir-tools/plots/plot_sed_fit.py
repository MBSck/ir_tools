from pathlib import Path

import astropy.units as u
import numpy as np
from matadrs.utils.plot import Plotter
from ppdmod.analysis import restore_from_fits
from ppdmod.fitting import compute_sed, compute_chi_sq, get_priors
from ppdmod.data import set_data, get_all_wavelengths
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_component_mosaic, \
    plot_fit, plot_overview, plot_sed, plot_corner, plot_chains


def ptform():
    pass


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/results_model_14:19:29")
    fit_plot_dir = path / "fits"
    fit_plot_dir.mkdir(exist_ok=True, parents=True)
    data_plot_dir = path / "data"
    data_plot_dir.mkdir(exist_ok=True, parents=True)

    OPTIONS.model.output = "non-normed"
    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    fits_file = [data_dir / "sed_fit" / "hd142527_average_sed.fits"]

    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)

    # plot_fits = Plotter(fits_file, plot_name="nband_data.pdf", save_dir=data_plot_dir)
    # plot_fits.add_uv(uv_extent=150).add_flux().plot(**plot_kwargs)

    data = set_data(fits_file, wavelengths="all", fit_data=["flux"])
    all_wavelengths = get_all_wavelengths()

    dim = 1024
    component_labels, components, sampler = restore_from_fits(path)
    model_flux = compute_sed(components, all_wavelengths)
    chi_sq = compute_chi_sq(
        data.flux.value, data.flux.err, model_flux, func_method="default")
    nfree_params = len(components[0].get_params(free=True))
    rchi_sq = chi_sq / (data.flux.value.size - nfree_params)
    print(f"rchi_sq: {rchi_sq:.2f}")
    breakpoint()

    # TODO: Save these as well in the fits file? Maybe even the sampler?
    # TODO: Make the labels work again
    labels, units = np.load(path / "labels.npy"), np.load(path / "units.npy", allow_pickle=True)
    plot_corner(sampler, labels, units, savefig=fit_plot_dir / "corner.pdf")
    # plot_chains(sampler, labels, units, savefig=fit_plot_dir / "chains.pdf")

    # plot_overview(savefig=data_plot_dir / "data_overview.pdf")
    # plot_fit(components=components, savefig=fit_plot_dir / "fit_results.pdf")
    # plot_component_mosaic(components, dim, 0.1, norm=0.2,
    #                       savefig=fit_plot_dir / "mosaic_model.pdf", zoom=8)
    plot_sed([7.9, 13.9] * u.um, components, scaling="nu", save_dir=fit_plot_dir)
    plot_sed([7.9, 13.9] * u.um, components, scaling=None, save_dir=fit_plot_dir)
    breakpoint()
    # plot_intermediate_products(dim, wavelength, components,
    #                            component_labels, save_dir=fit_plot_dir)
    # plot_interferometric_observables([1, 13.5]*u.um, components,
    #                                  component_labels, save_dir=fit_plot_dir)

