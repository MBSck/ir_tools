from pathlib import Path

import astropy.units as u
import numpy as np
from matadrs.utils.plot import Plotter
from ppdmod.analysis import restore_from_fits
from ppdmod.data import set_data
from ppdmod.options import OPTIONS
from ppdmod.plot import plot_component_mosaic, plot_corner, \
    plot_chains, plot_fit, plot_observables, plot_overview
from ppdmod.fitting import get_best_fit


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/2024-07-06/results_model_23:16:34")
    fit_plot_dir = path / "fits"
    fit_plot_dir.mkdir(exist_ok=True, parents=True)
    data_plot_dir = path / "data"
    data_plot_dir.mkdir(exist_ok=True, parents=True)

    data_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    wavelengths = {"hband": [1.6]*u.um, "kband": [2.25]*u.um,
                   "lband": [3.2]*u.um, "mband": [4.7]*u.um,
                   "nband": [8., 9., 10., 11.3, 12.5]*u.um}

    OPTIONS.model.output = "non-normed"
    fits_files = list((data_dir).glob("*fits"))

    plot_kwargs = dict(legend_format="short",
                       error=True, subplots=True,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)
    plot_fits = Plotter(list(data_dir.glob("*HAW*")),
                        plot_name="lband_data.pdf", save_dir=data_plot_dir)
    plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=False)
    plot_fits.plot(**plot_kwargs)

    plot_fits = Plotter(list(data_dir.glob("*AQU*")),
                        plot_name="nband_data.pdf", save_dir=data_plot_dir)
    plot_fits.add_uv(uv_extent=150).add_flux().add_vis(corr_flux=True).add_t3(unwrap=True)
    plot_fits.plot(**plot_kwargs)
    breakpoint()

    wavelength = np.concatenate((wavelengths["lband"], wavelengths["mband"], wavelengths["nband"]))
    data = set_data(fits_files, wavelengths=wavelength, fit_data=["flux", "vis", "t3"])
    component_labels, components, sampler = restore_from_fits(path)
    # plot_component_mosaic(components, 4096, 0.1, [3.5, 8, 9, 10, 11.3, 12], norm=0.2,
    #                       savefig=fit_plot_dir / "mosaic_model.pdf", zoom=20)
    # plot_fit(components=components, savefig=fit_plot_dir / "fit_results.pdf")
    # plot_observables([1, 12]*u.um, components, component_labels, save_dir=fit_plot_dir)

    # TODO: Check if the outputs are the best
    # get_best_fit(components, data, path / "post_fit_model.fits")

    # TODO: Find way to save labels or derive them as well
    # plot_corner(sampler, labels, **fit_params,
    #             savefig=plot_dir / "chains.pdf")
    # plot_chains(sampler, labels, **fit_params,
    #             savefig=plot_dir / "")

    # plot_overview(savefig=data_plot_dir / "data_overview.pdf")
