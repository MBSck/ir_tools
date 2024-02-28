from pathlib import Path

import astropy.units as u
from matadrs.utils.plot import Plotter
from ppdmod.data import set_data
from ppdmod.plot import plot_model, plot_overview, plot_target
from ppdmod.options import OPTIONS


if __name__ == "__main__":
    fitting_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/fitting_data")
    # model_file = Path("/Users/scheuck/Pictures/Meetings/thomas_and_roy/2024-01-12/nsteps10000_nw100_kband/model.fits")
    # OPTIONS["plot.color.background"] = "black"
    OPTIONS.fit.data = ["flux", "vis2", "t3phi"]
    OPTIONS.fit.wavelengths = [1.6, 2.25, 3.5, 8, 8.5,
                               9, 10, 11.3, 12.5]*u.um
    OPTIONS.data.binning.window = 0.2
    fits_files = list(fitting_dir.glob("*.fits"))
    # set_data(fits_files)
    # # plot_overview(savefig="data_overview.png")
    # target = "hd142527"
    # plot_target(target, wavelength_range=[1, 13]*u.um,
    #             title=target.upper(),
    #             filters=["2MASS", "WISE"], savefig=f"{target}.pdf")
    # plot_model(model_file, data_type="image", wavelength=2.25,
    #            pixel_size=0.1, savefig="image225.png", zoom=5)
    # plot_model(model_file, data_type="image", wavelength=3.5,
    #            pixel_size=0.1, savefig="image35.png", zoom=10)
    # plot_model(model_file, data_type="image", wavelength=10.,
    #            pixel_size=0.1, savefig="image10.png")
    # plot_model(model_file, data_type="image", wavelength=12.5,
    #            pixel_size=0.1, savefig="image125.png")
    # plot_model(model_file, data_type="temperature", savefig="temperature.png")
    # plot_model(model_file, data_type="flux", savefig="flux.png")
    # plot_model(model_file, data_type="brightness", savefig="brightness.png")
    # plot_model(model_file, data_type="density", savefig="density.png")
    # plot_model(model_file, data_type="thickness", savefig="thickness.png")
    # plot_model(model_file, data_type="depth", factor=0.505, savefig="depth.png")
    # path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/1.7.6/mat_tools/lband")
    # file = path / "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_HAWAII-2RG_FINAL_TARGET_INT.fits"
    # plotter = Plotter(file, plot_name="test.pdf")
    # unwrap = True if "nband" in str(file) else False
    # plotter.add_mosaic(unwrap=unwrap).plot(error=True, save=True)
