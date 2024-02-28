from pathlib import Path

from matadrs.utils.plot import Plotter
from matadrs.utils.tools import get_fits_by_tag


if __name__ == "__main__":
    fitting_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/fitting_data")
    # directory = Path("/Users/scheuck/Data/reduced_data/hd142666/gravity/fits/calibrated")
    # for fits_file in directory.glob("*fits"):
    #     plotter = Plotter(fits_file, save_path=fits_file.parent)
    #     plotter.add_flux().plot(error=True, save=True, margin=0.3)

    ut_files = list(fitting_dir.glob("*AQU*.fits"))
    uv_plotter = Plotter(ut_files, plot_name="uv_uts.pdf")
    uv_plotter.add_uv(color_grouping="file",
                      make_tracks=True).plot(save=True)
    # uv_plotter = Plotter(list(ut_files), plot_name="uv_mosaic.pdf")
    # uv_plotter.add_mosaic().plot(save=True, error=True)
    # uv_plotter.add_uv(color_grouping="file", make_tracks=True).plot(save=True)

    # directory = Path("/data/beegfs/astro-storage/groups/matisse/scheuck/data/matisse/GTO/hd142527/product/test/2021-03-08/reduced/coherent/lband/mat_raw_estimates.2021-03-08T07_25_46.HAWAII-2RG.rb")
    # fits_files = get_fits_by_tag(directory, "RAW_INT")
    # for fits_file in fits_files:
    #     plot_fits = Plotter(fits_file, save_path=directory)
    #     unwrap = True if "AQUARIUS" in str(fits_file) else False
    #     plot_fits.add_t3(unwrap=unwrap).add_vis().add_vis2()
    #     plot_fits.plot(save=True, error=True)
