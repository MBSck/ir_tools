from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    fitting_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/fitting_data")
    hd142527_dir = Path("/Users/scheuck/Data/fitting_data/hd142527")
    # directory = Path("/Users/scheuck/Data/reduced_data/hd142666/gravity/fits/calibrated")
    # for fits_file in directory.glob("*fits"):
    #     plotter = Plotter(fits_file, save_path=fits_file.parent)
    #     plotter.add_flux().plot(error=True, save=True, margin=0.3)

    ut_files = list(hd142527_dir.glob("*03-23*AQU*.fits"))
    uv_plotter = Plotter(ut_files, plot_name="ut.pdf")
    uv_plotter.add_flux().add_vis2().add_t3(unwrap=True).plot(
            save=True, error=True, dpi=100, size=3000)
    # uv_plotter.add_uv(color_grouping="file",
    #                   make_tracks=True).plot(save=True)
