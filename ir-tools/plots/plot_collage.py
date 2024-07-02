from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/fitting_data/hd142527")
    files_N, files_L = path.glob("*AQUARIUS*"), path.glob("*HAWAII*")

    files = map(lambda files: map(lambda x: path / x, files), [files_L, files_N])
    files = list(Path("/Users/scheuck/Data/reduced_data/hd142527/matisse/nband").glob("*07-01*"))
    for fits_file in files:
        unwrap = "HAWAII" not in fits_file.name
        plot_fits = Plotter(fits_file)
        plot_fits.add_flux()
        plot_fits.add_vis(corr_flux=True)
        # plot_fits.add_diff_phases(unwrap=unwrap)
        plot_fits.add_t3(unwrap=unwrap)
        plot_fits.plot(legend_format="short",
                       error=True, subplots=False,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True, save=True)
