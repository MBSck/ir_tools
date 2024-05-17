from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/fitting_data/hd142527")
    files_N, files_L = path.glob("*AQUARIUS*"), path.glob("*HAWAII*")

    files = map(lambda files: map(lambda x: path / x, files), [files_L, files_N])
    for fits_file in files:
        unwrap = "HAWAII" not in fits_file[0].name
        plot_fits = Plotter(fits_file)
        plot_fits.add_vis(corr_flux=True)
        plot_fits.add_diff_phases(unwrap=unwrap)
        plot_fits.add_cphases(unwrap=unwrap)
        plot_fits.plot(savefig=f"collage_{'N_band' if unwrap else 'L_band'}",
                       legend_format="short,"
                       error=True, subplots=False,
                       margin=0.3, legend_size="medium",
                       sharex=True, share_legend=True)
