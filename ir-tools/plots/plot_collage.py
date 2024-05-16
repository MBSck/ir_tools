from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from matadrs.utils.plot import Plotter

data_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
files_N, files_L = data_path.glob("*AQUARIUS*"), data_path.glob("*HAWAII*")

files_N = [data_path / f for f in files_N]
files_L = [data_path / f for f in files_L]
files_bands = [files_L, files_N]

legend_format = "short"
for files in files_bands:
    unwrap = "HAWAII" not in files[0].name
    plot_fits = Plotter(files)
    plot_fits.add_vis(corr_flux=True, legend_format=legend_format)
    plot_fits.add_diff_phases(unwrap=unwrap, legend_format=legend_format)
    plot_fits.add_cphases(unwrap=unwrap, legend_format=legend_format)
    plot_fits.plot(savefig=f"collage_{'N_band' if unwrap else 'L_band'}",
                   error=True, subplots=False,
                   margin=0.3, legend_size="medium",
                   sharex=True, share_legend=True)
