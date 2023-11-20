from pathlib import Path

import astropy.units as u
from ppdmod.data import set_data
from ppdmod.plot import plot_overview
from ppdmod.options import OPTIONS


if __name__ == "__main__":
    OPTIONS["fit.data"] = ["vis2", "t3phi"]
    OPTIONS["fit.wavelengths"] = [1.6, 2.25, 3.5, 8, 8.5,
                                  9, 10, 11.3, 12.5]*u.um
    OPTIONS["data.binning.window"] = 0.2
    fits_files = list(Path.cwd().glob("*.fits"))
    set_data(fits_files)
    plot_overview(["vis", "t3phi"], title="Data Overview",
                  savefig="data_overview.pdf")
