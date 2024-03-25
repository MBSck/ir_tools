from pathlib import Path

from matadrs.utils.plot import Plotter
from matadrs.utils.options import OPTIONS


if __name__ == "__main__":
    path200 = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse/2.0.0/mat_tools/nband")
    path176 = Path("/Users/scheuck/Data/reduced_data/jozsef_reductions/hd142666")
    file2019 = path176 / "HD_142666_2019-05-14T05_28_03_N_TARGET_FINALCAL_INT.fits"
    file2022_21 = path200 / "hd_142666_2022-04-21T07_18_22:2022-04-21T06_47_05_AQUARIUS_FINAL_TARGET_INT.fits"
    file2022_23 = path200 / "hd_142666_2022-04-23T03_05_25:2022-04-23T02_28_06_AQUARIUS_FINAL_TARGET_INT.fits"

    OPTIONS["plot.color.colormap"] = "default"
    for name, file in zip(["2019", "202221", "202223"],
                          [file2019, file2022_21, file2022_23]):
        plotter = Plotter(file, plot_name=f"{name}.pdf")
        plotter.add_vis().plot(error=True, save=True)
