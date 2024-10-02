from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    files = list(Path("/Users/scheuck/Data/fitting_data/hd142527").glob("*.fits"))
    plotter = Plotter(files, plot_name="uv_uts.pdf")
    plotter.filter(["array", "band"], ["uts", "nband"]).sort(by="date")
    plotter.add_uv(make_tracks=True).plot(color_by="file", save=True)
