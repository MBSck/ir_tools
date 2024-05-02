from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    uv_files = list(Path("/Users/scheuck/Data/reduced_data/hd142527/uv").glob("*.fits"))
    uv_plotter = Plotter(uv_files, plot_name="uv_hd142527.pdf")
    uv_plotter.add_uv(color_grouping="file", make_tracks=True).plot(save=True)
