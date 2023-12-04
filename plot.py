from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    directory = Path("/Users/scheuck/Data/reduced_data/hd142666/gravity/fits/calibrated")
    for fits_file in directory.glob("*fits"):
        plotter = Plotter(fits_file, save_path=fits_file.parent)
        plotter.add_flux().plot(error=True, save=True, margin=0.3)
