from pathlib import Path

from matadrs.utils.plot import Plotter


if __name__ == "__main__":
    directory = Path("/Users/scheuck/Data/reduced_data/hd142666/gravity/fits/calibrated")
    for fits_file in list(directory.glob("*.fits")):
        plotter = Plotter(fits_file, save_path=directory)
        plotter.add_flux().add_uv().add_vis().plot(error=True, save=True)
