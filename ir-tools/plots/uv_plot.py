from pathlib import Path

import matplotlib.pyplot as plt
from matadrs.utils.plot import Plotter

data_path = Path("/Users/scheuck/Data/reduced_data/hd142666/matisse")
files = list(data_path.glob("*AQUARIUS*"))
plot_fits = Plotter(files)
plot_fits.add_uv().plot(save=True, format="png")
