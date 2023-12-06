from pathlib import Path

from matadrs.utils.plot import plot_data_quality


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/reduced_data/hd142666/")
    plot_data_quality(path, path / "plots")
