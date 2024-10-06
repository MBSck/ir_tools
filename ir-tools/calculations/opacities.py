from pathlib import Path

import numpy as np
from ppdmod.utils import get_opacity


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-10-04/")
    dir_name = "averaged"
    # dir_name =  "only_low"
    # dir_name =  "downsampled"
    # dir_name =  "only_high"

    labels, weights = np.load(path / dir_name / "fits" / "silicate_labels_and_weights.npy")
    weights = weights.astype(float) / 100

    data_dir = Path("/Users/scheuck/Data/opacities")
    names = ["pyroxene", "enstatite", "forsterite", "silica", "olivine"]

    wl, opacity = get_opacity(data_dir, weights, names, "grf")
    np.save("hd142527_combined_silicate_opacities.npy", np.array([wl, opacity]))
