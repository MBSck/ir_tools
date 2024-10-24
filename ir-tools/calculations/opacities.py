from pathlib import Path

import numpy as np
from ppdmod.utils import get_opacity

SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
NAMES_DICT = dict(zip(SHORTNAMES, NAMES))

if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "model_results" / "sed_fits" / "2024-10-24"
    dir_name = "averaged"

    labels, weights = np.load(path / dir_name / "assets" / "silicate_labels_and_weights.npy")
    weights = weights.astype(float) / weights.astype(float).sum() * 1e2

    names = []
    for label in [NAMES_DICT[label.split("_")[1]] for label in labels]:
        if label not in names:
            names.append(label)

    wl, opacity = get_opacity(data_dir / "opacities", weights, names, "grf")
    np.save("hd142527_silicate_opacities.npy", np.array([wl, opacity]))
