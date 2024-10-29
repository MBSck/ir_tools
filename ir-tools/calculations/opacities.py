from pathlib import Path

import numpy as np
from ppdmod.utils import get_opacity

SHORTNAMES = ["pyrox", "enst", "forst", "sil", "oliv"]
NAMES = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]


def generate_opacity_file(
    data_dir: Path,
    weights: np.ndarray,
    method: str = "grf",
):
    wl, opacity = get_opacity(data_dir / "opacities", weights, NAMES, method)
    np.save(
        data_dir / "opacities" / f"hd142527_silicate_{method}_opacities.npy",
        np.array([wl, opacity]),
    )


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    path = data_dir / "model_results" / "sed_fits" / "2024-10-24"
    dir_name = "averaged"

    _, weights = np.load(
        path / dir_name / "assets" / "silicate_labels_and_weights.npy"
    )
    weights = weights.astype(float) / weights.astype(float).sum()
    # weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0]) / 1e2
    generate_opacity_file(data_dir, weights, method="grf")
