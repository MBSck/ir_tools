from pathlib import Path

import numpy as np
from pylatex import Document, LongTable, MultiColumn, Section

if __name__ == "__main__":
    path = Path().home() / "Data" / "model_results" / "sed_fits" / "2024-10-22"
    labels, weights = np.load(
        path / "downsampled" / "assets" / "silicate_labels_and_weights.npy"
    )
    weights = np.round(weights.astype(float), 2)

    labels = ["Pyroxene", "Enstatite", "Forsterite", "Silica", "Olivine"]
    boekel_weights = np.array([0, 73.2, 0.6, 14.2, 8.6, 0, 2.4, 1.0, 0, 0])
    juhasz_weights = np.array([0, 22.98, 3.23, 6.27, 5.7, 0, 6.56, 4.55, 37.14, 13.57])

    labels = [""] + [MultiColumn(2, align="c |", data=label) for label in labels]
    table = " ".join(
        ["c" if index % 2 == 0 else "c |" for index in range(1, weights.size + 2)]
    )

    doc = Document()
    with doc.create(Section("Opacity weights")):
        with doc.create(LongTable(table)) as table:
            table.add_row(labels)
            table.add_hline()
            table.add_row([""] + ["Small", "Large"] * (len(labels) - 1))
            table.add_hline()
            table.add_row(["Mine"] + weights.tolist())
            table.add_row(["Juhasz et al. (2009)"] + juhasz_weights.tolist())
            table.add_row(["Boekel et al. (2005)"] + boekel_weights.tolist())
            table.add_hline()

    doc.generate_pdf("hd142527_opacity_fits", clean_tex=False)
