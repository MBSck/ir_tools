from pathlib import Path
from typing import List

import numpy as np
from pylatex import Document, NoEscape, Section, Tabular


def survey_results(
    sources: List[str],
    parameter_labels: List[str],
    results: List[List[float]],
    chi_sqs: List[float],
    model_save_dir: Path,
) -> None:
    table = " ".join(["l"] + ["c" for _ in range(parameter_labels.size + 1)])
    labels = ["Object"] + [NoEscape(r"$\chi^2r$")] + parameter_labels.tolist()

    doc = Document()
    with doc.create(Section(" ".join(model_save_dir.name.split("_")).title())):
        with doc.create(Tabular(table)) as table:
            table.add_row(labels)
            table.add_hline()

            for source, chi_sq, result in zip(sources, chi_sqs, results):
                result = [NoEscape(f"${r[0]:.2f}^{{+{r[1]:.2f}}}_{{-{r[2]:.2f}}}$") for r in result]
                table.add_row([source] + [f"{chi_sq:.2f}"] + result)

            table.add_hline()

    doc.generate_pdf(model_save_dir / "results", clean_tex=False)


if __name__ == "__main__":
    survey_result_dir = Path().home() / "Data" / "survey" / "results"
    save_dir = survey_result_dir / "model_pt_eg"
    sources = np.load(save_dir / "sources.npy")
    parameter_labels = np.load(save_dir / "labels.npy")
    results = np.load(save_dir / "results.npy")
    chi_sqs = np.load(save_dir / "chi_sqs.npy")
    survey_results(sources, parameter_labels, results, chi_sqs, save_dir)
