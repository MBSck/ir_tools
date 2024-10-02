from pathlib import Path
from typing import List

import numpy as np
from ppdmod.analysis import restore_from_fits
from ppdmod.component import Component
from ppdmod.plot import format_labels
from pylatex import Document, Section, Tabular
from pylatex.utils import NoEscape


def make_fit_parameters_table(components: List[Component], savefig: Path) -> None:
    """Make a (.pdf) file containing a table of the fit parameters."""
    if len(components) == 1:
        params = components[0].get_params(free=True)
    else:
        params = {}
        # TODO: This needs to be finished
        # for index, component in enumerate(components, start=1):
        #     test = component.get_params(free=True)

    labels = format_labels(list(params.keys()))
    values = np.round(list(map(lambda x: x.value, params.values())), 2)
    units = list(map(lambda x: x.unit, params.values()))

    doc = Document()
    with doc.create(Section("Fit Parameters")):
        with doc.create(Tabular("c | c | c")) as table:
            table.add_hline()
            table.add_row(("Parameter", "Unit", "Value"))
            table.add_hline()

            for label, unit, value in zip(labels, units, values):
                table.add_row((NoEscape(label), unit, value))

            table.add_hline()

    doc.generate_pdf(savefig, clean_tex=False)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/model_results/sed_fits/2024-10-02/downsampled")
    component_labels, components = restore_from_fits(path)
    make_fit_parameters_table(components)

