from pathlib import Path

import pandas as pd
from astroquery.simbad import Simbad

from .sort_data import get_sources

Simbad.add_votable_fields("sptype", "ra", "dec")

if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    excel_file = data_dir / "survey" / "MATISSE data overview.xlsx"
    sources = get_sources(excel_file)

    results = {"source": [], "Spectral Type": [], "RA": [], "DEC": []}
    for source in sources:
        result = Simbad.query_object(source)
        results["source"].append(source)
        results["RA"].append(result["RA"].data[0].replace(" ", ":"))
        results["DEC"].append(result["DEC"].data[0].replace(" ", ":"))
        results["Spectral Type"].append(result["SP_TYPE"].data[0])

    pd.DataFrame(results).to_excel(data_dir / "survey" / "simbad_results.xlsx")
