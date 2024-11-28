from pathlib import Path
from typing import List

import pandas as pd
from astroquery.simbad import Simbad

from .sort_data import get_sources

Simbad.add_votable_fields("sptype", "ra", "dec")

FIELDS_TO_KEY = {
    "Spectral Type": "SP_TYPE",
    "RA": "RA",
    "DEC": "DEC",
    "source": "source",
}


def query(sources: List[str]) -> pd.DataFrame:
    """Queries simbad for information on the sources."""
    results = {key: [] for key in FIELDS_TO_KEY}
    for source in sources:
        result = Simbad.query_object(source)
        for field, key in FIELDS_TO_KEY.items():
            if field == "source":
                results[field].append(source)
            else:
                data = result[key].data[0]
                if key in ["RA", "DEC"]:
                    data = data.replace(" ", ":")

                results[field].append(data)

    return pd.DataFrame(results)


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    excel_file = data_dir / "survey" / "MATISSE data overview.xlsx"
    df = query(get_sources(excel_file))
    df.to_excel(data_dir / "survey" / "simbad_results.xlsx")
