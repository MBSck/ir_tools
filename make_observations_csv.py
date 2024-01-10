from pathlib import Path
from typing import Dict

import pandas as pd
from matadrs.utils.readout import ReadoutFits


STATIONS_TO_NAME = {"A0-B2-C1-D0": "small", "D0-G2-J3-K0": "medium",
                    "A0-G1-J2-J3": "large", "UT1-UT2-UT3-UT4": "UTs"}
NAME_TO_STATIONS = {v: k for k, v in STATIONS_TO_NAME.items()}


def set_data(path: Path) -> Dict:
    """Sets the data for the pandas dataframe."""
    data = {"Instrument": [], "Date": [], "Seeing": [], "tau_0": [],
            "Stations": []}
    for readout in list(map(ReadoutFits, path.glob("*.fits"))):
        data["Instrument"].append(readout.instrument.upper())
        data["Date"].append(" ".join(readout.tpl_start.split("T"))[:-3])
        data["Seeing"].append(round(readout.seeing, 1))
        data["tau_0"].append(round(readout.tau0*1e3, 1))
        data["Stations"].append(readout.stations)
        # data["Configuration"].append(readout.configuration)
        # data["Calibrator"].append(readout.calibrator)
    return data


def create_dataframe(data: Dict) -> pd.DataFrame:
    """Creates the pandas dataframe."""
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Instrument'])
    return df


if __name__ == "__main__":
    fitting_dir = Path("/Users/scheuck/Data/reduced_data/hd142666/fitting_data")
    data = set_data(fitting_dir)
    print(create_dataframe(data))
    # df.to_csv("observations.csv", index=False)
