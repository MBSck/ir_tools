from pathlib import Path
from typing import List, Optional

import pandas as pd
from matadrs.utils.readout import ReadoutFits


def read_to_table(fits_files: List[Path], save_as_csv: Optional[bool] = False,
                  savefig: Optional[Path] = False):
    """Read a FITS file and return the data as an astropy table."""
    data = {"instrument": [], "date": [], "seeing": [], "tau0": [],
            "stations": [], "array": [], "name": [], "ldd": [],
            "time": [], "Comment": []}

    for fits_file in fits_files:
        readout = ReadoutFits(fits_file)
        data["instrument"].append(readout.instrument_mode.upper())
        data["date"].append(readout.date[:-8])
        data["seeing"].append(round(readout.seeing, 1))
        data["tau0"].append(round(readout.tau0, 1))
        data["stations"].append(readout.stations)
        data["array"].append(readout.array)

        cal_name, cal_time, cal_ldd = readout.get_calib_info()
        data["name"].append(cal_name)
        data["ldd"].append(cal_ldd)
        data["time"].append(cal_time)

        comment = ""
        if readout.instrument.lower() == "pionier":
            comment = "H"
        elif readout.instrument.lower() == "gravity":
            comment = "K"
        elif readout.instrument.lower() == "matisse":
            if readout.array == "UTs":
                comment = "LN"
            else:
                comment = "L"

        data["Comment"].append(comment)

    df = pd.DataFrame(data).sort_values(by="date")
    breakpoint()
    df.to_csv("observations.csv", index=False, header=False)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/fitting_data/hd142527")
    fits_files = []
    fits_files.extend(path.glob("*HAW*.fits"))
    fits_files.extend(path.glob("PION*.fits"))
    read_to_table(fits_files)
