from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.io import fits

ARRAY_CONFIGS = {
    "A0-B2-D0-C1": "small",
    "K0-G2-D0-J3": "medium",
    "A0-G1-J2-J3": "large",
    "A0-G1-J2-K0": "large",
    "A0-B5-J2-J6": "large",
    "U1-U2-U3-U4": "UTs",
}

def read_to_table(fits_files: Path, save_as_csv: Optional[bool] = False,
                  savefig: Optional[Path] = False):
    """Read a FITS file and return the data as an astropy table."""
    data = {"date": [], "instrument": [], "seeing": [], "tau0": [],
            "stations": [], "array": [], "name": [], "ldd": [],
            "time": [], "Comment": []}

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header
            data["date"].append(header["DATE-OBS"][:-8])

            instrument = header["INSTRUME"].upper()
            instrument_mode = instrument
            if instrument.lower() == "matisse" and header["HIERARCH ESO DEL FT SENSOR"].lower() == "gravity":
                instrument_mode = "GRA4MAT"
            data["instrument"].append(instrument_mode)

            data["seeing"].append(
                round(np.mean([header["HIERARCH ESO ISS AMBI FWHM START"],
                               header["HIERARCH ESO ISS AMBI FWHM END"]]), 1))
            data["tau0"].append(
                round(np.mean([header["HIERARCH ESO ISS AMBI TAU0 START"],
                               header["HIERARCH ESO ISS AMBI TAU0 END"]])*1e3, 1))
            stations = "-".join([header[f"HIERARCH ESO ISS CONF STATION{i}"] for i in range(1, 5)])
            data["stations"].append(stations)

            if stations not in ARRAY_CONFIGS:
                array = "other"
            else:
                array = ARRAY_CONFIGS[stations]
            data["array"].append(array)

            cal_name, cal_ldd, cal_time = None, None, None
            if "HIERARCH ESO PRO CAL DB NAME" in header:
                cal_name = header["HIERARCH ESO PRO CAL DB NAME"].strip()
                cal_time = header["HIERARCH ESO PRO CAL TPL START"].split("T")[1][:5]
                cal_ldd = round(header["HIERARCH ESO PRO CAL DB DIAM"], 1)

            data["name"].append(cal_name)
            data["ldd"].append(cal_ldd)
            data["time"].append(cal_time)

            comment = None
            if instrument.lower() == "pionier":
                comment = "H"
            elif instrument.lower() == "gravity":
                comment = "K"
            elif instrument.lower() == "matisse":
                if array == "UTs":
                    comment = "LN"
                else:
                    comment = "L"
            data["Comment"].append(comment)

    pd.DataFrame(data).to_csv("observations.csv", index=False, header=False)


if __name__ == "__main__":
    read_to_table(Path("/Users/scheuck/Data/fitting_data/hd142527").glob("*HAW*.fits"))
