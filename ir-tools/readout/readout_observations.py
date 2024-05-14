import re
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import fits

ARRAY_CONFIGS = {}


def add_array_config(dictionary: Dict, key: str, value: str) -> Dict:
    """Adds all permutations of the configuration to the dictionary."""
    perms = map(lambda x: "-".join(x), permutations(key.split("-")))
    return {**dictionary, **{perm: value for perm in perms}}


# TODO: Potentially add this to the matadrs
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B2-D0-C1", "small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-B2-C1-D0", "small")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "K0-G2-D0-J3", "medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-H0-G1-I1", "medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "D0-G2-J3-K0", "medium")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-J3", "large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-G1-J2-K0", "large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-I1", "large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A1-G1-K0-J3", "large")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "A0-B5-J2-J6", "extended")
ARRAY_CONFIGS = add_array_config(ARRAY_CONFIGS, "U1-U2-U3-U4", "UTs")


def read_to_table(fits_files: List[Path], save_as_csv: Optional[bool] = False,
                  savefig: Optional[Path] = False):
    """Read a FITS file and return the data as an astropy table."""
    data = {"instrument": [], "date": [], "seeing": [], "tau0": [],
            "stations": [], "array": [], "name": [], "ldd": [],
            "time": [], "Comment": []}

    for fits_file in fits_files:
        with fits.open(fits_file) as hdul:
            header = hdul[0].header

            instrument = ""
            if "INSTRUME" not in header:
                if fits_file.name.startswith("PION"):
                    instrument = "PIONIER"
                elif fits_file.name.startswith("GRAV"):
                    instrument = "GRAVITY"
            else:
                instrument = header["INSTRUME"].upper().strip()
            instrument_mode = instrument
            if instrument.lower() == "matisse" and header["HIERARCH ESO DEL FT SENSOR"].lower() == "gravity":
                instrument_mode = "GRA4MAT"

            data["instrument"].append(instrument_mode)

            date = ""
            if "DATE-OBS" in header:
                date = header["DATE-OBS"]
            elif "DATE" in header:
                date = header["DATE"]
            else:
                date = re.findall(r"\d{4}-\d{2}-\d{2}", fits_file.name)[0]

            data["date"].append(date[:-8])

            seeing = ""
            if "HIERARCH ESO ISS AMBI FWHM START" in header:
                seeing = round(np.mean([header["HIERARCH ESO ISS AMBI FWHM START"],
                                        header["HIERARCH ESO ISS AMBI FWHM END"]]), 1)
            data["seeing"].append(seeing)

            tau0 = ""
            if "HIERARCH ESO ISS AMBI TAU0 START" in header:
                tau0 = round(np.mean([header["HIERARCH ESO ISS AMBI TAU0 START"],
                                    header["HIERARCH ESO ISS AMBI TAU0 END"]])*1e3, 1)
            data["tau0"].append(tau0)

            stations = ""
            if "HIERARCH ESO ISS CONF STATION1" in header:
                stations = "-".join([header[f"HIERARCH ESO ISS CONF STATION{i}"] for i in range(1, 5)])
            data["stations"].append(stations)

            array = ARRAY_CONFIGS[stations] if stations in ARRAY_CONFIGS else "other"
            data["array"].append(array)

            cal_name, cal_ldd, cal_time = "", "", ""
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

    df = pd.DataFrame(data).sort_values(by="date")
    df.to_csv("observations.csv", index=False, header=False)


if __name__ == "__main__":
    path = Path("/Users/scheuck/Data/fitting_data/hd142527")
    fits_files = []
    fits_files.extend(path.glob("*HAW*.fits"))
    fits_files.extend(path.glob("PION*.fits"))
    read_to_table(fits_files)
