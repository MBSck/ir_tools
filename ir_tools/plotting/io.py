import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from astropy.io import fits

KEY_TO_ATTRIBUTE = {
    "date": "date-obs",
    "instrument": "instrume",
}


def read(
    files_or_hduls: Path | fits.HDUList | List[Path] | List[fits.HDUList],
) -> List[fits.HDUList]:
    if not isinstance(files_or_hduls, (tuple, list, np.ndarray)):
        files_or_hduls = [files_or_hduls]

    hduls = []
    for fits_file in files_or_hduls:
        if isinstance(fits_file, fits.HDUList):
            hdul = fits_file
        else:
            with fits.open(fits_file) as hdul:
                hdul = copy.deepcopy(hdul)

        hduls.append(hdul)
    return hduls


def sort(hduls: List[fits.HDUList], by: str | List[str]) -> List[fits.HDUList]:
    by = [by] if not isinstance(by, (tuple, list, np.ndarray)) else by
    data = {
        "index": range(len(hduls)),
        **{key: [get_header_entry(hdul, key) for hdul in hduls] for key in by},
    }
    return [hduls[i] for i in pd.DataFrame(data).sort_values(by=by)["index"].tolist()]


def filter(hduls: List[fits.HDUList], conditions: Dict[str, Any]) -> List[fits.HDUList]:
    df = pd.DataFrame(
        {
            "index": range(len(hduls)),
            **{
                key: [
                    get_header_entry(hdul, key)
                    for key in conditions.keys()
                    for hdul in hduls
                ]
            },
        }
    )

    for key, value in conditions.items():
        df = df[df[key] == value]

    return [hduls[i] for i in df["index"].tolist()]


def get_header_entry(hdul: List[fits.HDUList], key: str) -> List[str]:
    content = hdul[0].header.get(KEY_TO_ATTRIBUTE.get(key, key).upper(), "")
    if key == "date":
        content = content.split("T")[0]
    return content


def get_hdu(
    hdul: fits.HDUList, name: str, index: int | None
) -> fits.BinTableHDU | None:
    try:
        return hdul[name, index]
    except (KeyError, IndexError):
        return None


def get_column(
    hdul: fits.HDUList,
    card: str,
    column: str,
    index: int | None = None,
    masked: bool = False,
) -> Any:
    hdu = get_hdu(hdul, card, index)
    try:
        values = hdu.data[column]
        if masked:
            values = np.ma.masked_array(values, mask=hdu.data["flag"])
        return values
    except (AttributeError, KeyError):
        return None


def get_labels(hduls: List[fits.HDUList]) -> List[str]:
    return [f"{chr(ord('A') + i)}" for i, _ in enumerate(hduls)]
