import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


def get_target(sheet: Path, target: str) -> pd.DataFrame:
    """Gets the night a target was observed for some criteria."""
    df = pd.read_excel(sheet, skiprows=1, header=[0, 1])
    target_column = df["Target"].iloc[:, 0].astype(str)
    df_cleaned = df.dropna(subset=[("Target", target_column.name)])
    return df_cleaned[
        df_cleaned["Target"].iloc[:, 0].str.contains(target, case=False, regex=True)
    ]
    # return target_df["Night", "Data quality", ]


def get_dir_name(target: str) -> str:
    """Returns the directory name for a target."""
    if target.startswith("HD"):
        return target[:2] + "_" + target[2:]


def get_date(night: str, nights: pd.DataFrame) -> Optional[str]:
    """Returns the date of the night the target was observed."""
    night = datetime.strptime(night, "%Y-%m-%d")
    night_before = (night - timedelta(days=1)).strftime("%Y-%m-%d")
    night_after = (night + timedelta(days=1)).strftime("%Y-%m-%d")
    night = night.strftime("%Y-%m-%d")

    if night in nights.values():
        key = night
    elif night_before in nights.values():
        key = night_before
    elif night_after in nights.values():
        key = night_after
    else:
        key = None

    return key


def sort_target(sheet: Path, target: str, source_dir: Path, target_dir: Path) -> None:
    """Sorts the data for a target."""
    target_dir /= target
    for quality in ["bad", "medium", "good", "tba"]:
        quality_dir = target_dir / quality
        quality_dir.mkdir(parents=True, exist_ok=True)

    df = get_target(sheet, target)
    nights = df["Night"].iloc[:, 0].apply(lambda x: x.strftime("%Y-%m-%d")).to_dict()
    reversed_nights = {v: k for k, v in nights.items()}
    for fits_file in list((source_dir / get_dir_name(target)).glob("*.fits")):
        file_name = fits_file.name
        key = get_date(re.search(r"\d{4}-\d{2}-\d{2}", file_name).group(), nights)
        if key is None:
            shutil.copy(fits_file, target_dir / "tba" / file_name)
        else:
            row = df.loc[reversed_nights[key]]
            config = row["Data description"]["Tel. config."]
            band = "L" if filter(lambda x: "L" in x, file_name.split("_")) else "N"

            if band == "N" and "AT" in config:
                shutil.copy(fits_file, target_dir / "tba" / file_name)
            else:
                quality = row["Data quality"][f"{band} band"]
                if "good" in quality:
                    quality = "good"
                elif "so so" in quality:
                    quality = "medium"
                elif "bad" in quality:
                    quality = "bad"
                else:
                    quality = "tba"

                shutil.copy(fits_file, target_dir / quality / file_name)


if __name__ == "__main__":
    data_dir = Path().home() / "Data"
    exel_file = data_dir / "survey" / "MATISSE data overview.xlsx"
    source_dir = data_dir / "reduced_data" / "jozsef_reductions" / "targets5"
    target_dir = data_dir / "survey"

    targets = ["HD142527"]
    for target in targets:
        sort_target(exel_file, target, source_dir, target_dir)
