from pathlib import Path

import pandas as pd


ROW_COLUMNS = ["Target id", "RAHours", "RAMinutes", "RASeconds",
               "DECHours", "DECMinutes", "DECSeconds", "Instrument",
               "Instrument setup", "Telescope", "Excution time (h)",
               "PI", "Protection", "Comments"]

OBSERVATION_MODES = {"MATISSE": "LR-LM&MR-LM&HR-L&LR-N&HR-N",
                     "VISIR": "VISIR spec-LR"}

TELESCOPES = {"VISIR": "UT2"}


def create_proposal_csv(planning_file: Path,
                        period: str, pi: str) -> None:
    """Creates the proposal csv from the
    planning data.
    """
    df = pd.read_excel(planning_file, sheet_name=f"{period.upper()}_Planning")
    df = df[(df[f"{period.lower()} UTs"] == True)
            | (df[f"{period.lower()} ATs"] == True)]
    # ra, dec = df["Coordinates (J2000)"], df["Unnamed: 9"]
    df[["RAHours", "RAMinutes", "RASeconds"]] = df["Coordinates (J2000)"].str.split(":", expand=True)
    df[["DECHours", "DECMinutes", "DECSeconds"]] = df["Unnamed: 9"].str.split(":", expand=True)

    new_df_rows = []
    for index, row in df.iterrows():
        for instrument in row["Observation Method"].split(","):
            time, name = instrument.split("h ")
            time, name = float(time.strip()), name.strip().upper()

            if row["Unnamed: 45"] and not isinstance(row["Unnamed: 45"], float):
                setup = row["Unnamed: 45"]
            else:
                setup = OBSERVATION_MODES[name]

            if name != "MATISSE":
                telescope = TELESCOPES[name]
            else:
                telescope = "VLTI/4UT" if row[f"{period.lower()} UTs"]\
                    else "VLTI/4AT"

            new_row = dict.fromkeys(ROW_COLUMNS)
            new_row["Target id"] = row["Name"]
            new_row["RAHours"] = df["RAHours"][index]
            new_row["RAMinutes"] = df["RAMinutes"][index]
            new_row["RASeconds"] = df["RASeconds"][index]
            new_row["DECHours"] = df["DECHours"][index]
            new_row["DECMinutes"] = df["DECMinutes"][index]
            new_row["DECSeconds"] = df["DECSeconds"][index]
            new_row["Instrument"] = name
            new_row["Instrument setup"] = setup
            new_row["Telescope"] = telescope
            new_row["Excution time (h)"] = time
            new_row["PI"] = pi.capitalize()
            new_row["Protection"] = 6
            new_row["Comments"] = ""
            new_df_rows.append(new_row)

    new_df = pd.DataFrame(new_df_rows)
    coord_names = {"RA": "Right Ascension", "DEC": "Declination"}
    for coord in ["RA", "DEC"]:
        for col in ["Hours", "Minutes", "Seconds"]:
            if col == "Hours":
                new_df = new_df.rename(
                        columns={f"{coord}{col}": coord_names[coord]})
            else:
                new_df = new_df.rename(columns={f"{coord}{col}": ""})

    new_df.to_csv("P114_MATISSE-consortium.csv", index=False)


if __name__ == "__main__":
    file = Path("P110-P111-P112-P113-P114-Planning.xlsx")
    create_proposal_csv(file, "p114", "lopez")
