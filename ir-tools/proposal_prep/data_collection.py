from typing import Dict

import pandas as pd


class TargetInformation:
    """Collects information on a certain target and
    prints it out in a readable manner.

    Parameters
    ----------
    target : str
        The target's name. For more query results input
        it in the form"""

    def __init__(self, target: str):
        """The class's constructor."""
        self.name = target

    def _get_target_rows(self, data_frame: pd.DataFrame):
        """Gets the rows containing the target."""
        rows_input = data_frame[data_frame.apply(
            lambda row: self.name in row.values, axis=1)]
        rows_wo_space = data_frame[data_frame.apply(
            lambda row: self.name.replace(" ", "")
            in row.values, axis=1)]
        rows_w_underline = data_frame[data_frame.apply(
            lambda row: self.name.replace(" ", "_")
            in row.values, axis=1)]
        return pd.concat([rows_input, rows_wo_space, rows_w_underline])

    def _get_log_entries(self) -> Dict[str, Dict]:
        """Gets all entries of the target from the observation logs."""
        excel_file = pd.ExcelFile("MATISSE observing logs.xlsx")
        observations = {}
        for sheet_name in excel_file.sheet_names[1:]:
            observation = {}
            for index, row in enumerate(self._get_target_rows(
                    excel_file.parse(sheet_name)).iterrows(), start=1):
                row = row[1]
                if not row.empty:
                    observation[f"Observation {index}"] = str(row[2])
            observations[sheet_name] = observation
            observations = {key: value for key, value
                            in observations.items() if value}
        return observations

    def _get_planning_entries(self) -> Dict[str, Dict]:
        """Gets all the entries from planning sheets."""
        excel_file = pd.ExcelFile("P110-P111-P112-P113-Planning.xlsx")
        periods = {}
        for sheet_name in excel_file.sheet_names:
            if "Planning" not in sheet_name:
                continue
            period = {}
            period_name = sheet_name.split("_")[0].lower()
            row = self._get_target_rows(excel_file.parse(sheet_name))
            if not row.empty:
                period["ATs"] =\
                    row[f"{period_name} ATs"].astype(bool).values[0]
                period["UTs"] =\
                    row[f"{period_name} UTs"].astype(bool).values[0]
                period["Notes"] =\
                    row[f"{period_name.capitalize()} Notes"].str.cat().strip()
                periods[period_name] = period
        return periods

    def _get_target_list_entries(self) -> Dict[str, Dict]:
        """Gets all the entries from the target list."""
        data_frame = pd.read_excel("mat_target_list.xlsx",
                                   sheet_name="obs_list_sci")
        observations = {}
        for row in self._get_target_rows(data_frame).iterrows():
            row = row[1]
            observation = {}
            observation["DIT"] = row["DIT"]
            observation["seeing"] = row["seeing"]
            observation["tau0"] = row["tau0"]
            observations[row["night"]] = observation
        return observations

    def print_log_entries(self):
        """Prints the entries from the observation logs."""
        print("The observation logs contain the following entries.")
        print(f"{'':-^60}")
        for date, observation in self._get_log_entries().items():
            print(f"{date}:")
            for key, value in observation.items():
                print(f"\t{key}'s comment: {value.capitalize()}.")
        print()

    def print_planning_entries(self):
        """Prints the target from the planning sheets."""
        print("The planning sheets contain the following entries.")
        print(f"{'':-^60}")
        for key, value in self._get_planning_entries().items():
            ats = "YES" if value["ATs"] else "NO"
            uts = "YES" if value["UTs"] else "NO"
            print(f"{key.capitalize()}:\tSelected for ATs '{ats}', "
                  f"selected for UTs '{uts}'.\n"
                  f"\tNotes - {value['Notes'].capitalize()}")
        print()

    def print_list_entries(self):
        """Prints the observed target from the targets list."""
        print("The target lists contain the following entries.")
        print(f"{'':-^60}")
        for key, value in self._get_target_list_entries().items():
            print(f"{key}: Seeing {value['seeing']} arcsec, "
                  f"tau0 {value['tau0']} ms, and DIT {value['DIT']}.")
        print()

    def print_overview(self):
        """Prints the entries from all the sheets."""
        self.print_log_entries()
        self.print_planning_entries()
        self.print_list_entries()


if __name__ == "__main__":
    target = TargetInformation("HD 142527")
    target.print_overview()
