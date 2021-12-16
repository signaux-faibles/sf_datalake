"""Config class.

Load the config from a config.json file.
"""

import json


class Config:  # pylint: disable=R0903
    """
    Class to load and store the config parameters.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data["TAC_VARIABLES"] = {f"tac_1y_{v}" for v in data["MRV_VARIABLES"]}
        data["SF_VARIABLES"] = (
            set(data["SUM_VARIABLES"])
            | set(data["AVG_VARIABLES"])
            | set(data["COMP_VARIABLES"])
        )
        data["SF_VARIABLES"] = list(data["SF_VARIABLES"])
        data["FEATURES"] = set(data["SF_VARIABLES"]) | set(data["MRV_VARIABLES"])
        data["FEATURES"] = list(data["FEATURES"])
        data["STD_SCALE_FEATURES"] = data["FEATURES"]
        self.data = data

    def get_config(self):
        """Return the config parameters as a dict

        Returns:
            dict: the config parameters
        """
        return self.data
