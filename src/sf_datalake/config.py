"""Get the config from a config.json file.
"""

import json


def get_config(config_path: str) -> dict:
    """Get the config from a config.json file.

    Args:
        config_path: path of the config .json file

    Returns:
        dict: the config parameters
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    config["TAC_VARIABLES"] = {f"tac_1y_{v}" for v in config["MRV_VARIABLES"]}
    config["SF_VARIABLES"] = list(
        config["SUM_VARIABLES"] + config["AVG_VARIABLES"] + config["COMP_VARIABLES"]
    )
    config["FEATURES"] = list(set(config["SF_VARIABLES"] + config["MRV_VARIABLES"]))

    config["TRANSFORMERS"] = [(config["FEATURES"], "StandardScaler")]
    return config
