"""Module to process economic preferences data"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.utils import constants
from src.utils import helpers
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE


def count_preference_choices(data: pd.DataFrame, econ_preference: str) -> pd.Series:
    """Count choices for economic preference. For loss aversion, count the number
    of decisions to toss the coin. For risk aversion, count the number of safe
    choices. For time preferences, count the number of smaller-sooner choices.

    Args:
        data (pd.DataFrame): Data
        econ_preference (str): Economic preference measure ('lossAversion',
        'riskAversion', 'timePreferences')

    Returns:
        pd.Series: Number of choices
    """
    if econ_preference == "wisconsin":
        return data["wisconsin.1.player.num_correct"]
    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    if econ_preference == "timePreferences":
        return (data[cols] == 1).sum(axis=1)
    return data[cols].sum(axis=1)


def count_switches(data: pd.DataFrame, econ_preference: str) -> pd.Series:
    """Count changes in loss aversion and risk preferences (number greater than
    1 suggest inconsistent preferences)

    Args:
        data (pd.DataFrame): Data
        econ_preference (str): Economic preference measure ('lossAversion',
        'riskPreferences')

    Returns:
        pd.Series: Number of switches
    """
    if econ_preference == "timePreferences":
        _data = data.copy()
        for r in range(1, 1 + constants.TIME_PREFERENCES_ROUNDS):
            cols = [c for c in _data.columns if f"timePreferences.{r}.player.q" in c]
            ## Convert to booleans
            _data[cols] = np.where(_data[cols] == 1, True, False)
            _data[f"count_{r}"] = (_data[cols] != _data[cols].shift(axis=1)).sum(
                axis=1
            ) - 1
        return _data[[c for c in _data.columns if "count_" in c]].sum(axis=1)

    column_selector = constants.CHOICES[econ_preference]
    cols = [c for c in data.columns if column_selector in c]
    return (data[cols] != data[cols].shift(axis=1)).sum(axis=1) - 1


def count_wisconsin_errors(
    data: pd.DataFrame, error_type: str, num_trials: int = 30
) -> pd.Series:
    """Count number of perseverative or set-loss erros from Wisconsin Card Sorting Task.
    Perseverative errors are failures to adapt decisions to negative feedback. Set-loss
    errors are failures to maintain a decision, given positive feedback.

    Args:
        data (pd.DataFrame): Wisconsin Card Sorting Task data
        error_type (str): `perseverative` or `set-loss`
        num_trials (int, optional): Total number of decisions. Defaults to 30.

    Raises:
        ValueError: When incorrect `error_type` defined.

    Returns:
        pd.Series: Series of total error for defined `error_type` per subject
    """
    _data = data.copy()
    ## Start at trial 2 since we can only have errors after 1st trial
    for n in range(2, 1 + num_trials):
        if error_type == "perseverative":
            criteria = [
                _data[f"correct_{n-1}"].eq(False)
                & (_data[f"guess_{n-1}"] == _data[f"guess_{n}"])
            ]
        elif error_type == "set-loss":
            criteria = [
                _data[f"correct_{n-1}"].eq(True)
                & (_data[f"guess_{n-1}"] != _data[f"guess_{n}"])
            ]
        else:
            raise ValueError(
                "Please, choose either `perseverative` or `set-loss` error type."
            )
        choices = [1]
        if n == 2:
            _data["n_error"] = np.select(criteria, choices, default=0)
        else:
            _data["n_error"] += np.select(criteria, choices, default=0)
    return _data["n_error"]


def create_econ_preferences_dataframe() -> pd.DataFrame:
    """Generate DataFrame with economic preference ("lossAversion", number of coins
    tossed; "riskPreferences", number safe lotteries chosen; "timePreferences",
    number of smaller-sooner payments chosen; "wisconsin", number of correct choices
    and number of perseverative and set-loss errors) measures for each subject

    Returns:
        pd.DataFrame: DataFrame with columns [participant.label,
        "lossAversion_choice_count", "riskPreferences_choice_count",
        "timePreferences_choice_count", "wisconsin_choice_count",
        "lossAversion_switches", "riskPreferences_switches",
        "timePreferences_switches", "wisconsin_PE", "wisconsin_SE"]
    """
    con = duckdb.connect(DATABASE_FILE, read_only=False)
    if table_exists(con, "lossAversion") == False:
        create_duckdb_database(con, initial_creation=True)
    dataframes = []
    for pref in ["lossAversion", "riskPreferences", "timePreferences", "wisconsin"]:
        _df = con.sql(f"SELECT * FROM {pref}").df()
        _df[f"{pref}_choice_count"] = count_preference_choices(_df, pref)
        if pref != "wisconsin":
            _df[f"{pref}_switches"] = count_switches(_df, pref)
            dataframes.append(
                _df[["participant.label", f"{pref}_choice_count", f"{pref}_switches"]]
            )
        else:
            _df["wisconsin_PE"] = count_wisconsin_errors(_df, "perseverative")
            _df["wisconsin_SE"] = count_wisconsin_errors(_df, "set-loss")
            dataframes.append(
                _df[
                    [
                        "participant.label",
                        f"{pref}_choice_count",
                        "wisconsin_PE",
                        "wisconsin_SE",
                    ]
                ]
            )
    return helpers.combine_series(
        dataframes=dataframes, how="left", on="participant.label"
    )


def main() -> None:
    """Run script"""
    df = create_econ_preferences_dataframe()
    logger.debug(df.shape)


if __name__ == "__main__":
    main()
