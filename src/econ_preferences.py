"""Module to process economic preferences data"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src.utils import constants
from src.utils import helpers
from src.utils.constants import EXP_1_DATABASE, EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_FILE_1 = Path(__file__).parents[1] / "data" / EXP_1_DATABASE
DATABASE_FILE_2 = Path(__file__).parents[1] / "data" / EXP_2_DATABASE

# Holt-Laury payoffs from savings-game/riskPreferences/stimuli.csv
HL_OPTION_A = (2.00, 1.60)  # safe (high, low)
HL_OPTION_B = (3.85, 0.10)  # risky (high, low)
HL_HIGH_PROBS = [i / 10 for i in range(1, 11)]  # P(high payoff) per row, 0.1..1.0
_CARA_GAMMA_EPS = 1e-9
_CARA_GAMMA_BRACKET = (-50.0, 50.0)


def _cara_eu(gamma: float, high: float, low: float, p: float) -> float:
    """Normalized CARA expected utility; reduces to expected value as gamma -> 0."""
    if abs(gamma) < _CARA_GAMMA_EPS:
        return p * high + (1 - p) * low
    return (1 - (p * np.exp(-gamma * high) + (1 - p) * np.exp(-gamma * low))) / gamma


def _indifference_gap(gamma: float, p: float) -> float:
    """EU_B - EU_A at probability p; monotone in gamma for fixed p."""
    return _cara_eu(gamma, *HL_OPTION_B, p) - _cara_eu(gamma, *HL_OPTION_A, p)


def _solve_gamma_at_p(p: float, bracket: tuple[float, float] = _CARA_GAMMA_BRACKET) -> float:
    """Solve gamma such that EU_A = EU_B at probability p."""
    if p >= 1.0 - 1e-12:
        return np.inf
    if p <= 1e-12:
        return -np.inf

    lo, hi = bracket
    gap_lo, gap_hi = _indifference_gap(lo, p), _indifference_gap(hi, p)
    if gap_lo > 0 and gap_hi > 0:
        return np.inf
    if gap_lo < 0 and gap_hi < 0:
        return -np.inf
    return brentq(_indifference_gap, lo, hi, args=(p,))


def cara_gamma_intervals(safe_counts: pd.Series) -> pd.DataFrame:
    """CARA gamma interval (low, high, midpoint) per Holt-Laury safe-choice count k.

    k safe choices imply indifference between rows k and k+1:
    gamma in [gamma*(p_k), gamma*(p_{k+1})]. Open sides return NaN; midpoint is NaN
    unless both endpoints are finite.

    Args:
        safe_counts (pd.Series): Number of safe (Option A) choices per participant.

    Returns:
        pd.DataFrame: Columns risk_gamma_low, risk_gamma_high, risk_gamma_mid.
    """
    edges = [_solve_gamma_at_p(p) for p in HL_HIGH_PROBS]
    rows: dict[int, tuple[float, float, float]] = {}
    for k in range(11):
        low = edges[k - 1] if k >= 1 else -np.inf
        high = edges[k] if k <= 9 else np.inf
        low = low if np.isfinite(low) else np.nan
        high = high if np.isfinite(high) else np.nan
        mid = (low + high) / 2 if np.isfinite(low) and np.isfinite(high) else np.nan
        rows[k] = (low, high, mid)

    def _lookup(count: float) -> tuple[float, float, float]:
        if pd.isna(count):
            return (np.nan, np.nan, np.nan)
        return rows[int(round(count))]

    out = safe_counts.map(_lookup)
    return pd.DataFrame(
        out.tolist(),
        index=safe_counts.index,
        columns=["risk_gamma_low", "risk_gamma_high", "risk_gamma_mid"],
    )


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
    """Count number of perseverative or set-loss errors from Wisconsin Card Sorting Task.
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


def create_econ_preferences_dataframe(
    db_connection: duckdb.DuckDBPyConnection, include_wisconsin_errors: bool = False
) -> pd.DataFrame:
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
    if table_exists(db_connection, "lossAversion") == False:
        create_duckdb_database(db_connection, initial_creation=True)
    dataframes = []
    for pref in ["lossAversion", "riskPreferences", "timePreferences", "wisconsin"]:
        _df = db_connection.sql(f"SELECT * FROM {pref}").df()

        # * Check if experiment 1 or 2 to include day or round column respectively
        if "participant.day" in _df.columns:
            round_column = "participant.day"
        else:
            round_column = "participant.round"
            _df[round_column] = 1

        _df[f"{pref}_choice_count"] = count_preference_choices(_df, pref)
        if pref != "wisconsin":
            _df[f"{pref}_switches"] = count_switches(_df, pref)
            dataframes.append(
                _df[
                    [
                        "participant.label",
                        f"{pref}_choice_count",
                        f"{pref}_switches",
                    ]
                ]
            )
        elif pref == "wisconsin" and include_wisconsin_errors:
            _df["wisconsin_PE"] = count_wisconsin_errors(_df, "perseverative")
            _df["wisconsin_SE"] = count_wisconsin_errors(_df, "set-loss")
            dataframes.append(
                _df[
                    [
                        "participant.label",
                        round_column,
                        f"{pref}_choice_count",
                        "wisconsin_PE",
                        "wisconsin_SE",
                    ]
                ]
            )
        else:
            dataframes.append(
                _df[["participant.label", round_column, f"{pref}_choice_count"]]
            )
    return helpers.combine_series(
        dataframes=dataframes, how="left", on="participant.label"
    )


def main() -> None:
    """Run script"""
    con = duckdb.connect(DATABASE_FILE_1, read_only=False)
    df = create_econ_preferences_dataframe(con)
    logger.debug(df.shape)
    logger.debug(df.columns.to_list())

    con = duckdb.connect(DATABASE_FILE_2, read_only=False)
    df = create_econ_preferences_dataframe(con)
    logger.debug(df.shape)
    logger.debug(df.columns.to_list())


if __name__ == "__main__":
    main()
