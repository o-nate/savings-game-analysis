"""Helper functions"""

import logging
from pathlib import Path
import sys
from typing import List

from functools import reduce
import matplotlib.pyplot as plt
import pandas as pd
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


def to_numeric_ignore(series: pd.Series) -> pd.Series:
    """Convert values to numbers, leaving the series unchanged if that fails.

    Pandas 3 removed ``errors="ignore"`` from ``DataFrame.apply(pd.to_numeric)``.
    That option converted a column only when every value was numeric.
    """
    try:
        return pd.to_numeric(series, errors="raise")
    except (ValueError, TypeError):
        return series


def combine_mean_std_dicts(mean_dict, std_dict):
    combined = {}
    for outer_key in mean_dict:
        combined[outer_key] = {}
        for inner_key in mean_dict[outer_key]:
            mean_val = mean_dict[outer_key][inner_key]
            std_val = std_dict.get(outer_key, {}).get(inner_key, None)
            if std_val is not None:
                combined[outer_key][inner_key] = f"{mean_val:.2f}<br>({std_val:.2f})"
            else:
                combined[outer_key][inner_key] = f"{mean_val:.2f}<br>(N/A)"

    return combined


def combine_series(dataframes: List[pd.DataFrame], **kwargs) -> pd.DataFrame:
    """Merge dataframes from a list

    Args:
        dataframes List[pd.DataFrame]: List of dataframes to merge

    Kwargs:
        on List[str]: Column(s) to merge on
        how {str}: 'left', 'right', 'inner', 'outer'

    Returns:
        pd.DataFrame: Combined dataframe
    """
    return reduce(lambda left, right: pd.merge(left, right, **kwargs), dataframes)


def export_plot(
    exported_file_path: Path, file_name: str, export_all_plots: str = "n"
) -> None:
    """Export plot to results folder in png format

    Args:
        exported_file_path (Path): directory to save plot
        file_name (str): plot file name
        export_all_plots (str, optional): if all plots are to be exported. Defaults to "n".
    """
    file_path = exported_file_path / file_name
    if export_all_plots == "y":
        plt.savefig(file_path, bbox_inches="tight")
    elif export_all_plots != "n":
        return
    elif input(f"Export {file_name}? (y) ").lower() == "y":
        plt.savefig(file_path, bbox_inches="tight")
