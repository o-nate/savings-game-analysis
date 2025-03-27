"""Measures of financial literacy, numeracy, and abilities to calculate compound interest"""

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.utils import helpers
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE


def count_correct_responses(data: pd.DataFrame, knowledge_measure: str) -> pd.Series:

    if knowledge_measure == "financial_literacy":
        _criteria = [
            data["Finance.1.player.finK_1"].eq(1)
            & data["Finance.1.player.finK_2"].eq(-1)
            & data["Finance.1.player.finK_9"].eq(1)
        ]
    if knowledge_measure == "numeracy":
        _criteria = [
            data["Numeracy.1.player.num_2b"].eq(20)
            | data["Numeracy.1.player.num_3"].eq(50)
        ]
    if knowledge_measure == "compound":
        _criteria = [
            data["Inflation.1.player.infCI_1"].eq(1100)
            & data["Inflation.1.player.infCI_2"].eq(2)
            & data["Inflation.1.player.infCI_3"].eq(2)
            & data["Inflation.1.player.infCI_4"].eq(32000)
        ]
    choices = [1]
    return np.select(_criteria, choices, default=0)


def create_knowledge_dataframe() -> pd.DataFrame:
    con = duckdb.connect(DATABASE_FILE, read_only=False)
    dataframes = []
    for i, j in zip(
        ["Finance", "Numeracy", "Inflation"],
        ["financial_literacy", "numeracy", "compound"],
    ):
        if table_exists(con, i) == False:
            create_duckdb_database(con, initial_creation=True)
        _df = con.sql(f"SELECT * FROM {i}").df()
        _df[j] = count_correct_responses(_df, j)
        dataframes.append(_df[["participant.label", j]])
    return helpers.combine_series(dataframes, how="left", on="participant.label")


def main() -> None:
    """Run script"""
    df = create_knowledge_dataframe()
    print(df.head())
    logger.debug(df[df["participant.label"] == "9xHTKNJ"])


if __name__ == "__main__":
    main()
