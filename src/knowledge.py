"""Measures of financial literacy, numeracy, and abilities to calculate compound interest"""

import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from src.utils import helpers
from src.utils.constants import EXP_1_DATABASE, EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_FILE_1 = Path(__file__).parents[1] / "data" / EXP_1_DATABASE
DATABASE_FILE_2 = Path(__file__).parents[1] / "data" / EXP_2_DATABASE


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


def create_knowledge_dataframe(
    db_connection: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    dataframes = []
    for i, j in zip(
        ["Finance", "Numeracy", "Inflation"],
        ["financial_literacy", "numeracy", "compound"],
    ):
        if table_exists(db_connection, i) == False:
            create_duckdb_database(db_connection=db_connection, initial_creation=True)
        _df = db_connection.sql(f"SELECT * FROM {i}").df()
        _df[j] = count_correct_responses(_df, j)

        # * Check if experiment 1 or 2 to include day or round column respectively
        if "participant.day" in _df.columns:
            round_column = "participant.day"
        else:
            round_column = "participant.round"
            _df[round_column] = 1

        dataframes.append(_df[["participant.label", round_column, j]])

    return helpers.combine_series(
        dataframes, how="left", on=["participant.label", round_column]
    )


def main() -> None:
    """Run script"""
    con = duckdb.connect(DATABASE_FILE_1, read_only=False)
    df = create_knowledge_dataframe(con)
    logger.debug(df.shape)
    logger.debug(df.columns.to_list())

    con = duckdb.connect(DATABASE_FILE_2, read_only=False)
    df = create_knowledge_dataframe(con)
    logger.debug(df.shape)
    logger.debug(df.columns.to_list())


if __name__ == "__main__":
    main()
