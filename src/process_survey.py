"""Process survey responses of perceived and expected inflation"""

import logging
import sys
from typing import Any, List, Tuple
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from src.utils.constants import INFLATION_DICT
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_series
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Declare duckdb database info
DATABASE_FILE = Path(__file__).parents[1] / "data" / "database.duckdb"


def calculate_estimate_bias(
    data: pd.DataFrame, estimate_col: str, real_inflation_col: str
) -> pd.Series:
    """Takes the difference bewteen real and estimate inflation (real minus estimate)

    Args:
        data (pd.DataFrame): DataFrame with estimate and real inflation columns
        estimate_col (str): column name of estimates
        real_inflation_col (str): column name of real inflation

    Returns:
        pd.Series: returns with column for bias
    """
    return data[estimate_col] - data[real_inflation_col]


def calculate_estimate_sensitivity(
    data: pd.DataFrame,
    estimate_col: str,
    real_inflation_col: str,
    new_column_name: str,
    fill_nans: bool = True,
) -> pd.DataFrame:
    """Pearson correlation between inflation estimations (expectations and perceptions)
    and realized inflation rate

    Args:
        data (pd.DataFrame)
        estimate_col (str): Name of column with estimate data
        real_inflation_col (str): Name of column with inflation data
        new_column_name (str): Name for new column
        fill_nans (bool): Fill occurances with no change (std = 0) with 0's

    Returns:
        pd.DataFrame: DataFrame with Pearson correlation coefficient, one for
        each participant.code
    """
    data_corr = pd.DataFrame(
        data.groupby(["participant.code"])[estimate_col].corr(data[real_inflation_col])
    ).reset_index()
    data_corr.rename(columns={estimate_col: new_column_name}, inplace=True)
    if fill_nans:
        return data_corr.fillna(0)
    return data_corr


def include_inflation_measures(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Adds inflation measure columns to DataFrame

    Args:
        data (pd.DataFrame): DataFrame with columns 'Quant Perception', 'Quant Expectation',
        'Actual', and 'Upcoming' inflation measures

    Kwargs:
        fill_nans (bool):  Fill occurances with no change (std = 0) with 0's for
        sensitivity measures

    Returns:
        pd.DataFrame: DataFrame with column for each new measure with '_bias' and '_sensitivity'
        suffixes
    """
    final_data = data.copy()
    for estimate, actual in zip(["Perception", "Expectation"], ["Actual", "Upcoming"]):
        final_data[f"{estimate}_bias"] = calculate_estimate_bias(
            final_data, f"Quant {estimate}", actual
        )
        _ = calculate_estimate_sensitivity(
            final_data, f"Quant {estimate}", actual, f"{estimate}_sensitivity", **kwargs
        )
        final_data = final_data.merge(_, how="left")
    return final_data


def determine_qualitative_accuracy(
    data: pd.DataFrame, qualitative_estimate_col: str, default: float
) -> npt.NDArray:
    """Generate series with boolean value for whether subjects' qualitative estimates
    were in-line with inflation

    Args:
        data (pd.DataFrame): DataFrame with inflation estimates
        qualitative_estimate_col (str): Name of column with qualitative estimate
        conditions_list (List[Tuple[Any, Any]]): Conditions of accurate estimates
        default (float): Default value to fill for 'else' clause

    Returns:
        npt.NDArray: Array with accuracy values
    """
    conditions_list = [
        (data["inf_phase"] == "high") & (data[qualitative_estimate_col] > 1),
        (data["inf_phase"] == "low")
        & (data[qualitative_estimate_col] >= 0)
        & (data[qualitative_estimate_col] <= 1),
        data[qualitative_estimate_col].isna(),
    ]
    return np.select(conditions_list, [1, 1, np.nan], default=default)


def pivot_inflation_measures(data: pd.DataFrame) -> pd.DataFrame:
    """Pivots inflation and inflation estimate dataframes to calculate inflation df_measures

    Args:
        data (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    df_inf = pd.DataFrame(INFLATION_DICT)
    df_inf = pd.pivot_table(
        df_inf,
        index=["participant.inflation", "participant.round", "Month"],
        columns="Measure",
    )

    ## Remove multiindex columns
    df_inf.columns = [
        "_".join(str(i) for i in a) for a in df_inf.columns.to_flat_index()
    ]
    df_inf.reset_index(inplace=True)

    df_pivot = pd.pivot_table(
        data=data,
        index=[
            "participant.code",
            "participant.label",
            "date",
            "participant.inflation",
            "participant.round",
            "Month",
        ],
        columns="Measure",
    )

    ## Remove multiindex columns
    df_pivot.columns = [
        "_".join(str(i) for i in a) for a in df_pivot.columns.to_flat_index()
    ]
    df_pivot.reset_index(inplace=True)

    df_pivot = df_pivot.merge(
        df_inf[[c for c in df_inf.columns if "inflation" not in c]],
        how="left",
        on=["Month", "participant.round"],
    )

    df_pivot.rename(
        columns={
            col: col.removeprefix("Estimate_")
            for col in df_pivot.columns
            if "Estimate_" in col
        },
        inplace=True,
    )
    return df_pivot


def include_uncertainty_measure(
    data: pd.DataFrame, estimate_col: str, val_uncertain: int, val_certain: int
) -> pd.Series:
    """
    Generate a series with a proxy for uncertain estimations based on the assumption
    that estimates that are multiples of 5 and not equal to 0 are uncertain.

    Args:
        data (pd.DataFrame): DataFrame with quantitative estimates
        estimate_col (str): Name of column to evaluate
        val_uncertain (int): Value assigned to uncertain estimates in series
        val_certain (int): Value assigned to certain estimates in series

    Returns:
        pd.Series: Series of proxy values
    """
    return np.where(
        (data[estimate_col] % 5 == 0) & (data[estimate_col] != 0),
        val_uncertain,
        val_certain,
    )


def create_survey_df(include_inflation: bool = False) -> pd.DataFrame:
    """Creates a merged dataframe with the estimates (perceptions
    and expectations) and actual inflation in a single dataframe"""

    con = duckdb.connect(DATABASE_FILE, read_only=False)
    if (
        table_exists(con, "inf_expectation")
        and table_exists(con, "inf_estimate")
        and table_exists(con, "qualitative_expectation")
        and table_exists(con, "qualitative_estimate")
    ) == False:
        create_duckdb_database(con, initial_creation=True)

    # * Combine estimates into one dataframe
    df1 = con.sql("SELECT * FROM inf_expectation").df()
    df2 = con.sql("SELECT * FROM inf_estimate").df()
    df3 = con.sql("SELECT * FROM qualitative_expectation").df()
    df4 = con.sql("SELECT * FROM qualitative_estimate").df()
    dfs = [df1, df2, df3, df4]
    df5 = combine_series(
        dfs,
        on=[
            "participant.code",
            "participant.label",
            "participant.inflation",
            "treatment",
            "date",
            "participant.round",
        ],
        how="left",
    )
    logger.debug(df5.columns.to_list())
    # * Interpolate qualitative responses of no change as 0 for quantitative
    perception_cols = [f"task.{month*12}.player.inf_estimate" for month in range(1, 11)]
    qual_perception_cols = [
        f"task.{month*12}.player.qualitative_estimate" for month in range(1, 11)
    ]
    expectation_cols = [
        (
            f"task.{month+1}.player.inf_expectation"
            if month == 0
            else f"task.{month*12}.player.inf_expectation"
        )
        for month in range(10)
    ]
    qual_expectation_cols = [
        (
            f"task.{month+1}.player.qualitative_expectation"
            if month == 0
            else f"task.{month*12}.player.qualitative_expectation"
        )
        for month in range(10)
    ]
    for qual, quant in zip(qual_perception_cols, perception_cols):
        df5[quant] = np.where(df5[qual] == 0, 0, df5[quant])
        df5[quant] = np.select(
            [
                (df5[qual] < 0) & (df5[quant] > 0),
                (df5[qual] > 0) & (df5[quant] < 0),
            ],
            [
                df5[quant] * -1,
                df5[quant] * -1,
            ],
            default=df5[quant],
        )
    for qual, quant in zip(qual_expectation_cols, expectation_cols):
        df5[quant] = np.where(df5[qual] == 0, 0, df5[quant])
        df5[quant] = np.select(
            [
                (df5[qual] < 0) & (df5[quant] > 0),
                (df5[qual] > 0) & (df5[quant] < 0),
            ],
            [
                df5[quant] * -1,
                df5[quant] * -1,
            ],
            default=df5[quant],
        )

    logger.debug(df5.columns.to_list())
    df_survey = df5.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "date",
            "participant.inflation",
            "participant.round",
        ],
        value_vars=[c for c in df5.columns if ("inf_" in c) or ("qualitative_" in c)],
        var_name="Measure",
        value_name="Estimate",
    )

    # * Extract month number
    df_survey["Month"] = df_survey["Measure"].str.extract(r"(\d+)")

    # * Convert columns to int, except participant.time_started_utc
    cols_to_convert = [c for c in df_survey.columns if c != "date"]
    df_survey[cols_to_convert] = df_survey[cols_to_convert].apply(
        pd.to_numeric, errors="ignore"
    )

    # * Rename df_measures
    df_survey["Measure"] = df_survey["Measure"].str.split("player.").str[1]
    df_survey["Measure"].replace(
        [
            "inf_estimate",
            "inf_expectation",
            "qualitative_estimate",
            "qualitative_expectation",
        ],
        [
            "Quant Perception",
            "Quant Expectation",
            "Qual Perception",
            "Qual Expectation",
        ],
        inplace=True,
    )
    logger.debug(df_survey.info())

    # * Add actual inflation
    if include_inflation:
        logger.debug("INFLATION_DICT %s", INFLATION_DICT)
        ## Convert to dataframe
        df_inf = pd.DataFrame(INFLATION_DICT)
        ## Merge with survey responses
        df_survey = pd.concat([df_survey, df_inf], ignore_index=True)
    df_survey["participant.inflation"].replace(
        [430, 1012],
        ["4x30", "10x12"],
        inplace=True,
    )

    return df_survey


def main() -> None:
    """Run script"""
    df = create_survey_df()
    df_measures = pivot_inflation_measures(df)
    df_measures = include_inflation_measures(df_measures)
    print(df_measures.head())

    print("participants included: ", df["participant.label"].nunique())

    # # * Plot qualitative responses
    qual_responses = ["Qual Perception", "Qual Expectation"]
    print(df[df["Measure"].isin(qual_responses)].head())

    df2 = df.copy()
    df2["Month"] = df2["Month"].astype(str)

    sns.catplot(
        data=df2[df2["Measure"].isin(qual_responses)],
        x="Estimate",
        y="Month",
        col="Measure",
        hue="participant.round",
        kind="violin",
        split=True,
        palette="bright",
    )

    # * Plot estimates over time
    estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
    g = sns.relplot(
        data=df[df["Measure"].isin(estimates)],
        x="Month",
        y="Estimate",
        errorbar=None,
        hue="Measure",
        style="Measure",
        kind="line",
        col="participant.round",
    )

    ## Adjust titles
    (
        g.set_axis_labels("Month", "Inflation rate (%)")
        # .set_titles("Savings Game round: {col_name}")
        .tight_layout(w_pad=0.5)
    )

    # sns.kdeplot(df=df[df["Measure"].isin(estimates)], x="Estimate", hue="Measure")
    # plt.yscale("log")
    # plt.xscale("log")

    plt.show()


if __name__ == "__main__":
    main()
