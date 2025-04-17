"""Miscellaneous functions that need to be significantly adjusted for Experiment 1 data"""

import numpy as np
import pandas as pd

from src.utils.constants import EXP_1_COLUMNS_HASH, INFLATION_DICT


def conform_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """For conforming column names from Experiment 1 to those of Experiment 2"""
    return data.rename(columns=EXP_1_COLUMNS_HASH)


def conform_participant_rounds(data: pd.DataFrame) -> pd.DataFrame:
    """For conforming rounds from Experiment 1 to those of Experiment 2"""
    return np.where(data["participant.day"].lt(3), 1, 2)


def create_survey_df(
    perceptions_data: pd.DataFrame,
    expectations_data: pd.DataFrame,
    include_inflation: bool = False,
) -> pd.DataFrame:
    ## Get perceptions and expectations data
    print(perceptions_data.shape, expectations_data.shape)
    df3 = perceptions_data.merge(expectations_data, how="left")
    df3 = conform_column_names(df3)
    df3["participant.round"] = conform_participant_rounds(df3)
    df_survey = df3.melt(
        id_vars=[
            "participant.code",
            "participant.label",
            "participant.inflation",
            "treatment",
            "date",
            "participant.round",
        ],
        value_vars=[c for c in df3.columns if "inf_" in c],
        var_name="Measure",
        value_name="Estimate",
    )
    ## Extract month number
    df_survey["Month"] = df_survey["Measure"].str.extract("(\d+)")
    ## Convert to int
    cols_to_convert = [c for c in df_survey.columns if c != "date"]
    df_survey[cols_to_convert] = df_survey[cols_to_convert].apply(
        pd.to_numeric, errors="ignore"
    )
    ## Rename measures
    df_survey["Measure"] = df_survey["Measure"].str.split("player.").str[1]
    df_survey["Measure"].replace(
        ["inf_estimate", "inf_expectation"],
        ["Quant Perception", "Quant Expectation"],
        inplace=True,
    )
    if include_inflation:
        ## Add actual inflation
        df_inf = pd.DataFrame(INFLATION_DICT)
        df_survey = pd.concat([df_survey, df_inf], ignore_index=True)
    df_survey["participant.inflation"].replace(
        [430, 1012],
        ["4x30", "10x12"],
        inplace=True,
    )

    return df_survey
