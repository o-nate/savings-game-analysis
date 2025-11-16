import numpy as np
import pandas as pd


def separate_inflation_measures(df_inf_measures: pd.DataFrame) -> pd.DataFrame:
    """Separates inflation measures by high- and low-inflation phases.

    Args:
        df_inf_measures: DataFrame with inflation measures.

    Returns:
        DataFrame with separated inflation measures.
    """
    df_inf_measures["inf_phase"] = np.where(
        df_inf_measures["inf_phase"] == 1,
        "high",
        "low",
    )

    df_bias = pd.pivot_table(
        data=df_inf_measures[
            [
                "participant.code",
                "inf_phase",
                "Perception_bias",
                "Expectation_bias",
                "Qual Perception",
                "Qual Expectation",
            ]
        ],
        index=["participant.code"],
        columns="inf_phase",
    )
    df_bias = df_bias.reset_index()
    df_bias.columns = df_bias.columns.map("_".join)
    df_bias.reset_index(inplace=True)

    df_bias.rename(columns={"participant.code_": "participant.code"}, inplace=True)

    return df_inf_measures.merge(df_bias, how="left")
