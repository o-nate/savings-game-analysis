"""Classify subjects by pattern of perceptions/expectations and decisions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from src import calc_opp_costs, process_survey

from src.utils.constants import (
    ANNUAL_INTEREST_RATE,
    PERSONAS,
    QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_12,
    QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_36,
)
from src.utils.plotting import (
    annotate_2d_histogram,
    create_performance_measures_table,
    visualize_persona_results,
)
from utils.logging_config import get_logger

# * Set logger
logger = get_logger(__name__)


def classify_subject_decision_patterns(
    data: pd.DataFrame,
    estimate_measure: str,
    decision_measure: str,
    month: int,
    coherent_decision: int,
    threshold_estimate: int | float,
) -> pd.DataFrame:
    if estimate_measure == "Quant Perception":
        months = [month]
    else:
        start_month = max(1, month - 12)
        months = [start_month, month]
    logger.debug("months: %s", months)
    df = data[data["Month"].isin(months)]
    measure = estimate_measure
    d_measure = decision_measure
    if estimate_measure != "Quant Perception" and month == 12:
        measure = f"previous_{estimate_measure}"
        logger.info("Setting measure to: %s", measure)
        df[measure] = df.groupby("participant.code")[estimate_measure].shift(1)
    if estimate_measure != "Quant Perception" and month > 12:
        d_measure = f"change_in_stock"
        logger.info("Setting decision measure to: %s", d_measure)
        df[f"previous_{decision_measure}"] = df.groupby("participant.code")[
            decision_measure
        ].shift(1)
        df = df[df["Month"] == month]
        df[d_measure] = df[decision_measure] - df[f"previous_{decision_measure}"]

    patterns = define_decision_patterns(
        df,
        estimate_measure=measure,
        decision_measure=d_measure,
        threshold_estimate=threshold_estimate,
        coherent_decision=coherent_decision,
        month=month,
    )
    personas = patterns.keys()
    conditions = patterns.values()
    new_column_name = f"{estimate_measure}_pattern_{month}"
    df[new_column_name] = np.select(
        condlist=conditions, choicelist=personas, default="N/A"
    )
    data = data.merge(
        df[["participant.code", "treatment", new_column_name]], how="left"
    )
    logger.debug("new column: %s", new_column_name)
    return data[data[new_column_name] != "N/A"]


def define_decision_patterns(
    data: pd.DataFrame,
    estimate_measure: str,
    decision_measure: str,
    threshold_estimate: int | float,
    coherent_decision: str,
    month: int,
) -> dict[str, list]:
    if month <= 12:
        conditions = [
            # Accurate estimate & Coherent decision
            (data[estimate_measure] <= threshold_estimate)
            & (data[decision_measure] <= coherent_decision),
            # Accurate estimate & Incoherent decision
            (data[estimate_measure] <= threshold_estimate)
            & (data[decision_measure] > coherent_decision),
            # Inaccurate estimate & Coherent decision
            (data[estimate_measure] > threshold_estimate)
            & (data[decision_measure] > coherent_decision),
            # Inaccurate estimate & Incoherent decision
            (data[estimate_measure] > threshold_estimate)
            & (data[decision_measure] <= coherent_decision),
        ]
    if month > 12:
        conditions = [
            # Accurate estimate & Coherent decision
            (data[estimate_measure] > threshold_estimate)
            & (data[decision_measure] > coherent_decision),
            # Accurate estimate & Incoherent decision
            (data[estimate_measure] > threshold_estimate)
            & (data[decision_measure] <= coherent_decision),
            # Inaccurate estimate & Coherent decision
            (data[estimate_measure] <= threshold_estimate)
            & (data[decision_measure] <= coherent_decision),
            # Inaccurate estimate & Incoherent decision
            (data[estimate_measure] <= threshold_estimate)
            & (data[decision_measure] > coherent_decision),
        ]
    return {persona: condition for persona, condition in zip(PERSONAS, conditions)}


def compare_decision_pattern_changes(
    data: pd.DataFrame,
    pattern_columns: list[str],
    performance_measure_columns: list[str],
    make_categorical: bool = True,
) -> pd.DataFrame:
    value_columns = pattern_columns + performance_measure_columns
    pivot_table = pd.pivot_table(
        data[data["Month"] == 120],
        values=value_columns,
        index=["participant.label", "treatment"],
        columns="participant.round",
        aggfunc="first",
    )

    for col in value_columns:
        if "pattern" in col:
            pivot_table[f"change_{col}"] = (
                pivot_table[(col, 1.0)] + pivot_table[(col, 2.0)]
            )
        if "_%" in col:
            pivot_table[f"change_{col}"] = (
                pivot_table[(col, 2.0)] - pivot_table[(col, 1.0)]
            )
    if make_categorical:
        old_columns = [c for c in pivot_table.columns if c[1] != ""]

        for col in old_columns:
            if col[0] in pattern_columns:
                pivot_table[col] = pd.Categorical(pivot_table[col], PERSONAS)

    return pivot_table


def main() -> None:
    """Run script"""
    df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

    df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
    df_opp_cost.head()

    df_survey = process_survey.create_survey_df(include_inflation=True)
    df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
    df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
    df_inf_measures["participant.inflation"] = np.where(
        df_inf_measures["participant.inflation"] == "4x30", 430, 1012
    )

    # * Add uncertainty measure
    df_inf_measures["Uncertain Expectation"] = (
        process_survey.include_uncertainty_measure(
            df_inf_measures, "Quant Expectation", 1, 0
        )
    )
    df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
        "participant.code"
    )["Uncertain Expectation"].transform("mean")

    df_decisions = df_opp_cost.merge(df_inf_measures, how="left")

    # * Store final savings at month t = 120
    df_decisions["finalSavings_120"] = (
        df_decisions[df_decisions["Month"] == 120]
        .groupby("participant.code")["finalSavings"]
        .transform("mean")
    )
    df_decisions["finalSavings_120"] = df_decisions.groupby("participant.code")[
        "finalSavings_120"
    ].bfill()

    logger.debug(df_decisions.shape)
    df_decisions = classify_subject_decision_patterns(
        data=df_decisions,
        estimate_measure="Quant Perception",
        decision_measure="finalStock",
        month=12,
        coherent_decision=0,
        threshold_estimate=ANNUAL_INTEREST_RATE,
    )
    logger.debug(df_decisions.shape)
    df_decisions = classify_subject_decision_patterns(
        data=df_decisions,
        estimate_measure="Quant Expectation",
        decision_measure="finalStock",
        month=12,
        coherent_decision=0,
        threshold_estimate=ANNUAL_INTEREST_RATE,
    )
    logger.debug(df_decisions.shape)
    df_decisions = classify_subject_decision_patterns(
        data=df_decisions,
        estimate_measure="Qual Expectation",
        decision_measure="finalStock",
        month=12,
        coherent_decision=0,
        threshold_estimate=QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_12,
    )
    logger.debug(df_decisions.shape)
    df_decisions = classify_subject_decision_patterns(
        data=df_decisions,
        estimate_measure="Quant Expectation",
        decision_measure="finalStock",
        month=36,
        coherent_decision=0,
        threshold_estimate=ANNUAL_INTEREST_RATE,
    )
    logger.debug(df_decisions.shape)
    df_decisions = classify_subject_decision_patterns(
        data=df_decisions,
        estimate_measure="Qual Expectation",
        decision_measure="finalStock",
        month=36,
        coherent_decision=0,
        threshold_estimate=QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_36,
    )
    logger.debug(df_decisions.shape)
    pattern_cols = [
        "Quant Perception_pattern_12",
        "Quant Expectation_pattern_12",
        "Qual Expectation_pattern_12",
        "Quant Expectation_pattern_36",
        "Qual Expectation_pattern_36",
    ]
    performance_cols = ["early_%", "excess_%", "sreal_%"]

    # * Performance measures
    for p in pattern_cols:
        performance_results = create_performance_measures_table(
            df_decisions[
                (df_decisions["Month"] == 120)
                & (df_decisions["participant.round"] == 1)
            ],
            p,
            performance_cols,
        )
        print(f"--------------{p}--------------")
        print(performance_results)

    pattern_changes = compare_decision_pattern_changes(
        df_decisions,
        pattern_cols,
        performance_cols,
        True,
    )

    fig, axs = plt.subplots(3, 3, figsize=(15, 15), sharex=True, sharey=True)
    for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
        subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

        axs[0][i] = sns.histplot(
            subset,
            x=("Quant Perception_pattern_12", 1),
            y=("Quant Perception_pattern_12", 2),
            ax=axs[0][i],
            cbar=False,
        )

        axs[0][i] = annotate_2d_histogram(
            axs[0][i],
            ("Quant Perception_pattern_12", 1),
            ("Quant Perception_pattern_12", 2),
            subset,
            # fontweight="bold",
            color="black",
            fontsize=12,
        )
        axs[0][i].set_xlabel("Round 1")
        axs[0][i].set_ylabel("Round 2")
        axs[0][i].set_title(treatment)

    for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
        subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

        axs[1][i] = sns.histplot(
            subset,
            x=("Quant Expectation_pattern_36", 1),
            y=("Quant Expectation_pattern_36", 2),
            ax=axs[1][i],
            cbar=False,
        )

        axs[1][i] = annotate_2d_histogram(
            axs[1][i],
            ("Quant Expectation_pattern_36", 1),
            ("Quant Expectation_pattern_36", 2),
            subset,
            # fontweight="bold",
            color="black",
            fontsize=12,
        )
        axs[1][i].set_xlabel("Round 1")
        axs[1][i].set_ylabel("Round 2")
        axs[1][i].set_title(treatment)

    for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
        subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

        axs[2][i] = sns.histplot(
            subset,
            x=("Qual Expectation_pattern_36", 1),
            y=("Qual Expectation_pattern_36", 2),
            ax=axs[2][i],
            cbar=False,
        )

        axs[2][i] = annotate_2d_histogram(
            axs[2][i],
            ("Qual Expectation_pattern_36", 1),
            ("Qual Expectation_pattern_36", 2),
            subset,
            # fontweight="bold",
            color="black",
            fontsize=12,
        )
        axs[2][i].set_xlabel("Round 1")
        axs[2][i].set_ylabel("Round 2")
        axs[2][i].set_title(treatment)

    plt.show()


if __name__ == "__main__":
    main()
