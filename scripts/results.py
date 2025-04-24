"""Present results from both experiments"""

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import calc_opp_costs, decision_patterns, process_survey
from src.utils import exp_1_patches

from src.stats_analysis import (
    apply_statistical_test,
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.constants import (
    ANNUAL_INTEREST_RATE,
    QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_12,
    QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_36,
)
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_series, export_plot
from src.utils.plotting import annotate_2d_histogram, create_performance_measures_table
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)


# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

## Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

con_exp_1 = duckdb.connect(constants.EXP_1_DATABASE_FILE, read_only=False)
con_exp_2 = duckdb.connect(constants.EXP_2_DATABASE_FILE, read_only=False)

# ! Export plots
export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
FILE_PATH = Path(__file__).parents[1] / "results"

# %%
ACCURATE_PERCEPTIONS_THRESHOLD = 0.75
ACCURATE_QUALITATIVE_THRESHOLD = 3
COLS = [
    "participant.code",
    "treatment",
    "participant.round",
    "Month",
    "decision",
    "finalStock",
    "Perception_sensitivity",
    "perception_accuracy",
    "total_phase_purchases",
    "initial_stock",
    "constrained_purchases",
    "unconstrained_purchases",
    "unconstrained_purchases_12",
    "unconstrained_purchases_30",
    "unconstrained_purchases_42",
    "purchase_adaptation_30",
    "purchase_adaptation_12",
]

# %% [markdown]
if not table_exists(con_exp_1, "Questionnaire"):
    create_duckdb_database(con_exp_1, experiment=1, initial_creation=True)
if not table_exists(con_exp_2, "Questionnaire"):
    create_duckdb_database(con_exp_2, experiment=2, initial_creation=True)

# %% [markdown]
## Experiment 1
df_expectations = con_exp_1.sql("SELECT * FROM inf_expectation").df()
df_perceptions = con_exp_1.sql("SELECT * FROM inf_estimate").df()

df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con_exp_1, experiment=1)

df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

df_survey = exp_1_patches.create_survey_df(
    df_perceptions, df_expectations, include_inflation=True
)
df_survey = df_survey.drop("treatment", axis=1)

df_inf_measures = process_survey.pivot_inflation_measures(df_survey)

df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
df_inf_measures["participant.inflation"] = np.where(
    df_inf_measures["participant.inflation"] == "4x30", 430, 1012
)
df_decisions_1 = df_opp_cost.merge(df_inf_measures, how="left")
df_decisions_1 = df_decisions_1.merge(
    df_expectations[["participant.code", "participant.day"]], how="left"
)

# # * Filter for 4x30 inflation only
# df_decisions_1 = df_decisions_1[df_decisions_1["participant.inflation"] == 430]

# * Store final savings at month t = 120
df_decisions_1["finalSavings_120"] = (
    df_decisions_1[df_decisions_1["Month"] == 120]
    .groupby("participant.code")["finalSavings"]
    .transform("mean")
)
df_decisions_1["finalSavings_120"] = df_decisions_1.groupby("participant.code")[
    "finalSavings_120"
].bfill()

# %% [markdown]
### Classify behavioral patterns
for measure in [
    "Perception_bias",
    "Perception_sensitivity",
    "Expectation_bias",
    "Expectation_sensitivity",
]:
    df_decisions_1[measure] = df_decisions_1.groupby("participant.code")[
        measure
    ].bfill()

df_decisions_1["perception_accuracy"] = decision_patterns.classify_perception(
    df_decisions_1["Perception_sensitivity"], ACCURATE_PERCEPTIONS_THRESHOLD
)
for month in [12, 30]:
    df_decisions_1[f"purchase_adaptation_{month}"] = (
        decision_patterns.classify_purchase_adaptation(
            df_decisions_1, comparison_start_month=month
        )
    )
    df_decisions_1[f"decision_pattern_{month}"] = (
        decision_patterns.classify_new_decision_patterns(
            df_decisions_1, f"purchase_adaptation_{month}", "perception_accuracy"
        )
    )

df_decisions_1[
    (df_decisions_1["Month"] == 1)
    & (df_decisions_1["participant.round"] == 1)
    & (df_decisions_1["participant.inflation"] == 430)
].value_counts(
    [
        # "purchase_adaptation_12",
        # "purchase_adaptation_30",
        # "decision_pattern_12",
        "decision_pattern_30",
    ]
)

# %% [markdown]
## Experiment 2
df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con_exp_2, experiment=2)

df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

df_survey = process_survey.create_survey_df(include_inflation=True)
df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
df_inf_measures["participant.inflation"] = np.where(
    df_inf_measures["participant.inflation"] == "4x30", 430, 1012
)

# * Add uncertainty measure
df_inf_measures["Uncertain Expectation"] = process_survey.include_uncertainty_measure(
    df_inf_measures, "Quant Expectation", 1, 0
)
df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
    "participant.code"
)["Uncertain Expectation"].transform("mean")

df_decisions_2 = df_opp_cost.merge(df_inf_measures, how="left")
df_decisions_2["participant.day"] = df_decisions_2["participant.round"]

# * Filter for 4x30 inflation only
df_decisions_2 = df_decisions_2[df_decisions_2["participant.inflation"] == 430]

# * Store final savings at month t = 120
df_decisions_2["finalSavings_120"] = (
    df_decisions_2[df_decisions_2["Month"] == 120]
    .groupby("participant.code")["finalSavings"]
    .transform("mean")
)
df_decisions_2["finalSavings_120"] = df_decisions_2.groupby("participant.code")[
    "finalSavings_120"
].bfill()

# %% [markdown]
### Classify behavioral patterns
for measure in [
    "Perception_bias",
    "Perception_sensitivity",
    "Expectation_bias",
    "Expectation_sensitivity",
]:
    df_decisions_2[measure] = df_decisions_2.groupby("participant.code")[
        measure
    ].bfill()

df_decisions_2["perception_accuracy"] = decision_patterns.classify_perception(
    df_decisions_2["Perception_sensitivity"], ACCURATE_PERCEPTIONS_THRESHOLD
)

# * Include qualitative perceptions for Experiment 2
df_decisions_2["qualitative_perception_36"] = np.where(
    df_decisions_2["Month"] == 36, df_decisions_2["Qual Perception"], np.nan
)
df_decisions_2["qualitative_perception_36"] = (
    df_decisions_2["qualitative_perception_36"].bfill().ffill()
)
df_decisions_2["qualitative_perception_accuracy"] = (
    decision_patterns.classify_perception(
        df_decisions_2["qualitative_perception_36"], ACCURATE_QUALITATIVE_THRESHOLD
    )
)

for month in [12, 30]:
    df_decisions_2[f"purchase_adaptation_{month}"] = (
        decision_patterns.classify_purchase_adaptation(
            df_decisions_2, comparison_start_month=month
        )
    )
    for inflation_measure in ["perception_accuracy", "qualitative_perception_accuracy"]:
        df_decisions_2[f"decision_pattern_{month}_{inflation_measure}"] = (
            decision_patterns.classify_new_decision_patterns(
                df_decisions_2, f"purchase_adaptation_{month}", inflation_measure
            )
        )

df_decisions_2[
    (df_decisions_2["Month"] == 1) & (df_decisions_2["participant.round"] == 1)
].value_counts(
    [
        # "purchase_adaptation_12",
        # "purchase_adaptation_30",
        # "decision_pattern_12_perception_accuracy",
        # "decision_pattern_12_qualitative_perception_accuracy",
        # "decision_pattern_30_perception_accuracy",
        "decision_pattern_30_qualitative_perception_accuracy",
    ]
)

# %%
df_decisions_1["exp"] = 1
df_decisions_1 = df_decisions_1.rename(
    columns={"decision_pattern_30": "decision_pattern"}
)
df_decisions_1["qual_decision_pattern"] = np.nan
df_decisions_2["exp"] = 2
df_decisions_2 = df_decisions_2.rename(
    columns={
        "decision_pattern_30_perception_accuracy": "decision_pattern",
        "decision_pattern_30_qualitative_perception_accuracy": "qual_decision_pattern",
    }
)
cols = [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Mean Perception Bias",
    "Mean Expectation Bias",
]
new_cols = {
    "sreal_%": "Total performance (%)",
    "early_%": "Over-stocking (%)",
    "excess_%": "Wasteful-stocking (%)",
}
df_decisions_all = pd.concat(
    [
        df_decisions_1[
            (df_decisions_1["participant.round"] == 1)
            & (df_decisions_1["participant.day"] == 1)
        ],
        df_decisions_2[
            (df_decisions_2["participant.round"] == 1)
            & (df_decisions_1["participant.day"] == 1)
        ],
    ]
)

df_decisions_all[["Mean Perception Bias", "Mean Expectation Bias"]] = (
    df_decisions_all.groupby("participant.code")[
        ["Perception_bias", "Expectation_bias"]
    ].transform("mean")
)

summary = (
    df_decisions_all[df_decisions_all["Month"] == 120]
    .groupby(["exp", "participant.inflation"])[cols]
    .describe()[[(c, "mean") for c in cols]]
    .reset_index()
)

summary.columns = summary.columns.get_level_values(0)

summary = summary.rename(
    columns={
        **new_cols
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Mean Perception Bias": "Perception Bias",
            "Mean Expectation Bias": "Expectation Bias",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
        }
    }
)
summary[[c for c in new_cols.values()]] = summary[[c for c in new_cols.values()]] * 100
summary["Inflation"] = np.where(summary["Inflation"] == 430, "4x30", "10x12")

summary.groupby(["Experiment", "Inflation"]).describe()[
    [(c, "mean") for c in summary.columns[2:]]
]

# %% [markdown]
## Test difference between experiments
for measure in [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Mean Perception Bias",
    "Mean Expectation Bias",
]:
    result = apply_statistical_test(
        df_decisions_all[
            (df_decisions_all["Month"] == 120)
            & (df_decisions_all["participant.day"] == 1)
            & (df_decisions_all["participant.inflation"] == 430)
        ],
        measure_column=measure,
        group_column="exp",
        group1=1,
        group2=2,
        test="mannwhitneyu",
    )
    print(f"{measure} p-value: {result.pvalue}")
