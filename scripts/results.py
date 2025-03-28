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

# %% [markdown]
## Descriptive statistics: Subjects
if not table_exists(con_exp_1, "Questionnaire"):
    create_duckdb_database(con_exp_1, experiment=1, initial_creation=True)
if not table_exists(con_exp_2, "Questionnaire"):
    create_duckdb_database(con_exp_2, experiment=2, initial_creation=True)

# %%
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

# * Filter for 4x30 inflation only
df_decisions_1 = df_decisions_1[df_decisions_1["participant.inflation"] == 430]

# * Store final savings at month t = 120
df_decisions_1["finalSavings_120"] = (
    df_decisions_1[df_decisions_1["Month"] == 120]
    .groupby("participant.code")["finalSavings"]
    .transform("mean")
)
df_decisions_1["finalSavings_120"] = df_decisions_1.groupby("participant.code")[
    "finalSavings_120"
].bfill()

df_decisions_1 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_1,
    estimate_measure="Quant Perception",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)
df_decisions_1 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_1,
    estimate_measure="Quant Expectation",
    decision_measure="finalStock",
    month=36,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)

pattern_cols = [
    "Quant Perception_pattern_12",
    "Quant Expectation_pattern_36",
]
performance_cols = ["early_%", "excess_%", "sreal_%"]

# * Performance measures
for p in pattern_cols:
    performance_results = create_performance_measures_table(
        df_decisions_1[
            (df_decisions_1["Month"] == 120)
            & (df_decisions_1["participant.round"] == 1)
        ],
        p,
        performance_cols,
    )
    print(f"--------------{p}--------------")
    print(performance_results)

pattern_changes = decision_patterns.compare_decision_pattern_changes(
    df_decisions_1,
    pattern_cols,
    performance_cols,
    True,
)

# %%
fig, axs = plt.subplots(2, 2, figsize=(15, 15), sharex=True, sharey=True)
for i, treatment in zip(range(2), ["intervention", "control"]):
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

for i, treatment in zip(range(2), ["intervention", "control"]):
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

# %%
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

df_decisions_2 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_2,
    estimate_measure="Quant Perception",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)
df_decisions_2 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_2,
    estimate_measure="Quant Expectation",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)
df_decisions_2 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_2,
    estimate_measure="Qual Expectation",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_12,
)
df_decisions_2 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_2,
    estimate_measure="Quant Expectation",
    decision_measure="finalStock",
    month=36,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)
df_decisions_2 = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_2,
    estimate_measure="Qual Expectation",
    decision_measure="finalStock",
    month=36,
    coherent_decision=0,
    threshold_estimate=QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_36,
)
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
        df_decisions_2[
            (df_decisions_2["Month"] == 120)
            & (df_decisions_2["participant.round"] == 1)
        ],
        p,
        performance_cols,
    )
    print(f"--------------{p}--------------")
    print(performance_results)

pattern_changes = decision_patterns.compare_decision_pattern_changes(
    df_decisions_2,
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
