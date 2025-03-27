"""Present results from experiment"""

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

from src import calc_opp_costs, discontinuity, intervention, econ_preferences, knowledge

from src.process_survey import (
    create_survey_df,
    include_inflation_measures,
    include_uncertainty_measure,
    pivot_inflation_measures,
)
from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_series, export_plot
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

## Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(constants.DATABASE_FILE, read_only=False)

# ! Export plots
export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
FILE_PATH = Path(__file__).parents[1] / "results"

# %% [markdown]
## Descriptive statistics: Subjects
if table_exists(con, "Questionnaire") == False:
    create_duckdb_database(con, initial_creation=True)
df_questionnaire = con.sql("SELECT * FROM Questionnaire").df()

df_questionnaire[
    [
        m
        for m in df_questionnaire
        if any(qm in m for qm in constants.QUESTIONNAIRE_MEASURES)
    ]
].describe().T[["mean", "std", "min", "50%", "max"]]


# %% [markdown]
## Behavior in the Savings Game
### Overall performance
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()
calc_opp_costs.plot_savings_and_stock(df_opp_cost, col="phase", palette="tab10")


# %% [markdown]
### Performance measures: Over- and wasteful-stocking and purchase adaptation
df_measures = discontinuity.purchase_discontinuity(
    df_opp_cost, constants.DECISION_QUANTITY, constants.WINDOW
)

## Set avg_q and avg_q_% as month=33 value
df_pivot_measures = pd.pivot_table(
    df_measures[df_measures["month"] == 33][["participant.code", "avg_q", "avg_q_%"]],
    index="participant.code",
)
df_pivot_measures.reset_index(inplace=True)
df_measures = df_measures[[m for m in df_measures.columns if "avg_q" not in m]].merge(
    df_pivot_measures, how="left"
)

## Rename columns for results table
df_measures.rename(
    columns={
        k: v
        for k, v in zip(
            constants.PERFORMANCE_MEASURES_OLD_NAMES
            + constants.PURCHASE_ADAPTATION_OLD_NAME,
            constants.PERFORMANCE_MEASURES_NEW_NAMES
            + constants.PURCHASE_ADAPTATION_NEW_NAME,
        )
    },
    inplace=True,
)
df_measures[(df_measures["month"] == 120) & (df_measures["phase"] == "pre")].describe()[
    constants.PERFORMANCE_MEASURES_NEW_NAMES + constants.PURCHASE_ADAPTATION_NEW_NAME
].T[["mean", "std", "min", "50%", "max"]]

# %%
df_pivot_measures = df_measures[
    (df_measures["month"] == 120) & (df_measures["participant.round"] == 1)
].melt(
    id_vars="participant.label",
    value_vars=constants.PERFORMANCE_MEASURES_NEW_NAMES
    + constants.PURCHASE_ADAPTATION_NEW_NAME,
    var_name="Performance measure",
    value_name="Percent of maximum",
)
df_pivot_measures["Percent of maximum"] = df_pivot_measures["Percent of maximum"] * 100

## Create figure and subplots to join both box plots
fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 5))

ax = sns.boxplot(
    data=df_pivot_measures[
        df_pivot_measures["Performance measure"].isin(
            constants.PERFORMANCE_MEASURES_NEW_NAMES
        )
    ],
    x="Performance measure",
    y="Percent of maximum",
    ax=ax,
)

df_pivot_measures["Percent of maximum"] = df_pivot_measures["Percent of maximum"] / 100
df_pivot_measures.rename(columns={"Percent of maximum": "Units of good"}, inplace=True)
bx = sns.boxplot(
    data=df_pivot_measures[
        df_pivot_measures["Performance measure"].isin(
            constants.PURCHASE_ADAPTATION_NEW_NAME
        )
    ],
    x="Performance measure",
    y="Units of good",
    ax=bx,
)
ax.set_xlabel(None)
bx.set_xlabel(None)
plt.tight_layout()

# ! Export plot
export_plot(FILE_PATH, "perfromance_measures.png", export_all_plots=export_all_plots)

plt.show()

# %% [markdown]
## Expectation and perception of inflation
### Quality of inflation expectations and perceptions and performance

df_survey = create_survey_df(include_inflation=True)

# * Plot estimates over time
estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
g = sns.relplot(
    data=df_survey[
        (df_survey["Measure"].isin(estimates)) & (df_survey["participant.round"] == 1)
    ],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Measure",
    style="Measure",
    kind="line",
)

## Adjust titles
(g.set_axis_labels("Month", "Inflation rate (%)").tight_layout(w_pad=0.5))

# ! Export plot
export_plot(FILE_PATH, "inflation_time_series.png", export_all_plots=export_all_plots)

plt.show()

# %%
df_inf_measures = pivot_inflation_measures(df_survey)
df_inf_measures = include_inflation_measures(df_inf_measures)


# %% [markdown]
#### Relationship between expectations, perceptions, and decisions
# (Difference between quantitative and qualitative estimates)
df_inf_measures.rename(columns={"Month": "month"}, inplace=True)
df_inf_measures = df_inf_measures.merge(
    df_measures[[m for m in df_measures.columns if "participant.inflation" not in m]],
    how="left",
)
df_inf_measures.rename(
    columns={
        k: v
        for k, v in zip(
            constants.PERFORMANCE_MEASURES_NEW_NAMES
            + constants.PURCHASE_ADAPTATION_NEW_NAME,
            constants.PERFORMANCE_MEASURES_OLD_NAMES
            + constants.PURCHASE_ADAPTATION_OLD_NAME,
        )
    },
    inplace=True,
)

# %%
## Separate inflation measures by high- and low-inflation
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
df_bias.head()

df_bias.rename(columns={"participant.code_": "participant.code"}, inplace=True)

df_inf_measures = df_inf_measures.merge(df_bias, how="left")

# %%
## Determine qualitative estimates accuracy
conditions_list = [
    (df_inf_measures["inf_phase"] == "high") & (df_inf_measures["Qual Perception"] > 1),
    (df_inf_measures["inf_phase"] == "low")
    & (df_inf_measures["Qual Perception"] >= 0)
    & (df_inf_measures["Qual Perception"] <= 1),
    df_inf_measures["Qual Perception"].isna(),
]
df_inf_measures["Qual Perception Accuracy"] = np.select(
    conditions_list, [1, 1, np.nan], default=0
)

conditions_list = [
    (df_inf_measures.groupby("participant.code")["inf_phase"].shift(-1) == "high")
    & (df_inf_measures["Qual Expectation"] > 1),
    (df_inf_measures.groupby("participant.code")["inf_phase"].shift(-1) == "low")
    & (df_inf_measures["Qual Expectation"] >= 0)
    & (df_inf_measures["Qual Expectation"] <= 1),
    df_inf_measures["Qual Expectation"].isna(),
]

df_inf_measures["Qual Expectation Accuracy"] = np.select(
    conditions_list, [1, 1, np.nan], default=0
)

df_accuracy = pd.pivot_table(
    df_inf_measures[
        ["participant.code", "Qual Perception Accuracy", "Qual Expectation Accuracy"]
    ],
    index="participant.code",
    aggfunc="mean",
)

df_accuracy.rename(
    columns={col: f"Avg {col}" for col in df_accuracy.columns}, inplace=True
)

df_inf_measures = df_inf_measures.merge(df_accuracy.reset_index(), how="left")

# * Add uncertainty measure
df_inf_measures["Uncertain Expectation"] = include_uncertainty_measure(
    df_inf_measures, "Quant Expectation", 1, 0
)
df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
    "participant.code"
)["Uncertain Expectation"].transform("mean")

df_inf_measures[df_inf_measures["participant.round"] == 1].describe()[
    constants.INFLATION_RESULTS_MEASURES[2:]
].T[["mean", "std", "min", "50%", "max"]]

# %%
create_pearson_correlation_matrix(
    df_inf_measures[
        (df_inf_measures["participant.round"] == 1) & (df_inf_measures["month"] == 120)
    ][
        constants.INFLATION_RESULTS_MEASURES[2:]
        + [
            constants.PURCHASE_ADAPTATION_OLD_NAME[0],
            "avg_q_%",
            "sreal",
        ]
    ],
    p_values=constants.P_VALUE_THRESHOLDS,
)


# %% [markdown]
## Real life vs. savings game
### Comparison to trends from surveys in real life
#### Figure 5 – Correlation between perceived and expected inflation (%) <br><br>
sns.lmplot(
    df_inf_measures[df_inf_measures["participant.round"] == 1],
    x="Quant Perception",
    y="Quant Expectation",
    hue="participant.round",
    legend=None,
)

# ! Export plot
export_plot(
    FILE_PATH,
    "perception_expectations_correlations.png",
    export_all_plots=export_all_plots,
)


# %%[markdown]
#### Correlation matrix of inflation measures versus actual inflation
create_pearson_correlation_matrix(
    df_inf_measures[df_inf_measures["participant.round"] == 1][
        [
            "Actual",
            "Upcoming",
            "Quant Perception",
            "Quant Expectation",
            "Qual Perception",
            "Qual Expectation",
        ]
    ],
    [0.1, 0.05, 0.01],
)

# %% [markdown]
#### Tableau 3 – Réponses à la question qualitative sur l’anticipation à un an <br><br>
df_inf_measures.groupby("inf_phase")[["Qual Expectation"]].value_counts(normalize=True)
df = df_inf_measures[df_inf_measures["participant.round"] == 1][
    ["participant.code", "inf_phase", "month", "Qual Perception", "Qual Expectation"]
].melt(
    id_vars=["participant.code", "inf_phase", "month"],
    value_vars=["Qual Perception", "Qual Expectation"],
    var_name="Estimate Type",
    value_name="Estimate",
)
sns.displot(
    df, x="Estimate", col="inf_phase", row="Estimate Type", bins=5, common_norm=True
)

# ! Export plot
export_plot(
    FILE_PATH,
    "per_period_distribution_of_qual_responses.png",
    export_all_plots=export_all_plots,
)


# %% [markdown]
#### Distribution of estimations table
df_inf_measures[df_inf_measures["participant.round"] == 1][
    ["inf_phase", "Quant Perception", "Quant Expectation"]
].groupby("inf_phase").describe()[
    [
        ("Quant Perception", "mean"),
        ("Quant Perception", "std"),
        ("Quant Expectation", "mean"),
        ("Quant Expectation", "std"),
    ]
].T

# %% [markdown]
#### Figure III – Distribution of perceived and expected inflaiton (% of respondents)
df = df_inf_measures[df_inf_measures["participant.round"] == 1][
    ["participant.code", "inf_phase", "month", "Quant Perception", "Quant Expectation"]
].melt(
    id_vars=["participant.code", "inf_phase", "month"],
    value_vars=["Quant Perception", "Quant Expectation"],
    var_name="Estimate Type",
    value_name="Estimate",
)
sns.displot(
    df, x="Estimate", hue="inf_phase", col="Estimate Type", kde=True, common_norm=False
)

# ! Export plot
export_plot(
    FILE_PATH, "distribution_of_qual_responses.png", export_all_plots=export_all_plots
)

# %%
# Figure V – Change in estimation uncertainty (% of responses)
df_uncertain = (
    pd.pivot_table(
        df_inf_measures[df_inf_measures["participant.round"] == 1][
            [
                "month",
                "Quant Expectation",
                "Uncertain Expectation",
                "Actual",
            ]
        ],
        index="month",
        aggfunc="mean",
    )
    .reset_index()
    .dropna()
)

df_uncertain["Uncertain Expectation"] = df_uncertain["Uncertain Expectation"] * 100

g = sns.relplot(
    df_uncertain.melt(
        id_vars="month",
        value_vars=["Quant Expectation", "Actual", "Uncertain Expectation"],
        var_name="Measure",
        value_name="Value",
    ),
    x="month",
    y="Value",
    hue="Measure",
    style="Measure",
    kind="line",
)

## Adjust titles
(
    g.set_axis_labels("Month", "Inflation rate (%)")
    # .set_titles("Savings Game round: {col_name}")
    .tight_layout(w_pad=0.5)
)
# plt.legend(loc="best")

# ! Export plot
export_plot(FILE_PATH, "uncertainty_time_series.png", export_all_plots=export_all_plots)

# %% [markdown]
## The role of individual characteristics and behavior
df_knowledge = knowledge.create_knowledge_dataframe()
df_econ_preferences = econ_preferences.create_econ_preferences_dataframe()
df_individual_char = combine_series(
    [df_inf_measures, df_knowledge, df_econ_preferences],
    how="left",
    on="participant.label",
)
## Set mean perception and expectation biases
df_individual_char["avg_perception_bias"] = df_individual_char.groupby(
    "participant.code"
)["Perception_bias"].transform("mean")
df_individual_char["avg_expectation_bias"] = df_individual_char.groupby(
    "participant.code"
)["Expectation_bias"].transform("mean")

# %% [markdown]
### Results of knowledge tasks
df_knowledge.describe().T[["mean", "std"]]

# %% [markdown]
### Results of economic preference tasks
df_econ_preferences.describe(percentiles=[0.5]).T[["mean", "std", "min", "50%", "max"]]

# %% [markdown]
### Correlations between knowledge and performance measures
data = df_individual_char[
    (df_individual_char["participant.round"] == 1)
    & (df_individual_char["month"] == 120)
]
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between economic preferences and performance measures
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES,
    constants.PERFORMANCE_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation bias and sensitivity measures
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.QUANT_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
    filtered_results=True,
)

# %% [markdown]
### Correlations between economic preferences and inflation bias and sensitivity measures
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[0:2],
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[2:4],
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[4:6],
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[6:],
    constants.QUANT_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.KNOWLEDGE_MEASURES,
    constants.QUAL_INFLATION_MEASURES,
    "pointbiserial",
    decimal_places=constants.DECIMAL_PLACES,
)

# %% [markdown]
### Correlations between knowledge and inflation qualitative inflation measures
# * Bonferroni correction
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[0:2],
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[2:4],
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[4:6],
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)
# %%
create_bonferroni_correlation_table(
    data,
    constants.ECON_PREFERENCE_MEASURES[6:],
    constants.QUAL_INFLATION_MEASURES,
    "pearson",
    decimal_places=constants.DECIMAL_PLACES,
)


# %% [markdown]
## Efficacy of interventions
data = df_individual_char[df_individual_char["month"] == 120]

# %% [markdown]
### Change in performance between first and second session (Learning effect)
df_learning_effect, df_learning_pivot = intervention.create_learning_effect_table(
    data, constants.PERFORMANCE_MEASURES, constants.P_VALUE_THRESHOLDS
)
df_learning_effect

# %% [markdown]
### Diff-in-diff of treatments
df_treatments = intervention.create_diff_in_diff_table(
    data,
    constants.PERFORMANCE_MEASURES,
    constants.TREATMENTS,
    constants.P_VALUE_THRESHOLDS,
)
df_treatments

# %% [markdown]
## Regressions
# * Pre-processing
# * Get average quantity purchased in each 12-month window
df_adapt = df_opp_cost.copy()
df_adapt["avg_purchase"] = df_adapt.groupby("participant.code")["decision"].transform(
    lambda x: x.rolling(12).mean()
)
df_inf_adapt = df_individual_char.copy()
df_inf_adapt = df_inf_adapt[
    [c for c in df_inf_adapt.columns if "cum_decision" not in c]
].merge(
    df_adapt[["participant.code", "cum_decision", "avg_purchase", "month"]], how="left"
)

# * Get previous window's inflation expectation
df_inf_adapt["previous_expectation"] = df_inf_adapt.groupby("participant.code")[
    "Quant Expectation"
].shift(1)
df_inf_adapt["previous_qual_expectation"] = df_inf_adapt.groupby("participant.code")[
    "Qual Expectation"
].shift(1)
df_inf_adapt["previous_qual_expectation_accuracy"] = df_inf_adapt.groupby(
    "participant.code"
)["Qual Expectation"].shift(1)

df_inf_adapt.rename(
    columns={
        "Quant Perception": "current_perception",
        "Qual Perception": "current_qual_perception",
    },
    inplace=True,
)

# * Replace qualitative estimates with boolean for stay the same/decrease or increase
# * (see Andrade et al. (2023))
df_inf_adapt["current_qual_perception"] = np.where(
    df_inf_adapt["current_qual_perception"] <= 0, 0, 1
)
df_inf_adapt["previous_qual_expectation"] = np.where(
    df_inf_adapt["previous_qual_expectation"] <= 0, 0, 1
)

# * Set qualitative estimates as ordinal variables
df_inf_adapt["current_qual_perception"] = pd.Categorical(
    df_inf_adapt["current_qual_perception"],
    ordered=True,
    categories=[0, 1],
)
df_inf_adapt["previous_qual_expectation"] = pd.Categorical(
    df_inf_adapt["previous_qual_expectation"],
    ordered=True,
    categories=[0, 1],
)
df_inf_adapt["previous_qual_expectation_accuracy"] = pd.Categorical(
    df_inf_adapt["previous_qual_expectation_accuracy"],
    ordered=True,
    categories=[0, 1],
)

assert (
    df_inf_adapt.shape[0] == df_individual_char.shape[0]
    and df_inf_adapt.shape[1] == df_individual_char.shape[1] + 4
)
_, df_performance_pivot = intervention.create_learning_effect_table(
    df_inf_adapt,
    constants.PERFORMANCE_MEASURES
    + constants.QUAL_INFLATION_MEASURES
    + constants.QUANT_INFLATION_MEASURES,
    constants.P_VALUE_THRESHOLDS,
)
df_performance_pivot.rename(
    columns={
        "Change in sreal_%": "diff_performance",
        "Change in early_%": "diff_early",
        "Change in excess_%": "diff_excess",
        "Change in Avg Qual Expectation Accuracy": "diff_avg_qual_exp",
        "Change in Avg Qual Perception Accuracy": "diff_avg_qual_perc",
        "Change in Average Uncertain Expectation": "diff_avg_uncertainty",
        "Change in Perception_sensitivity": "diff_perception_sensitivity",
        "Change in avg_perception_bias": "diff_perception_bias",
        "Change in Expectation_sensitivity": "diff_expectation_sensitivity",
        "Change in avg_expectation_bias": "diff_expectation_bias",
    },
    inplace=True,
)
df_performance_pivot = df_performance_pivot[
    [
        "participant.label",
        "diff_performance",
        "diff_early",
        "diff_excess",
        "diff_avg_qual_perc",
        "diff_avg_qual_exp",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ]
]
df_performance_pivot.columns = df_performance_pivot.columns.droplevel()
df_performance_pivot.columns = [
    "participant.label",
    "diff_performance",
    "diff_early",
    "diff_excess",
    "diff_avg_qual_perc",
    "diff_avg_qual_exp",
    "diff_avg_uncertainty",
    "diff_perception_sensitivity",
    "diff_perception_bias",
    "diff_expectation_sensitivity",
    "diff_expectation_bias",
]

df_inf_adapt = df_inf_adapt.merge(df_performance_pivot, how="left")

# %% [markdown]
### Do subjects account for inflation in their purchase adaptation?
#### Without treatment (only round 1)
df_inf_adapt.rename(
    columns={
        "Uncertain Expectation": "uncertainty",
        "Avg Qual Perception Accuracy": "Avg_Qual_Perception_Accuracy",
        "Avg Qual Expectation Accuracy": "Avg_Qual_Expectation_Accuracy",
        "Qual Perception Accuracy": "Qual_Perception_Accuracy",
    },
    inplace=True,
)

regressions = {}

for m in constants.ADAPTATION_MONTHS:
    model = smf.ols(
        formula="""avg_purchase ~ Actual + current_perception + previous_expectation \
                + current_qual_perception + previous_qual_expectation + C(uncertainty)""",
        data=df_inf_adapt[
            (df_inf_adapt["phase"] == "pre") & (df_inf_adapt["month"] == m)
        ],
    )
    regressions[f"Month {m}"] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results


# %% [markdown]
#### With treatment
regressions = {}

for m in constants.ADAPTATION_MONTHS:
    model = smf.ols(
        formula="""avg_purchase ~ Actual  \
            + C(treatment) * (C(phase) + current_perception + previous_expectation \
                + current_qual_perception + previous_qual_expectation + C(uncertainty))""",
        data=df_inf_adapt[df_inf_adapt["month"] == m],
    )
    regressions[f"Month {m}"] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
### Comparison of regressions of performance measures
data = df_inf_adapt.copy()

# * Add columns for expectations in month 1
data["quant_expectation_month_1"] = data.groupby("participant.code")[
    "Quant Expectation"
].transform("first")
data["qual_expectation_month_1"] = data.groupby("participant.code")[
    "Qual Expectation"
].transform("first")

data["qual_expectation_month_1"] = pd.Categorical(
    data["qual_expectation_month_1"],
    ordered=True,
    categories=[0, 1],
)

data.rename(
    columns={
        "Avg Qual Expectation Accuracy": "Avg_Qual_Expectation_Accuracy",
        "Avg Qual Perception Accuracy": "Avg_Qual_Perception_Accuracy",
        "Average Uncertain Expectation": "Average_Uncertain_Expectation",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
    inplace=True,
)

# %% [markdown]
#### Benchmark regression
regressions = {}

for m in ["sreal_percent", "early_percent", "excess_percent"]:
    model = smf.ols(
        formula=f"""{m} ~ Expectation_sensitivity + avg_expectation_bias\
            + Perception_sensitivity + avg_perception_bias""",
        data=data[(data["phase"] == "pre") & (data["month"] == 120)],
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
#### Compared to model with qualitative instead of quantitative
regressions = {}

for m in ["sreal_percent", "early_percent", "excess_percent"]:
    model = smf.ols(
        formula=f"""{m} ~ Avg_Qual_Expectation_Accuracy
                + Avg_Qual_Perception_Accuracy + Average_Uncertain_Expectation""",
        data=data[(data["phase"] == "pre") & (data["month"] == 120)],
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %%
#### Change in performance

regressions = {}

for m in [
    "diff_performance",
    "diff_early",
    "diff_excess",
    "diff_avg_qual_perc",
    "diff_avg_qual_exp",
    "diff_avg_uncertainty",
    "diff_perception_sensitivity",
    "diff_perception_bias",
    "diff_expectation_sensitivity",
    "diff_expectation_bias",
]:
    model = smf.ols(
        formula=f"{m} ~ C(treatment)",
        data=data[data["month"] == 120],
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
### Impact of individual characteristics
#### From round 1
regressions = {}
data = df_inf_adapt[(df_inf_adapt["phase"] == "pre") & (df_inf_adapt["month"] == 120)]
data.rename(
    columns={
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
    inplace=True,
)

for m in [
    "sreal_percent",
    "early_percent",
    "excess_percent",
]:
    model = smf.ols(
        formula=f"{m} ~ financial_literacy + numeracy + compound + wisconsin_choice_count \
            + wisconsin_SE + wisconsin_PE + riskPreferences_choice_count \
                + riskPreferences_switches + lossAversion_choice_count \
                    + lossAversion_switches + timePreferences_choice_count \
                        + timePreferences_switches",
        data=data,
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %%
#### With treatment
regressions = {}
data = df_inf_adapt[(df_inf_adapt["phase"] == "post") & (df_inf_adapt["month"] == 120)]

for m in [
    "diff_performance",
    "diff_early",
    "diff_excess",
]:
    model = smf.ols(
        formula=f"{m} ~ C(treatment) : (financial_literacy + numeracy \
            + compound + wisconsin_choice_count \
            + wisconsin_SE + wisconsin_PE + riskPreferences_choice_count \
                + riskPreferences_switches + lossAversion_choice_count \
                    + lossAversion_switches + timePreferences_choice_count \
                        + timePreferences_switches)",
        data=data,
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
##### With forward selection
df_qs = df_questionnaire.copy()
df_qs.columns = [c.replace("Questionnaire.1.player.", "") for c in df_qs.columns]

regressions = {}

df_forward = df_inf_adapt[df_inf_adapt["month"] == 120].copy()
df_forward["n_switches"] = df_forward[
    [c for c in df_forward.columns if "_switches" in c]
].sum(axis=1)
df_forward = df_forward.merge(
    df_qs[["participant.label"] + constants.FEATURES[-6:-1]], how="left"
)

data = df_forward[df_forward["phase"] == "pre"]
data.rename(
    columns={
        "Avg Qual Expectation Accuracy": "Avg_Qual_Expectation_Accuracy",
        "Avg Qual Perception Accuracy": "Avg_Qual_Perception_Accuracy",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
    inplace=True,
)
data = data[["sreal_percent", "early_percent", "excess_percent"] + constants.FEATURES]
data.rename(
    columns={
        "Avg Qual Expectation Accuracy": "Avg_Qual_Expectation_Accuracy",
        "Avg Qual Perception Accuracy": "Avg_Qual_Perception_Accuracy",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
    inplace=True,
)
for measure in ["sreal_percent", "early_percent", "excess_percent"]:
    ## Remove performance measures not needed for current regression
    not_measures = [
        m
        for m in ["sreal_percent", "early_percent", "excess_percent"]
        if m is not measure
    ]
    regressions[measure] = run_forward_selection(
        data[[c for c in data.columns if c not in not_measures]],
        response=measure,
        categoricals=["compound", "numeracy", "financial_literacy"],
    )
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
##### Forward selection
regressions = {}
data = df_forward[df_forward["phase"] == "post"].copy()
data = data[
    ["treatment", "diff_performance", "diff_early", "diff_excess"] + constants.FEATURES
]
for measure in ["diff_performance", "diff_early", "diff_excess"]:
    ## Remove performance measures not needed for current regression
    not_measures = [
        m for m in ["diff_performance", "diff_early", "diff_excess"] if m is not measure
    ]
    regressions[measure] = run_treatment_forward_selection(
        data[[c for c in data.columns if c not in not_measures]],
        response=measure,
        treatment="treatment",
        categoricals=["compound", "numeracy", "financial_literacy"],
    )
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
#### Mediation analysis
# * Replace treatment with dummy categories
criteria = [
    df_inf_adapt["treatment"] == "Intervention 1",
    df_inf_adapt["treatment"] == "Intervention 2",
]
choices = [1, 2]
df_inf_adapt["control"] = np.where(df_inf_adapt["treatment"] == "Control", 1, 0)
df_inf_adapt["intervention_1"] = np.where(
    df_inf_adapt["treatment"] == "Intervention 1", 1, 0
)
df_inf_adapt["intervention_2"] = np.where(
    df_inf_adapt["treatment"] == "Intervention 2", 1, 0
)
data = df_inf_adapt[df_inf_adapt["month"] == 120]
print(constants.MEDIATION_CONTROL)
mediation_analysis(
    data,
    x=constants.MEDIATION_CONTROL,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_qual_exp",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)

# %%
print(constants.MEDIATION_INTERVENTION_1)
mediation_analysis(
    data,
    x=constants.MEDIATION_INTERVENTION_1,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_qual_exp",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)

# %%
print(constants.MEDIATION_INTERVENTION_2)
mediation_analysis(
    data,
    x=constants.MEDIATION_INTERVENTION_2,
    m=[
        "diff_avg_qual_perc",
        "diff_avg_qual_exp",
        "diff_avg_uncertainty",
        "diff_perception_sensitivity",
        "diff_perception_bias",
        "diff_expectation_sensitivity",
        "diff_expectation_bias",
    ],
    y="diff_performance",
    alpha=0.05,
    seed=42,
)
