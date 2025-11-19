"""Present results from both experiments"""

# %%
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from scipy.stats import pearsonr
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import (
    calc_opp_costs,
    decision_patterns,
    econ_preferences,
    intervention,
    knowledge,
    process_survey,
)
from src.utils import ancova_restructurer, exp_1_patches

from src.stats_analysis import (
    apply_statistical_test,
    create_bonferroni_correlation_table,
    create_dynamic_correlation_matrix,
)
from src.utils.constants import ANNUAL_INTEREST_RATE
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_mean_std_dicts, combine_series, export_plot
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

DATA_FILE_PATH = Path(__file__).parents[1] / "data"

# ! Export plots
# export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
EXPORT_FILE_PATH = Path(__file__).parents[1] / "results"

# %%
QUESTIONNAIRE_COLS = [
    "Questionnaire.1.player.age",
    "Questionnaire.1.player.gender",
    "Questionnaire.1.player.educationLevel",
    "Questionnaire.1.player.employmentStatus",
    "Questionnaire.1.player.financialStatusIncome",
    "Questionnaire.1.player.financialStatusSavings_1",
    "Questionnaire.1.player.financialStatusSavings_2",
    "Questionnaire.1.player.financialStatusDebt_1",
    "Questionnaire.1.player.savingsAccounts",
    "Questionnaire.1.player.retirementAccounts",
]
ACCURATE_PERCEPTIONS_THRESHOLD = 0.75
ACCURATE_QUALITATIVE_THRESHOLD = 3
PARTICIPANT_WITHOUT_BELIEF_DATA = "JKmBvh7"
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
PERFORMANCE_COLS = [
    "sreal_%",
    "early_%",
    "late_%",
    "excess_%",
    "Perception_sensitivity",
    "Perception_bias_low",
    "Perception_bias_high",
    "Expectation_sensitivity",
    "Expectation_bias_low",
    "Expectation_bias_high",
]
BEHAVIOR_COLS = {
    "sreal_%": "Total performance (%)",
    "early_%": "Over-stocking (%)",
    "late_%": "Under-stocking (%)",
    "excess_%": "Wasteful-stocking (%)",
}
CLASSIFICATION_COLS = [
    "Month",
    "sreal_%",
    "early_%",
    "excess_%",
]
NEW_CLASSIFICIATION_COLS = {
    "Month": "Proportion (%)",
    "sreal_%": "Total performance (%)",
    "early_%": "Over-stocking (%)",
    "excess_%": "Wasteful-stocking (%)",
}
CORRELATION_COLS = [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Expectation_sensitivity",
    "purchase_adaptation_30",
    "Quant Perception_consistent_12",
    "financial_literacy",
    "numeracy",
    "compound",
    "wisconsin_choice_count",
    "timePreferences_choice_count",
    "n_switches",
    "lossAversion_choice_count",
    "riskPreferences_choice_count",
]
TREATMENT_GROUPS = [
    "Intervention 1",
    "Intervention 2",
    "Intervention 3",
]
LOGIT_COLS = [
    "purchase_adaptation_30",
    "perception_consistent_12",
    "decision_pattern_30_perception_accuracy_SN",
    "decision_pattern_30_perception_accuracy_SA",
    "decision_pattern_30_perception_accuracy_IN",
    "decision_pattern_30_perception_accuracy_IA",
    "perception_pattern_12_AC",
    "perception_pattern_12_AI",
    "perception_pattern_12_IC",
]


# %%
if not table_exists(con_exp_1, "Questionnaire"):
    create_duckdb_database(con_exp_1, experiment=1, initial_creation=True)
if not table_exists(con_exp_2, "Questionnaire"):
    create_duckdb_database(con_exp_2, experiment=2, initial_creation=True)

# %% [markdown]
## Experiment 1
df_questionnaire_1 = con_exp_1.sql("SELECT * FROM Questionnaire").df()

df_questionnaire_1[QUESTIONNAIRE_COLS].describe()

# %% [markdown]
### Questionnaire - education levels
df_questionnaire_1.value_counts("Questionnaire.1.player.educationLevel") / len(
    df_questionnaire_1
) * 100

# %% [markdown]
### Questionnaire - employment

df_questionnaire_1.value_counts("Questionnaire.1.player.employmentStatus") / len(
    df_questionnaire_1
) * 100

# %% [markdown]
### Average final remuneration (with participant fee)
df_pay = con_exp_1.sql("SELECT * FROM final_payments").df()
df_pay = df_pay[
    df_pay["participant.label"].isin(df_questionnaire_1["participant.label"])
]
assert len(df_pay) == len(df_questionnaire_1)
float(df_pay["participant.payoff"].mean())

# %%
df_expectations = con_exp_1.sql("SELECT * FROM inf_expectation").df()
df_perceptions = con_exp_1.sql("SELECT * FROM inf_estimate").df()

df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con_exp_1, experiment=1)

df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

df_survey_1 = exp_1_patches.create_survey_df(
    df_perceptions, df_expectations, include_inflation=True
)
df_survey_1 = df_survey_1.drop("treatment", axis=1)

df_inf_measures = process_survey.pivot_inflation_measures(df_survey_1)

df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
df_inf_measures["participant.inflation"] = np.where(
    df_inf_measures["participant.inflation"] == "4x30", 430, 1012
)
df_decisions_1 = df_opp_cost.merge(df_inf_measures, how="left")
df_decisions_1 = df_decisions_1.merge(
    df_expectations[["participant.code", "participant.day"]], how="left"
)

df_decisions_1 = process_survey.separate_inflation_bias_by_phase(df_decisions_1)

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
## Experiment 2
df_questionnaire = con_exp_2.sql("SELECT * FROM Questionnaire").df()
df_questionnaire = df_questionnaire[
    df_questionnaire["participant.label"] != PARTICIPANT_WITHOUT_BELIEF_DATA
]

df_questionnaire[QUESTIONNAIRE_COLS].describe()

# %% [markdown]
### Questionnaire - education levels
df_questionnaire.value_counts("Questionnaire.1.player.educationLevel") / len(
    df_questionnaire
) * 100

# %% [markdown]
### Questionnaire - employment

df_questionnaire.value_counts("Questionnaire.1.player.employmentStatus") / len(
    df_questionnaire
) * 100

# %% [markdown]
### Average final remuneration (with participant fee)
df_pay = con_exp_2.sql("SELECT * FROM final_payments").df()
df_pay = df_pay[df_pay["participant.label"].isin(df_questionnaire["participant.label"])]
assert len(df_pay) == len(df_questionnaire)
float(
    df_pay["participant.payoff"].mean() / 750 + 5
)  # Participant fee of €5 and conversion rate of 750 points/€

# %% [markdown]
### Demographics of combined experiments
df_demographics = pd.concat(
    [df_questionnaire_1[QUESTIONNAIRE_COLS], df_questionnaire[QUESTIONNAIRE_COLS]]
).reset_index()
df_demographics = df_demographics.drop(columns="index")

for demographic in [
    "Questionnaire.1.player.educationLevel",
    "Questionnaire.1.player.employmentStatus",
]:
    print(df_demographics.value_counts(demographic) / len(df_demographics) * 100, "\n")

df_demographics.describe()

# %% [markdown]
## Inflation estimations
df_real_inflation = pd.read_csv(DATA_FILE_PATH / "France HICP_20251112134158.csv")
df_real_inflation = df_real_inflation.drop(columns=["TIME PERIOD"])
df_real_inflation["DATE"] = pd.to_datetime(df_real_inflation["DATE"])
df_real_inflation = df_real_inflation.rename(
    columns={
        "HICP - Overall index - France (ICP.M.FR.N.000000.4.ANR)": "HICP (%)",
        "DATE": "Date",
    }
)
title_fontsize = 20
label_fontsize = 16
legend_fontsize = 14
tick_fontsize = 12

fig, ax = plt.subplots(figsize=(13, 5))
sns.lineplot(data=df_real_inflation, x="Date", y="HICP (%)", ax=ax)
ax.axvline(x=pd.to_datetime("2023-02-28"), color="r", linestyle="--", label="Exp 1")
ax.axvline(x=pd.to_datetime("2024-05-31"), color="g", linestyle="--", label="Exp 2")
ax.axhline(y=0, color="black", linestyle="-")
ax.set_title("HICP (% Change) - France, Monthly", fontsize=title_fontsize)
ax.set_xlabel("Date", fontsize=label_fontsize)
ax.set_ylabel("HICP (%)", fontsize=label_fontsize)
ax.tick_params(axis="x", labelsize=tick_fontsize)
ax.tick_params(axis="y", labelsize=tick_fontsize)
ax.legend(fontsize=legend_fontsize)
plt.show()

# %%

df_inflation_estimates_1 = con_exp_1.sql("SELECT * FROM Inflation").df()
df_inflation_estimates_1 = df_inflation_estimates_1[
    [
        c
        for c in df_inflation_estimates_1.columns
        if ("infK_" in c)
        and (c not in ["Inflation.1.player.infK_4", "Inflation.1.player.infK_5"])
    ]
]

df_inflation_estimates_1["experiment"] = 1

df_inflation_estimates_2 = con_exp_2.sql("SELECT * FROM Inflation").df()
df_inflation_estimates_2 = df_inflation_estimates_2[
    df_inflation_estimates_2["participant.label"] != PARTICIPANT_WITHOUT_BELIEF_DATA
]
df_inflation_estimates_2 = df_inflation_estimates_2[
    [c for c in df_inflation_estimates_2.columns if "infK_" in c]
]
df_inflation_estimates_2["experiment"] = 2

df_inflation_estimates_1.columns = df_inflation_estimates_2.columns

df_inflation_estimates = pd.concat([df_inflation_estimates_1, df_inflation_estimates_2])

for measure in df_inflation_estimates.columns:
    result = apply_statistical_test(df_inflation_estimates, measure, "experiment", 1, 2)
    print(f"{measure} p-value: {result.pvalue}")

results = df_inflation_estimates.groupby("experiment").mean()

print(
    """\n*Differences in inflation estimates between experiments are statically significant
to the p<0.01 level, except for the estimate of the lowest inflation rate in the
last 30 years.*"""
)
results.T

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


# %%
## Separate inflation measures by high- and low-inflation
df_decisions_2 = df_opp_cost.merge(df_inf_measures, how="left")
df_decisions_2 = process_survey.separate_inflation_bias_by_phase(df_decisions_2)

# %%
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

# %%
df_decisions_1["exp"] = 1
df_decisions_2["exp"] = 2

df_decisions_all = pd.concat([df_decisions_1, df_decisions_2]).reset_index()

# * Drop participant who somehow did not have Quant Expectation or Perception in round 1
df_decisions_all = df_decisions_all[
    df_decisions_all["participant.label"] != PARTICIPANT_WITHOUT_BELIEF_DATA
]

# Add benchmark/naive strategies cumulative purchases
df_decisions_all["cum_decision_optimal"] = df_decisions_all.groupby(
    ["participant.code", "participant.inflation"]
)["qoptimal"].cumsum()
df_decisions_all["cum_decision_naive"] = df_decisions_all.groupby(
    ["participant.code", "participant.inflation"]
)["qnaive"].cumsum()

df_decisions_all.head()


# %% [markdown]
## Savings Game parameters
fig, axs = plt.subplots(1, 1, figsize=(10, 7))

# Plot savings and stock on first subplot
calc_opp_costs.plot_savings_and_stock(
    df_decisions_all[df_decisions_all["participant.inflation"] == 430],
    month_col="Month",
    strategy_stock_cols=["sgoptimal", "sgnaive"],
    strategy_savings_cols=["soptimal", "snaive"],
    strategy_names=["Benchmark", "Naïve"],
    palette="tab10",
    ax=axs,
    set_ylim=True,
    fontsize=16,
)

axs.set_xlabel("Period", fontsize=16)

axs.legend(fontsize=16)
axs.set_xticks(axs.get_xticks()[0:120:12])

# calc_opp_costs.plot_savings_and_stock(
#     df_decisions_all[df_decisions_all["participant.inflation"] == 1012],
#     month_col="Month",
#     strategy_stock_cols=["sgoptimal", "sgnaive"],
#     strategy_savings_cols=["soptimal", "snaive"],
#     strategy_names=["Benchmark", "Naïve"],
#     palette="tab10",
#     ax=axs[0][1],
#     set_ylim=True,
#     fontsize=20,
# )

# axs[0][1].set_xlabel("")

# axs[0][1].legend(loc="upper left", fontsize=20)
# axs[0][1].set_xticks(axs[0][1].get_xticks()[0:120:12])

# Plot inflation estimates on second subplot

# %%
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

estimates = ["4x30"]
df_inf_plot = df_survey.copy()
df_inf_plot = df_inf_plot.replace("Actual", "4x30")
df_inf_plot = df_inf_plot.rename(columns={"Measure": "Inflation sequence"})
sns.lineplot(
    data=df_inf_plot[df_inf_plot["Inflation sequence"].isin(estimates)],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Inflation sequence",
    style="Inflation sequence",
    ax=axs[0],
)

# Adjust titles and labels
axs[0].set_title("4x30 sequence", fontsize=20)
axs[0].set_xlabel("", labelpad=20, fontsize=20)
axs[0].set_ylabel("Inflation rate (%)", labelpad=20, fontsize=20)
axs[0].legend("", frameon=False)

estimates = ["10x12"]
df_inf_plot = df_survey_1[df_survey_1["participant.inflation"] == "10x12"].copy()
df_inf_plot = df_inf_plot.replace("Actual", "10x12")
df_inf_plot = df_inf_plot.rename(columns={"Measure": "Inflation sequence"})

sns.lineplot(
    data=df_inf_plot[df_inf_plot["Inflation sequence"].isin(estimates)],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Inflation sequence",
    style="Inflation sequence",
    ax=axs[1],
)

# Adjust titles and labels
axs[1].set_title("10x12 sequence", fontsize=20)
axs[1].set_xlabel("Month", labelpad=20, fontsize=20)
axs[1].set_ylabel("Inflation rate (%)", labelpad=20, fontsize=20)
axs[1].legend("", frameon=False)

# plt.tight_layout()
plt.show()


# %% [markdown]
## Overall performance
df_decisions_all[
    [
        "Mean Expectation Bias",
        "Mean Perception Bias",
    ]
] = df_decisions_all.groupby("participant.code")[
    [
        "Expectation_bias",
        "Perception_bias",
    ]
].transform(
    "mean"
)

# Store performance measures for appendix results below
df_performance_measures = df_decisions_all.copy()

summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120) & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["participant.inflation"])[PERFORMANCE_COLS]
    .describe()[[(c, "mean") for c in PERFORMANCE_COLS]]
    .reset_index()
)
summary_std = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120) & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["participant.inflation"])[PERFORMANCE_COLS]
    .describe()[[(c, "std") for c in PERFORMANCE_COLS]]
    .reset_index()
)

summary.columns = summary.columns.get_level_values(0)

summary = summary.rename(
    columns={
        **BEHAVIOR_COLS
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Mean Perception Bias Low": "Perception_bias_low",
            "Mean Perception Bias High": "Perception_bias_high",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Mean Expectation Bias Low": "Expectation_bias_low",
            "Mean Expectation Bias High": "Expectation_bias_high",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary[[c for c in BEHAVIOR_COLS.values()]] = (
    summary[[c for c in BEHAVIOR_COLS.values()]] * 100
)
summary["Inflation"] = np.where(summary["Inflation"] == 430, "4x30", "10x12")
# summary["Day"] = summary["Day"].astype(int)

summary_dict = (
    summary.groupby(["Inflation"])
    .describe()[[(c, "mean") for c in summary.columns[1:5]]]
    .to_dict()
)

summary_std.columns = summary_std.columns.get_level_values(0)

summary_std = summary_std.rename(
    columns={
        **BEHAVIOR_COLS
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Mean Perception Bias Low": "Perception_bias_low",
            "Mean Perception Bias High": "Perception_bias_high",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Mean Expectation Bias Low": "Expectation_bias_low",
            "Mean Expectation Bias High": "Expectation_bias_high",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary_std[[c for c in BEHAVIOR_COLS.values()]] = (
    summary_std[[c for c in BEHAVIOR_COLS.values()]] * 100
)
summary_std["Inflation"] = np.where(summary_std["Inflation"] == 430, "4x30", "10x12")
# summary_std["Day"] = summary_std["Day"].astype(int)

summary_std_dict = (
    summary_std.groupby(["Inflation"])
    .describe()[[(c, "mean") for c in summary_std.columns[1:5]]]
    .to_dict()
)
combined_dict = combine_mean_std_dicts(summary_dict, summary_std_dict)

pd.DataFrame(combined_dict).sort_index(ascending=False).style

# %% [markdown]
### Inflation beliefs
summary_dict = (
    summary.groupby(["Inflation"])
    .describe()[[(c, "mean") for c in summary.columns[5:]]]
    .to_dict()
)
summary_std_dict = (
    summary_std.groupby(["Inflation"])
    .describe()[[(c, "mean") for c in summary_std.columns[5:]]]
    .to_dict()
)
combined_dict = combine_mean_std_dicts(summary_dict, summary_std_dict)

pd.DataFrame(combined_dict).style


# %% [markdown]
### Performance plots
fig, axs = plt.subplots(1, 1, figsize=(12, 7))

# Plot savings and stock on first subplot
calc_opp_costs.plot_savings_and_stock(
    df_decisions_all[
        (df_decisions_all["participant.inflation"] == 430)
        & (df_decisions_all["phase"] == "pre")
    ],
    month_col="Month",
    strategy_stock_cols=["sgoptimal", "sgnaive", "finalStock"],
    strategy_savings_cols=["soptimal", "snaive", "sreal"],
    strategy_names=["Benchmark", "Naïve", "Average"],
    palette="tab10",
    ax=axs,
    set_ylim=True,
    fontsize=16,
)

axs.set_xlabel("Period", fontsize=16)

axs.legend(loc="upper center", fontsize=16)
axs.set_xticks(axs.get_xticks()[0:120:12])

# %% [markdown]
### Inflation beliefs plots
xlabel_fontsize = 16
ylabel_fontsize = 16
legend_fontsize = 16

fig, axs = plt.subplots(1, 1, figsize=(10, 7))
estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
hue_order = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]

# Manually define colors using the tab10 palette
color_palette = sns.color_palette("tab10")
palette_dict = {
    "Quant Expectation": color_palette[3],  # red
    "Quant Perception": color_palette[2],  # green
    "Actual": color_palette[0],  # blue
    "Upcoming": color_palette[1],  # orange
}
sns.lineplot(
    data=df_survey[df_survey["Measure"].isin(estimates)],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Measure",
    style="Measure",
    ax=axs,
    palette=palette_dict,
    hue_order=hue_order,
)

# Adjust titles and labels
axs.set_xlabel("Period", fontsize=xlabel_fontsize)
axs.set_ylabel("Inflation rate (%)", labelpad=20, fontsize=ylabel_fontsize)

# Manually define legend labels
handles, _ = axs.get_legend_handles_labels()
# New labels corresponding to the hue_order
new_labels = ["Perceived", "Expected", "Actual", "Upcoming"]
axs.legend(
    handles=handles,
    labels=new_labels,
    loc="upper left",
    fontsize=legend_fontsize,
)

plt.tight_layout()
plt.show()

# %% [markdown]
### Correlation: perceptions and expectations
data = df_decisions_all[
    (df_decisions_all["participant.inflation"] == 430)
    & (df_decisions_all["phase"] == "pre")
    & (df_decisions_all["Month"] >= 12)
    & (df_decisions_all["Quant Perception"].notna())
    & (df_decisions_all["Quant Expectation"].notna())
]
correlation, p_value = pearsonr(data["Quant Perception"], data["Quant Expectation"])

print(
    f"Correlation between Quant Perception and Quant Expectation: {correlation:.3f} (p-value: {p_value:.10f})"
)

# %% [markdown]
## Test difference between experiments
for measure in [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Perception_bias_high",
    "Perception_bias_low",
    "Expectation_sensitivity",
    "Expectation_bias_high",
    "Expectation_bias_low",
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

# %% [markdown]
## OLS regressions: Overall performance measures on inflation measures
df_regress = df_decisions_all[(df_decisions_all["participant.inflation"] == 430)]
df_regress = df_regress.rename(
    columns={
        "Mean Perception Bias": "avg_perception_bias",
        "Mean Expectation Bias": "avg_expectation_bias",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
)

# %%
regressions = {}

for m in ["sreal_percent", "early_percent", "excess_percent"]:
    model = smf.ols(
        formula=f"""{m} ~ Expectation_sensitivity + avg_expectation_bias\
            + Perception_sensitivity + avg_perception_bias""",
        data=df_regress[(df_regress["phase"] == "pre") & (df_regress["Month"] == 120)],
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results


# %% [markdown]
### Classify behavioral patterns
#### t = 12
df_decisions_all = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_all,
    estimate_measure="Quant Perception",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
    drop_na=True,
)

# %%
summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120)
        & (df_decisions_all["phase"] == "pre")
        & (df_decisions_all["participant.inflation"] == 430)
    ]
    .groupby(["Quant Perception_pattern_12"])[CLASSIFICATION_COLS]
    .describe()[
        [(c, "count") if c == "Month" else (c, "mean") for c in CLASSIFICATION_COLS]
    ]
    .reset_index()
)

# For overall performance, bottom row
summary_all = df_decisions_all[
    (df_decisions_all["Month"] == 120)
    & (df_decisions_all["phase"] == "pre")
    & (df_decisions_all["participant.inflation"] == 430)
].describe()
summary_all = summary_all[summary_all.index == "mean"]
summary_all = summary_all.rename(columns=NEW_CLASSIFICIATION_COLS)

summary.columns = summary.columns.get_level_values(0)
summary["Month"] = summary["Month"] / summary["Month"].sum()
summary = summary.rename(columns=NEW_CLASSIFICIATION_COLS)
summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] = (
    summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] * 100
)

summary.loc[len(summary)] = [
    "Overall",
    100,
    summary_all["Total performance (%)"].iat[0] * 100,
    summary_all["Over-stocking (%)"].iat[0] * 100,
    summary_all["Wasteful-stocking (%)"].iat[0] * 100,
]

summary


# %%
for measure in [
    "Perception_bias",
    "Perception_sensitivity",
    "Expectation_bias",
    "Expectation_sensitivity",
]:
    df_decisions_all[measure] = df_decisions_all.groupby("participant.code")[
        measure
    ].bfill()


df_decisions_all["perception_accuracy"] = decision_patterns.classify_perception(
    df_decisions_all["Perception_sensitivity"],
    ACCURATE_PERCEPTIONS_THRESHOLD,
    accurate_label="S",
    inaccurate_label="I",
)

# * Include qualitative perceptions for Experiment 2
df_decisions_all["qualitative_perception_36"] = (
    df_decisions_all[df_decisions_all["Month"] == 36]
    .groupby("participant.code")["Qual Perception"]
    .transform("mean")
)
df_decisions_all["qualitative_perception_36"] = (
    df_decisions_all.groupby("participant.code")["qualitative_perception_36"]
    .bfill()
    .ffill()
)
df_decisions_all["qualitative_perception_accuracy"] = (
    decision_patterns.classify_perception(
        df_decisions_all["qualitative_perception_36"], ACCURATE_QUALITATIVE_THRESHOLD
    )
)

for month in [12, 30]:
    df_decisions_all[f"purchase_adaptation_{month}"] = (
        decision_patterns.classify_purchase_adaptation(
            df_decisions_all,
            comparison_start_month=month,
            positive_adaptation_label="A",
            negative_adaptation_label="N",
        )
    )
    for inflation_measure in ["perception_accuracy", "qualitative_perception_accuracy"]:
        df_decisions_all[f"decision_pattern_{month}_{inflation_measure}"] = (
            decision_patterns.classify_new_decision_patterns(
                df_decisions_all, f"purchase_adaptation_{month}", inflation_measure
            )
        )

# %%[markdown]
#### Quantitative pattern, t=30
summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120)
        & (df_decisions_all["phase"] == "pre")
        & (df_decisions_all["participant.inflation"] == 430)
    ]
    .groupby(["decision_pattern_30_perception_accuracy"])[CLASSIFICATION_COLS]
    .describe()[
        [(c, "count") if c == "Month" else (c, "mean") for c in CLASSIFICATION_COLS]
    ]
    .reset_index()
)

# For overall performance, bottom row
summary_all = df_decisions_all[
    (df_decisions_all["Month"] == 120)
    & (df_decisions_all["phase"] == "pre")
    & (df_decisions_all["participant.inflation"] == 430)
].describe()
summary_all = summary_all[summary_all.index == "mean"]
summary_all = summary_all.rename(columns=NEW_CLASSIFICIATION_COLS)

summary.columns = summary.columns.get_level_values(0)
summary["Month"] = summary["Month"] / summary["Month"].sum()
summary = summary.rename(columns=NEW_CLASSIFICIATION_COLS)
summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] = (
    summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] * 100
)

summary.loc[len(summary)] = [
    "Overall",
    100,
    summary_all["Total performance (%)"].iat[0] * 100,
    summary_all["Over-stocking (%)"].iat[0] * 100,
    summary_all["Wasteful-stocking (%)"].iat[0] * 100,
]

summary

# %%
# * Remove perception accuracy to only compare coherent decisions
df_decisions_all["Quant Perception_consistent_12"] = df_decisions_all[
    "Quant Perception_pattern_12"
].str[1]

# * Set both purchase adapations as binary variables
df_decisions_all["Quant Perception_consistent_12"] = np.where(
    df_decisions_all["Quant Perception_consistent_12"] == "C", 1, 0
)
df_decisions_all["purchase_adaptation_30"] = np.where(
    df_decisions_all["purchase_adaptation_30"] == "A", 1, 0
)

# %% [markdown]
## Behavioral measures
df_behavioral = df_decisions_all[
    (df_decisions_all["phase"] >= "pre")
    & (df_decisions_all["participant.inflation"] == 430)
]

df_knowledge_1 = knowledge.create_knowledge_dataframe(con_exp_1)
df_econ_preferences_1 = econ_preferences.create_econ_preferences_dataframe(con_exp_1)

# * Only keep day 1 knowledge measures from Experiment 1 (since they were repeated post-intervention)
df_knowledge_1 = df_knowledge_1[df_knowledge_1["participant.day"] < 3]
df_knowledge_1[["participant.round", "exp"]] = 1
df_knowledge_1 = df_knowledge_1.drop(columns="participant.day")

df_econ_preferences_1[["participant.round", "exp"]] = 1
df_econ_preferences_1 = df_econ_preferences_1.drop(columns="participant.day")

df_knowledge_2 = knowledge.create_knowledge_dataframe(con_exp_2)
df_econ_preferences_2 = econ_preferences.create_econ_preferences_dataframe(con_exp_2)
df_knowledge_2[["exp"]] = 2
df_econ_preferences_2[["exp"]] = 2

# %%
df_knowledge_all = pd.concat([df_knowledge_1, df_knowledge_2]).reset_index()
df_knowledge_all = df_knowledge_all.drop(columns="index")
df_econ_preferences_all = pd.concat(
    [df_econ_preferences_1, df_econ_preferences_2]
).reset_index()
df_econ_preferences_all = df_econ_preferences_all.drop(columns="index")

# %%
df_behavioral = combine_series(
    [df_behavioral, df_knowledge_all, df_econ_preferences_all],
    how="left",
    on=["participant.label", "participant.round", "exp"],
)

y_cols = [c for c in df_behavioral.columns if c.endswith("_y")]
df_behavioral = df_behavioral.drop(columns=y_cols)

df_behavioral.columns = df_behavioral.columns.str.removesuffix("_x")

# %%
df_behavioral["n_switches"] = df_behavioral[
    ["lossAversion_switches", "riskPreferences_switches", "timePreferences_switches"]
].sum(axis=1)

# %%
df_corr = create_dynamic_correlation_matrix(
    df_behavioral[df_behavioral["Month"] == 120][CORRELATION_COLS],
    p_values=[0.1, 0.05, 0.01],
    include_stars=True,
    display=False,
    decimal_places=2,
)
df_corr[df_corr.index.isin(CORRELATION_COLS[7:13])][CORRELATION_COLS[:7]]

# %% [markdown]
## OLS/Logistic regression of performance and decision patterns on behavioral variables (Condensed)
df_regress = pd.get_dummies(
    df_behavioral,
    columns=["decision_pattern_30_perception_accuracy"],
    drop_first=False,
    dtype=int,
)

df_regress = pd.get_dummies(
    df_regress,
    columns=["Quant Perception_pattern_12"],
    drop_first=False,
    dtype=int,
)


# %%
df_regress = df_regress.rename(
    columns={
        "sreal_%": "sreal_percent",
        "Quant Perception_consistent_12": "perception_consistent_12",
        "Quant Perception_pattern_12_AC": "perception_pattern_12_AC",
        "Quant Perception_pattern_12_AI": "perception_pattern_12_AI",
        "Quant Perception_pattern_12_IC": "perception_pattern_12_IC",
    },
)

df_regress_individual_chars = df_regress.copy()

# %%
regressions = {}

for m in ["sreal_percent"] + LOGIT_COLS[:2] + LOGIT_COLS[3:5]:
    formula = f"""{m} ~ C(financial_literacy) + C(numeracy) + C(compound) + n_switches\
                + wisconsin_choice_count + lossAversion_choice_count + riskPreferences_choice_count\
                    + timePreferences_choice_count"""
    if m in LOGIT_COLS:
        model = smf.logit(
            formula=formula,
            data=df_regress[
                (df_regress["phase"] == "pre") & (df_regress["Month"] == 120)
            ],
        )
    else:
        model = smf.ols(
            formula=formula,
            data=df_regress[
                (df_regress["phase"] == "pre") & (df_regress["Month"] == 120)
            ],
        )
    regressions[m] = model.fit()

# Define custom info_dict to show Pseudo R-squared for logistic models
info_dict = {
    "Pseudo R-squared": lambda x: (
        "%#8.3f" % x.prsquared if hasattr(x, "prsquared") else ""
    ),
}

# Create the summary table with the info_dict parameter
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
    info_dict=info_dict,
)

results

# %% [markdown]
## Learning effect
df_learn = df_decisions_all[
    (df_decisions_all["treatment"].isin(["control", "Control"]))
    & (df_decisions_all["participant.inflation"] == 430)
    & (df_decisions_all["Month"] == 120)
]

df_learn = pd.get_dummies(
    df_learn,
    columns=["decision_pattern_30_perception_accuracy"],
    drop_first=False,
    dtype=int,
)

df_learn = pd.get_dummies(
    df_learn,
    columns=["Quant Perception_pattern_12"],
    drop_first=False,
    dtype=int,
)
# %%

learning_effect, _ = intervention.create_learning_effect_table(
    df_learn,
    [
        "sreal_%",
        "early_%",
        "excess_%",
        "Perception_sensitivity",
        "Expectation_sensitivity",
        "purchase_adaptation_30",
        "Quant Perception_consistent_12",
        "decision_pattern_30_perception_accuracy_IA",
    ],
    p_value_threshold=[0.1, 0.05, 0.01],
    decimal_places=4,
)
learning_effect = learning_effect.set_index("")
learning_effect

# %% [markdown]
## Treatment effect
df_treat = df_decisions_all[
    (df_decisions_all["participant.inflation"] == 430)
    & (df_decisions_all["Month"] == 120)
]

df_treat = df_treat[df_treat["participant.label"] != "JKmBvh7"]

treatments_rename = {
    "control": "Control",
    "Control": "Control",
    "intervention": "Intervention 1",
    "Intervention 1": "Intervention 2",
    "Intervention 2": "Intervention 3",
}

df_treat["treatment"] = np.select(
    condlist=[df_treat["treatment"] == k for k in treatments_rename.keys()],
    choicelist=[v for v in treatments_rename.values()],
    default="",
)

treatment_effect = intervention.create_diff_in_diff_table(
    df_treat,
    [
        "sreal_%",
        "early_%",
        "excess_%",
        "Perception_sensitivity",
        "Expectation_sensitivity",
        "purchase_adaptation_30",
        "Quant Perception_consistent_12",
    ],
    treatments=TREATMENT_GROUPS,
    control="Control",
    p_value_threshold=[0.1, 0.05, 0.01],
    decimal_places=4,
)
treatment_effect = treatment_effect.set_index("")

treatment_effect

# %% [markdown]
## Appendix E
# %% [markdown]
### Overall performance
df_performance_measures[["Mean Perception Bias", "Mean Expectation Bias"]] = (
    df_performance_measures.groupby("participant.code")[
        ["Perception_bias", "Expectation_bias"]
    ].transform("mean")
)

summary = (
    df_performance_measures[
        (df_performance_measures["Month"] == 120)
        & (df_performance_measures["phase"] == "pre")
    ]
    .groupby(["exp", "participant.inflation"])[PERFORMANCE_COLS]
    .describe()[[(c, "mean") for c in PERFORMANCE_COLS]]
    .reset_index()
)
summary_std = (
    df_performance_measures[
        (df_performance_measures["Month"] == 120)
        & (df_performance_measures["phase"] == "pre")
    ]
    .groupby(["exp", "participant.inflation"])[PERFORMANCE_COLS]
    .describe()[[(c, "std") for c in PERFORMANCE_COLS]]
    .reset_index()
)

summary.columns = summary.columns.get_level_values(0)

summary = summary.rename(
    columns={
        **BEHAVIOR_COLS
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Perception_bias_low": "Mean Perception Bias Low",
            "Perception_bias_high": "Mean Perception Bias High",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Expectation_bias_low": "Mean Expectation Bias Low",
            "Expectation_bias_high": "Mean Expectation Bias High",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary[[c for c in BEHAVIOR_COLS.values()]] = (
    summary[[c for c in BEHAVIOR_COLS.values()]] * 100
)
summary["Inflation"] = np.where(summary["Inflation"] == 430, "4x30", "10x12")
# summary["Day"] = summary["Day"].astype(int)

summary_dict = (
    summary.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary.columns[2:6]]]
    .to_dict()
)

summary_std.columns = summary_std.columns.get_level_values(0)

summary_std = summary_std.rename(
    columns={
        **BEHAVIOR_COLS
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Perception_bias_low": "Mean Perception Bias Low",
            "Perception_bias_high": "Mean Perception Bias High",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Expectation_bias_low": "Mean Expectation Bias Low",
            "Expectation_bias_high": "Mean Expectation Bias High",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary_std[[c for c in BEHAVIOR_COLS.values()]] = (
    summary_std[[c for c in BEHAVIOR_COLS.values()]] * 100
)
summary_std["Inflation"] = np.where(summary_std["Inflation"] == 430, "4x30", "10x12")
# summary_std["Day"] = summary_std["Day"].astype(int)

summary_std_dict = (
    summary_std.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary_std.columns[2:6]]]
    .to_dict()
)
combined_dict = combine_mean_std_dicts(summary_dict, summary_std_dict)

pd.DataFrame(combined_dict).style

# %% [markdown]
### Inflation beliefs
summary_dict = (
    summary.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary.columns[6:]]]
    .to_dict()
)
summary_std_dict = (
    summary_std.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary_std.columns[6:]]]
    .to_dict()
)
combined_dict = combine_mean_std_dicts(summary_dict, summary_std_dict)

pd.DataFrame(combined_dict).style

# %%
data = df_performance_measures.copy()
data = data.rename(columns={"participant.inflation": "Inflation"})
data["Inflation"] = np.where(data["Inflation"] == 430, "4x30", "10x12")
sns.lmplot(
    data[data["participant.round"] == 1],
    x="Quant Perception",
    y="Quant Expectation",
    hue="participant.round",
    col="Inflation",
    legend=None,
)

# %%
### OLS Regression: Effects of belief accuracy on performance with inflation-phase biases, 4×30 sequence
df_regress = df_performance_measures[
    (df_performance_measures["participant.inflation"] == 430)
]
df_regress = df_regress.rename(
    columns={
        "Mean Perception Bias": "avg_perception_bias",
        "Mean Expectation Bias": "avg_expectation_bias",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
)

regressions = {}

for m in ["sreal_percent", "early_percent", "excess_percent"]:
    model = smf.ols(
        formula=f"""{m} ~ Expectation_sensitivity + Expectation_bias_low + Expectation_bias_high\
            + Perception_sensitivity + Perception_bias_low + Perception_bias_high""",
        data=df_regress[(df_regress["phase"] == "pre") & (df_regress["Month"] == 120)],
    )
    regressions[m] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results

# %% [markdown]
### Belief-behavior classification (qualitative)
summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120)
        & (df_decisions_all["exp"] == 2)
        & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["decision_pattern_30_qualitative_perception_accuracy"])[
        CLASSIFICATION_COLS
    ]
    .describe()[
        [(c, "count") if c == "Month" else (c, "mean") for c in CLASSIFICATION_COLS]
    ]
    .reset_index()
)

# For overall performance, bottom row
summary_all = df_decisions_all[
    (df_decisions_all["Month"] == 120)
    & (df_decisions_all["exp"] == 2)
    & (df_decisions_all["phase"] == "pre")
].describe()
summary_all = summary_all[summary_all.index == "mean"]
summary_all = summary_all.rename(columns=NEW_CLASSIFICIATION_COLS)

summary.columns = summary.columns.get_level_values(0)
summary["Month"] = summary["Month"] / summary["Month"].sum()
summary = summary.rename(columns=NEW_CLASSIFICIATION_COLS)
summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] = (
    summary[[c for c in NEW_CLASSIFICIATION_COLS.values()]] * 100
)

summary.loc[len(summary)] = [
    "Overall",
    100,
    summary_all["Total performance (%)"].iat[0] * 100,
    summary_all["Over-stocking (%)"].iat[0] * 100,
    summary_all["Wasteful-stocking (%)"].iat[0] * 100,
]

summary

# %% [markdown]
### complete behavioral correlation matrix
df_corr = create_dynamic_correlation_matrix(
    df_behavioral[df_behavioral["Month"] == 120][CORRELATION_COLS],
    p_values=[0.1, 0.05, 0.01],
    include_stars=True,
    display=False,
    decimal_places=2,
)
df_corr[df_corr.index.isin(CORRELATION_COLS[7:])][CORRELATION_COLS[:7]]

# %% [markdown]
### Results of Bonferroni correction
create_bonferroni_correlation_table(
    df_behavioral[df_behavioral["Month"] == 120][CORRELATION_COLS],
    CORRELATION_COLS[7:],
    CORRELATION_COLS[:7],
    filtered_results=False,
)

# %% [markdown]
### OLS/Logistic regression of performance and decision patterns on behavioral variables (complete)

df_regress_individual_chars = df_regress_individual_chars.rename(
    columns={"Quant Perception_consistent_12": "perception_consistent_12"}
)

regressions = {}

for m in ["sreal_percent"] + LOGIT_COLS:
    formula = f"""{m} ~ C(financial_literacy) + C(numeracy) + C(compound) + n_switches\
                + wisconsin_choice_count + lossAversion_choice_count + riskPreferences_choice_count\
                    + timePreferences_choice_count"""
    if m in LOGIT_COLS:
        model = smf.logit(
            formula=formula,
            data=df_regress_individual_chars[
                (df_regress_individual_chars["phase"] == "pre")
                & (df_regress_individual_chars["Month"] == 120)
            ],
        )
    else:
        model = smf.ols(
            formula=formula,
            data=df_regress_individual_chars[
                (df_regress_individual_chars["phase"] == "pre")
                & (df_regress_individual_chars["Month"] == 120)
            ],
        )
    regressions[m] = model.fit()

# Define custom info_dict to show Pseudo R-squared for logistic models
info_dict = {
    "Pseudo R-squared": lambda x: (
        "%#8.3f" % x.prsquared if hasattr(x, "prsquared") else ""
    ),
}

# Create the summary table with the info_dict parameter
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
    info_dict=info_dict,
)

results

# %% [markdown]
### Learning effect (complete)
learning_effect, _ = intervention.create_learning_effect_table(
    df_learn,
    [
        "sreal_%",
        "early_%",
        "excess_%",
        "Perception_sensitivity",
        "Expectation_sensitivity",
        "purchase_adaptation_30",
        "Quant Perception_consistent_12",
        "decision_pattern_30_perception_accuracy_SA",
        "decision_pattern_30_perception_accuracy_IN",
        "decision_pattern_30_perception_accuracy_IA",
        "decision_pattern_30_perception_accuracy_SN",
        "Quant Perception_pattern_12_AC",
        "Quant Perception_pattern_12_AI",
        "Quant Perception_pattern_12_IC",
    ],
    p_value_threshold=[0.1, 0.05, 0.01],
)
learning_effect = learning_effect.set_index("")
learning_effect


# %%
### ANCOVA: Intervention effects
df_diff = pd.pivot_table(
    df_treat[["participant.label", "phase", "treatment"] + CORRELATION_COLS[:7]],
    index=["participant.label", "treatment"],
    columns=["phase"],
)
df_diff.reset_index(inplace=True)
for m in CORRELATION_COLS[:7]:
    df_diff[f"change_{m}"] = df_diff[(m, "post")] - df_diff[(m, "pre")]

# Combine column names if the second level is not blank
df_diff.columns = df_diff.columns.map(
    lambda col: (
        col[0]
        if isinstance(col, tuple)
        and (len(col) < 2 or col[1] is None or str(col[1]).strip() == "")
        else "_".join(col) if isinstance(col, tuple) else col
    )
)

df_diff = df_diff.merge(
    df_behavioral[df_behavioral["Month"] == 120][
        ["participant.label"] + CORRELATION_COLS[7:]
    ],
    how="left",
)

df_diff.columns = [
    col.replace("%", "percent") if isinstance(col, str) else col
    for col in df_diff.columns
]
df_diff.columns = [
    col.replace(" ", "_") if isinstance(col, str) else col for col in df_diff.columns
]

regressions = {}
for measure in CORRELATION_COLS[:7]:
    measure_sanitized = (
        measure.replace("%", "percent") if "%" in measure else measure.replace(" ", "_")
    )
    regressions[measure] = {}
    formula = f"""{measure_sanitized}_post ~ C(treatment) + {measure_sanitized}_pre"""
    model = smf.ols(
        formula=formula,
        data=df_diff,
    )
    regressions[measure] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)

results

# %% [markdown]
### ANCOVA of heterogeneous treatment effects: Individual characteristics

regressions = {}
for measure in CORRELATION_COLS[:7]:
    measure_sanitized = (
        measure.replace("%", "percent") if "%" in measure else measure.replace(" ", "_")
    )
    pre, post = f"{measure_sanitized}_pre", f"{measure_sanitized}_post"
    regressions[measure] = {}
    for treatment in TREATMENT_GROUPS:
        for characteristic in CORRELATION_COLS[7:]:
            if characteristic in ["numeracy", "financial_literacy", "compound"]:
                formula = f"""{post} ~ C(treatment)*C({characteristic}) + {pre}"""
            else:
                formula = f"""{post} ~ C(treatment)*{characteristic} + {pre}"""

            model = smf.ols(
                formula=formula,
                data=df_diff,
            )
            regressions[measure][characteristic] = model.fit()

results_tables = {}
for measure in CORRELATION_COLS[:7]:
    results = summary_col(
        results=list(regressions[measure].values()),
        stars=True,
        model_names=list(regressions[measure].keys()),
    )
    results_tables[measure] = results

timestr = time.strftime("%Y%m%d-%H%M%S")
with pd.ExcelWriter(
    EXPORT_FILE_PATH / f"ancova_heterogeneous_effects_{timestr}.xlsx"
) as writer:
    for measure, results_table in results_tables.items():
        df_to_write = results_table.tables[0]
        sanitized_measure = measure.replace("%", "pct").replace(" ", "_")[:31]
        df_to_write.to_excel(writer, sheet_name=sanitized_measure)

logger.info("Exported ANCOVA results to Excel. Now, condensing results for display.")

input_file = EXPORT_FILE_PATH / f"ancova_heterogeneous_effects_{timestr}.xlsx"
output_file = (
    EXPORT_FILE_PATH / f"ancova_heterogeneous_effects_condensed_{timestr}.xlsx"
)

ancova_restructurer.restructure_excel_file(input_file, output_file)
