"""Present results from both experiments"""

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns

# from sklearn.preprocessing import OneHotEncoder
import statsmodels.formula.api as smf
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
from src.utils import exp_1_patches

from src.stats_analysis import (
    apply_statistical_test,
    create_bonferroni_correlation_table,
    create_dynamic_correlation_matrix,
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
from src.utils.helpers import combine_mean_std_dicts, combine_series, export_plot
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
# export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
FILE_PATH = Path(__file__).parents[1] / "results"

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
CORRELATION_COLS = [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Expectation_sensitivity",
    "purchase_adaptation_30",
    "Quant Perception_pattern_12",
    "financial_literacy",
    "numeracy",
    "compound",
    "n_switches",
    "wisconsin_choice_count",
    "lossAversion_choice_count",
    "riskPreferences_choice_count",
    "timePreferences_choice_count",
]

TREATMENT_GROUPS = [
    "Intervention (Exp 1)",
    "Intervention 1 (Exp 2)",
    "Intervention 2 (Exp 2)",
]

LOGIT_COLS = [
    "decision_pattern_30_perception_accuracy_AN",
    "decision_pattern_30_perception_accuracy_AP",
    "decision_pattern_30_perception_accuracy_IN",
    "decision_pattern_30_perception_accuracy_IP",
]

# %%
if not table_exists(con_exp_1, "Questionnaire"):
    create_duckdb_database(con_exp_1, experiment=1, initial_creation=True)
if not table_exists(con_exp_2, "Questionnaire"):
    create_duckdb_database(con_exp_2, experiment=2, initial_creation=True)

# %% [markdown]
## Experiment 1
df_questionnaire = con_exp_1.sql("SELECT * FROM Questionnaire").df()

df_questionnaire[QUESTIONNAIRE_COLS].describe()

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
df_decisions_1 = df_decisions_1.merge(
    df_expectations[["participant.code", "participant.day"]], how="left"
)

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

df_questionnaire[QUESTIONNAIRE_COLS].describe()

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
cols = [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Mean Perception Bias",
    "Expectation_sensitivity",
    "Mean Expectation Bias",
]
new_cols = {
    "sreal_%": "Total performance (%)",
    "early_%": "Over-stocking (%)",
    "excess_%": "Wasteful-stocking (%)",
}
df_decisions_all = pd.concat([df_decisions_1, df_decisions_2]).reset_index()

logger.debug("df_decisions_all.shape: %s", df_decisions_all.shape)

# %% [markdown]
## Savings Game parameters
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot savings and stock on first subplot
calc_opp_costs.plot_savings_and_stock(
    df_decisions_all[df_decisions_all["participant.inflation"] == 430],
    month_col="Month",
    strategy_stock_cols=["sgoptimal", "sgnaive"],
    strategy_savings_cols=["soptimal", "snaive"],
    strategy_names=["Best", "Naïve"],
    palette="tab10",
    ax=axs[0],
    set_ylim=True,
    fontsize=20,
)

axs[0].set_xlabel("")

axs[0].legend(loc="upper left", fontsize=20)
axs[0].set_xticks(axs[0].get_xticks()[0:120:12])

# Plot inflation estimates on second subplot
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
    ax=axs[1],
)

# Adjust titles and labels
axs[1].set_xlabel("Month", labelpad=20, fontsize=20)
axs[1].set_ylabel("Inflation rate (%)", labelpad=20, fontsize=20)
axs[1].legend(loc="upper left", fontsize=20)

plt.tight_layout()
plt.show()


# %% [markdown]
## Overall performance


df_decisions_all[["Mean Perception Bias", "Mean Expectation Bias"]] = (
    df_decisions_all.groupby("participant.code")[
        ["Perception_bias", "Expectation_bias"]
    ].transform("mean")
)

summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120) & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["exp", "participant.inflation"])[cols]
    .describe()[[(c, "mean") for c in cols]]
    .reset_index()
)
summary_std = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120) & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["exp", "participant.inflation"])[cols]
    .describe()[[(c, "std") for c in cols]]
    .reset_index()
)

summary.columns = summary.columns.get_level_values(0)

summary = summary.rename(
    columns={
        **new_cols
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Mean Perception Bias": "Perception Bias",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Mean Expectation Bias": "Expectation Bias",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary[[c for c in new_cols.values()]] = summary[[c for c in new_cols.values()]] * 100
summary["Inflation"] = np.where(summary["Inflation"] == 430, "4x30", "10x12")
# summary["Day"] = summary["Day"].astype(int)

summary_dict = (
    summary.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary.columns[2:]]]
    .to_dict()
)

summary_std.columns = summary_std.columns.get_level_values(0)

summary_std = summary_std.rename(
    columns={
        **new_cols
        | {
            "Perception_sensitivity": "Perception Sensitivity",
            "Mean Perception Bias": "Perception Bias",
            "Expectation_sensitivity": "Expectation Sensitivity",
            "Mean Expectation Bias": "Expectation Bias",
            "participant.inflation": "Inflation",
            "exp": "Experiment",
            # "participant.day": "Day",
        }
    }
)
summary_std[[c for c in new_cols.values()]] = (
    summary_std[[c for c in new_cols.values()]] * 100
)
summary_std["Inflation"] = np.where(summary_std["Inflation"] == 430, "4x30", "10x12")
# summary_std["Day"] = summary_std["Day"].astype(int)

summary_std_dict = (
    summary_std.groupby(["Experiment", "Inflation"])
    .describe()[[(c, "mean") for c in summary_std.columns[2:]]]
    .to_dict()
)
combined_dict = combine_mean_std_dicts(summary_dict, summary_std_dict)

pd.DataFrame(combined_dict).style

# %% [markdown]
### Plots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot savings and stock on first subplot
calc_opp_costs.plot_savings_and_stock(
    df_decisions_all[df_decisions_all["participant.inflation"] == 430],
    month_col="Month",
    strategy_stock_cols=["sgoptimal", "sgnaive", "finalStock"],
    strategy_savings_cols=["soptimal", "snaive", "sreal"],
    strategy_names=["Best", "Naïve", "Average"],
    palette="tab10",
    ax=axs[0],
    set_ylim=True,
    fontsize=20,
)

axs[0].set_xlabel("")

axs[0].legend(loc="upper left", fontsize=16)
axs[0].set_xticks(axs[0].get_xticks()[0:120:12])

# Plot inflation estimates on second subplot
estimates = ["Quant Perception", "Quant Expectation", "Actual", "Upcoming"]
sns.lineplot(
    data=df_survey[df_survey["Measure"].isin(estimates)],
    x="Month",
    y="Estimate",
    errorbar=None,
    hue="Measure",
    style="Measure",
    ax=axs[1],
)

# Adjust titles and labels
axs[1].set_xlabel("Month", labelpad=20, fontsize=20)
axs[1].set_ylabel("Inflation rate (%)", labelpad=20, fontsize=20)
axs[1].legend(loc="upper left", fontsize=16)

plt.tight_layout()
plt.show()

# %% [markdown]
## Test difference between experiments
for measure in [
    "sreal_%",
    "early_%",
    "excess_%",
    "Perception_sensitivity",
    "Mean Perception Bias",
    "Expectation_sensitivity",
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


# %% [markdown]
### Classify behavioral patterns
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
    df_decisions_all["Perception_sensitivity"], ACCURATE_PERCEPTIONS_THRESHOLD
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
            df_decisions_all, comparison_start_month=month
        )
    )
    for inflation_measure in ["perception_accuracy", "qualitative_perception_accuracy"]:
        df_decisions_all[f"decision_pattern_{month}_{inflation_measure}"] = (
            decision_patterns.classify_new_decision_patterns(
                df_decisions_all, f"purchase_adaptation_{month}", inflation_measure
            )
        )

# %%
cols = [
    "Month",
    "sreal_%",
    "early_%",
    "excess_%",
]
new_cols = {
    "Month": "Proportion (%)",
    "sreal_%": "Total performance (%)",
    "early_%": "Over-stocking (%)",
    "excess_%": "Wasteful-stocking (%)",
}

# %%[markdown]
#### Quantitative pattern
summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120)
        & (df_decisions_all["phase"] == "pre")
        & (df_decisions_all["participant.inflation"] == 430)
    ]
    .groupby(["decision_pattern_30_perception_accuracy"])[cols]
    .describe()[[(c, "count") if c == "Month" else (c, "mean") for c in cols]]
    .reset_index()
)

# For overall performance, bottom row
summary_all = df_decisions_all[
    (df_decisions_all["Month"] == 120)
    & (df_decisions_all["phase"] == "pre")
    & (df_decisions_all["participant.inflation"] == 430)
].describe()
summary_all = summary_all[summary_all.index == "mean"]
summary_all = summary_all.rename(columns=new_cols)

summary.columns = summary.columns.get_level_values(0)
summary["Month"] = summary["Month"] / summary["Month"].sum()
summary = summary.rename(columns=new_cols)
summary[[c for c in new_cols.values()]] = summary[[c for c in new_cols.values()]] * 100

summary.loc[len(summary)] = [
    "Overall",
    100,
    summary_all["Total performance (%)"].iat[0] * 100,
    summary_all["Over-stocking (%)"].iat[0] * 100,
    summary_all["Wasteful-stocking (%)"].iat[0] * 100,
]

summary

# %%[markdown]
#### Qualitative pattern
summary = (
    df_decisions_all[
        (df_decisions_all["Month"] == 120)
        & (df_decisions_all["exp"] == 2)
        & (df_decisions_all["phase"] == "pre")
    ]
    .groupby(["decision_pattern_30_qualitative_perception_accuracy"])[cols]
    .describe()[[(c, "count") if c == "Month" else (c, "mean") for c in cols]]
    .reset_index()
)

# For overall performance, bottom row
summary_all = df_decisions_all[
    (df_decisions_all["Month"] == 120)
    & (df_decisions_all["exp"] == 2)
    & (df_decisions_all["phase"] == "pre")
].describe()
summary_all = summary_all[summary_all.index == "mean"]
summary_all = summary_all.rename(columns=new_cols)

summary.columns = summary.columns.get_level_values(0)
summary["Month"] = summary["Month"] / summary["Month"].sum()
summary = summary.rename(columns=new_cols)
summary[[c for c in new_cols.values()]] = summary[[c for c in new_cols.values()]] * 100

summary.loc[len(summary)] = [
    "Overall",
    100,
    summary_all["Total performance (%)"].iat[0] * 100,
    summary_all["Over-stocking (%)"].iat[0] * 100,
    summary_all["Wasteful-stocking (%)"].iat[0] * 100,
]

summary

# %%
df_decisions_all = decision_patterns.classify_subject_decision_patterns(
    data=df_decisions_all,
    estimate_measure="Quant Perception",
    decision_measure="finalStock",
    month=12,
    coherent_decision=0,
    threshold_estimate=ANNUAL_INTEREST_RATE,
)

# * Remove perception accuracy to only compare coherent decisions
df_decisions_all["Quant Perception_pattern_12"] = df_decisions_all[
    "Quant Perception_pattern_12"
].str[1]

# * Set both purchase adapations as binary variables
df_decisions_all["Quant Perception_pattern_12"] = np.where(
    df_decisions_all["Quant Perception_pattern_12"] == "C", 1, 0
)
df_decisions_all["purchase_adaptation_30"] = np.where(
    df_decisions_all["purchase_adaptation_30"] == "P", 1, 0
)

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
assert df_behavioral.shape[1] == 76

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
    # mask_upper_triangle=True,
)
df_corr[df_corr.index.isin(CORRELATION_COLS[7:])][CORRELATION_COLS[:7]]

# %% [markdown]
## OLS/Logistic regression of performance and decision patterns on behavioral variables
df_regress = pd.get_dummies(
    df_behavioral,
    columns=["decision_pattern_30_perception_accuracy"],
    drop_first=False,
    dtype=int,
)


# %%
df_regress = df_regress.rename(
    columns={"sreal_%": "sreal_percent"},
)


# %%
regressions = {}

for m in ["sreal_percent"] + LOGIT_COLS:
    formula = f"""{m} ~ financial_literacy + numeracy + compound + n_switches\
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

learning_effect, _ = intervention.create_learning_effect_table(
    df_learn,
    [
        "sreal_%",
        "early_%",
        "excess_%",
        "Perception_sensitivity",
        "Expectation_sensitivity",
        "purchase_adaptation_30",
        "Quant Perception_pattern_12",
    ],
    p_value_threshold=[0.1, 0.05, 0.01],
)
learning_effect = learning_effect.set_index("")
learning_effect

# %% [markdown]
## Treatment effect
df_treat = df_decisions_all[
    (df_decisions_all["participant.inflation"] == 430)
    & (df_decisions_all["Month"] == 120)
]

# * Drop participant who somehow did not have Quant Expectation in round 1
df_treat = df_treat[df_treat["participant.label"] != "JKmBvh7"]

treatments_rename = {
    "control": "Control",
    "Control": "Control",
    "intervention": "Intervention (Exp 1)",
    "Intervention 1": "Intervention 1 (Exp 2)",
    "Intervention 2": "Intervention 2 (Exp 2)",
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
        "Quant Perception_pattern_12",
    ],
    treatments=TREATMENT_GROUPS,
    control="Control",
    p_value_threshold=[0.1, 0.05, 0.01],
    decimal_places=4,
)
treatment_effect = treatment_effect.set_index("")

treatment_effect
