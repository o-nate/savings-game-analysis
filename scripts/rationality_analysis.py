# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from src import calc_opp_costs, process_survey

from src.utils.constants import INTEREST_RATE
from utils.logging_config import get_logger
from src.utils.plotting import annotate_2d_histogram, visualize_persona_results

# * Set logger
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

PERSONAS = ["AC", "AI", "IC", "II"]
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100
MEASURE = "finalStock"


def create_performance_measures_table(
    data: pd.DataFrame, inflation_measure: str, performance_measures: list[str]
) -> pd.DataFrame:

    df_final_stats = data.groupby(inflation_measure)[performance_measures].describe()

    final_stats_count = df_final_stats[(performance_measures[0], "count")].rename(
        "percentage_participants"
    )
    final_stats_percent = final_stats_count / final_stats_count.sum()

    stats_cols = [c for c in df_final_stats.columns if "mean" in c[1]]
    df_final_stats = df_final_stats[stats_cols].reset_index()
    df_final_stats.columns = df_final_stats.columns.droplevel(level=1)
    df_final_stats = df_final_stats.assign(
        percentage_participants=final_stats_percent.values
    )
    stats_cols = df_final_stats.columns.to_list()
    stats_cols.insert(1, stats_cols.pop())
    df_final_stats = df_final_stats[stats_cols]
    df_final_stats[stats_cols[1:]] = df_final_stats[stats_cols[1:]] * 100

    return df_final_stats


# %%
df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

# %%
df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

# %%
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

print(df_decisions.shape)
df_decisions.head()

# %% [markdown]
## Classify subjects per accurate vs. inaccurate estimate and coherent vs. incoherent decision

# %% [markdown]
### Perceptions

MAX_RATIONAL_STOCK = 0
MONTH = 12
ESTIMATE = "Quant Perception"

df_personas = df_decisions[df_decisions["Month"] == MONTH]

data = df_personas.copy()

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (data[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (data[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision
    (data[ESTIMATE] > ANNUAL_INTEREST_RATE) & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (data[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data["perception_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")
df_decisions = df_decisions.merge(
    df_personas[["participant.code", "treatment", "perception_pattern_12"]], how="left"
)
print(df_decisions.shape)

# print(data[data["participant.round"] == 1].value_counts("perception_pattern_12"))

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["perception_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "perception_pattern_12", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
### Expectations

MAX_RATIONAL_STOCK = 0
MONTH = 12
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (df_personas["previous_expectation"] <= ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision"
    (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (df_personas["previous_expectation"] > ANNUAL_INTEREST_RATE)
    & (df_personas["finalStock"] <= MAX_RATIONAL_STOCK),
]

df_personas[f"quant_expectation_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)
df_decisions = df_decisions.merge(
    df_personas[["participant.code", "treatment", "quant_expectation_pattern_12"]]
)
df_decisions = df_decisions[df_decisions["quant_expectation_pattern_12"] != "nan"]
print(df_decisions.shape)

# %%
MEASURES = ["previous_expectation"] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["quant_expectation_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "quant_expectation_pattern_12", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with qualitative expectations

MAX_RATIONAL_STOCK = 0
MONTH = 12
QUALITATIVE_EXPECTATION_THRESHOLD = 2
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([1, MONTH])]
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)

data = df_personas.copy()

CONDITIONS = [
    # Accurate estimate & Coherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
    # Accurate estimate & Incoherent decision
    (data["previous_expectation"] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Coherent decision"
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] > MAX_RATIONAL_STOCK),
    # Inaccurate estimate & Incoherent decision
    (data["previous_expectation"] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (data["finalStock"] <= MAX_RATIONAL_STOCK),
]

data[f"qual_expectation_pattern_12"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
)

# * Add column for persona based on MAX_RATIONAL_STOCK to track how distribution changes
df_personas = df_personas.merge(data, how="left")
df_decisions = df_decisions.merge(
    df_personas[["participant.code", "treatment", "qual_expectation_pattern_12"]]
)
df_decisions = df_decisions[df_decisions["qual_expectation_pattern_12"] != "nan"]

print(df_decisions.shape)

# %%
MEASURES = ["previous_expectation"] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["qual_expectation_pattern_12"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "qual_expectation_pattern_12", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with 12 months after first shock
# $(\Delta{\text{S}_{36}} > 0, E_{36} > 25)$ => Good expectations, coherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} > 25)$ => Good expectations, incoherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} <= 25)$ => Bad expectations, coherent decision
# $(\Delta{\text{S}_{36}} > 0, E_{36} <= 25)$ => Bad expectations, incoherent decision

# %%
END_MONTH = 36
START_MONTH = END_MONTH - 12
CHANGE_IN_STOCK = 1
ESTIMATE = "Quant Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([START_MONTH, END_MONTH])]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Accurate estimate & Coherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Accurate estimate & Incoherent decision
    (df_personas[ESTIMATE] > ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Coherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Inaccurate estimate & Incoherent decision
    (df_personas[ESTIMATE] <= ANNUAL_INTEREST_RATE)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["quant_expectation_pattern_36"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)
df_decisions = df_decisions.merge(
    df_personas[["participant.code", "treatment", "quant_expectation_pattern_36"]]
)
df_decisions = df_decisions[df_decisions["quant_expectation_pattern_36"] != "nan"]


print(df_decisions.shape)

# * Drop too N/A subjects
df_personas = df_personas[df_personas["quant_expectation_pattern_36"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "quant_expectation_pattern_36"
    )
)

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas.groupby(["quant_expectation_pattern_36", "treatment", "participant.round"])[
    MEASURES
].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "quant_expectation_pattern_36", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
#### Repeat with qualitative expectations
# $(\Delta{\text{S}_{36}} > 0, E_{36} > 1)$ => Good expectations, coherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} > 1)$ => Good expectations, incoherent decision
# $(\Delta{\text{S}_{36}} <= 0, E_{36} <= 1)$ => Bad expectations, coherent decision
# $(\Delta{\text{S}_{36}} > 0, E_{36} <= 1)$ => Bad expectations, incoherent decision

# %%
END_MONTH = 36
START_MONTH = END_MONTH - 12
CHANGE_IN_STOCK = 0
QUALITATIVE_EXPECTATION_THRESHOLD = 1
ESTIMATE = "Qual Expectation"

df_personas = df_decisions[df_decisions["Month"].isin([START_MONTH, END_MONTH])]
df_personas["previous_stock"] = df_personas.groupby("participant.code")[
    "finalStock"
].shift(1)
df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
    ESTIMATE
].shift(1)
df_personas = df_personas[df_personas["Month"] == END_MONTH]
df_personas["change_in_stock"] = (
    df_personas["finalStock"] - df_personas["previous_stock"]
)


CONDITIONS = [
    # Good expectations, coherent decision
    (df_personas[ESTIMATE] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
    # Good expectations, incoherent decision
    (df_personas[ESTIMATE] > QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, coherent decision
    (df_personas[ESTIMATE] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] <= CHANGE_IN_STOCK),
    # Bad expectations, incoherent decision
    (df_personas[ESTIMATE] <= QUALITATIVE_EXPECTATION_THRESHOLD)
    & (df_personas["change_in_stock"] > CHANGE_IN_STOCK),
]

df_personas["qual_expectation_pattern_36"] = np.select(
    condlist=CONDITIONS, choicelist=PERSONAS, default="N/A"
)
df_decisions = df_decisions.merge(
    df_personas[["participant.code", "treatment", "qual_expectation_pattern_36"]]
)
df_decisions = df_decisions[df_decisions["qual_expectation_pattern_36"] != "nan"]

print(df_decisions.shape)

# * Add column for persona based on stock to track how distribution change

# * Drop too N/A subjects
df_personas = df_personas[df_personas["qual_expectation_pattern_36"] != "N/A"]

print(
    df_personas[df_personas["participant.round"] == 1].value_counts(
        "qual_expectation_pattern_36"
    )
)

# %%
MEASURES = [ESTIMATE] + [MEASURE]
df_personas[df_personas["participant.round"] == 1].dropna().groupby(
    ["qual_expectation_pattern_36"]
)[MEASURES].describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]

# %%
figure = visualize_persona_results(
    df_personas, "qual_expectation_pattern_36", MEASURES, figsize=(20, 10)
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %% [markdown]
## Change in patterns
value_cols = [c for c in df_decisions.columns if ("pattern" in c) or ("_%" in c)]
pattern_changes = pd.pivot_table(
    df_decisions[df_decisions["Month"] == 120],
    values=value_cols,
    index=["participant.label", "treatment"],
    columns="participant.round",
    aggfunc="first",
)

for col in value_cols:
    if "pattern" in col:
        pattern_changes[f"change_{col}"] = (
            pattern_changes[(col, 1.0)] + pattern_changes[(col, 2.0)]
        )
    if "_%" in col:
        pattern_changes[f"change_{col}"] = (
            pattern_changes[(col, 2.0)] - pattern_changes[(col, 1.0)]
        )

# %%
pattern_cols = [c for c in pattern_changes.columns if c[1] != ""]

for col in pattern_cols:
    if "_%" not in col[0]:
        pattern_changes[col] = pd.Categorical(pattern_changes[col], PERSONAS)

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
    subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

    axs[i] = sns.histplot(
        subset,
        x=("perception_pattern_12", 1),
        y=("perception_pattern_12", 2),
        ax=axs[i],
        cbar=False,
    )

    axs[i] = annotate_2d_histogram(
        axs[i],
        ("perception_pattern_12", 1),
        ("perception_pattern_12", 2),
        subset,
        # fontweight="bold",
        color="black",
        fontsize=12,
    )
    axs[i].set_xlabel("Round 1")
    axs[i].set_ylabel("Round 2")
    axs[i].set_title(treatment)

fig.suptitle("Quant Perception, t=12")
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
    subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

    axs[i] = sns.histplot(
        subset,
        x=("quant_expectation_pattern_36", 1),
        y=("quant_expectation_pattern_36", 2),
        ax=axs[i],
        cbar=False,
    )

    axs[i] = annotate_2d_histogram(
        axs[i],
        ("quant_expectation_pattern_36", 1),
        ("quant_expectation_pattern_36", 2),
        subset,
        # fontweight="bold",
        color="black",
        fontsize=12,
    )
    axs[i].set_xlabel("Round 1")
    axs[i].set_ylabel("Round 2")
    axs[i].set_title(treatment)

fig.suptitle("Quant Expectation, t=36")
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
for i, treatment in zip(range(3), ["Intervention 2", "Intervention 1", "Control"]):
    subset = pattern_changes[pattern_changes.index.isin([treatment], level=1)]

    axs[i] = sns.histplot(
        subset,
        x=("qual_expectation_pattern_36", 1),
        y=("qual_expectation_pattern_36", 2),
        ax=axs[i],
        cbar=False,
    )

    axs[i] = annotate_2d_histogram(
        axs[i],
        ("qual_expectation_pattern_36", 1),
        ("qual_expectation_pattern_36", 2),
        subset,
        # fontweight="bold",
        color="black",
        fontsize=12,
    )
    axs[i].set_xlabel("Round 1")
    axs[i].set_ylabel("Round 2")
    axs[i].set_title(treatment)

fig.suptitle("Qual Expectation, t=36")
plt.show()

# %% [markdown]
## Performance measures
### Perceptions `t=12`
performance_results = create_performance_measures_table(
    df_decisions[
        (df_decisions["Month"] == 120) & (df_decisions["participant.round"] == 1)
    ],
    "perception_pattern_12",
    ["early_%", "sreal_%", "late_%", "excess_%"],
)
performance_results

# %% [markdown]
### Qualitative `t=12`
performance_results = create_performance_measures_table(
    df_decisions[
        (df_decisions["Month"] == 120) & (df_decisions["participant.round"] == 1)
    ],
    "qual_expectation_pattern_12",
    ["early_%", "sreal_%", "late_%", "excess_%"],
)
performance_results

# %% [markdown]
### Quantitative `t=36`
performance_results = create_performance_measures_table(
    df_decisions[
        (df_decisions["Month"] == 120) & (df_decisions["participant.round"] == 1)
    ],
    "quant_expectation_pattern_36",
    ["early_%", "sreal_%", "late_%", "excess_%"],
)
performance_results

# %% [markdown]
### Qualitative `t=36`
performance_results = create_performance_measures_table(
    df_decisions[
        (df_decisions["Month"] == 120) & (df_decisions["participant.round"] == 1)
    ],
    "qual_expectation_pattern_36",
    ["early_%", "sreal_%", "late_%", "excess_%"],
)
performance_results

# %% [markdown]
## Performance measures, changes in pattern
### Perceptions `t=12`
cols = [c for c in pattern_changes.columns if c[1] == ""]
df_changes = pattern_changes[cols].reset_index()
df_changes.columns = df_changes.columns.droplevel(level=1)

performance_cols = [
    c for c in df_changes.columns if ("_%" in c) and ("finalSavings" not in c)
]

performance_results = create_performance_measures_table(
    df_changes,
    "change_perception_pattern_12",
    performance_cols,
)
performance_results

# %% [markdown]
### Qualitative `t=12`
performance_results = create_performance_measures_table(
    df_changes,
    "change_qual_expectation_pattern_12",
    performance_cols,
)
performance_results

# %% [markdown]
### Quantitative `t=36`
performance_results = create_performance_measures_table(
    df_changes,
    "change_quant_expectation_pattern_36",
    performance_cols,
)
performance_results

# %% [markdown]
### Qualitative `t=36`
performance_results = create_performance_measures_table(
    df_changes,
    "change_qual_expectation_pattern_36",
    [c for c in df_changes.columns if ("_%" in c) and ("finalSavings" not in c)],
)
performance_results

# %% [markdown]
## Regressions
MONTHS = [1] + [m * 12 for m in range(1, 4)]
df_regression = df_decisions[df_decisions["Month"].isin(MONTHS)]
df_regression["previous_stock"] = df_regression.groupby("participant.code")[
    "finalStock"
].shift(1)

models = {
    "Quantitative Perception": "finalStock ~ Actual + Quant_Perception + previous_stock + Month",
    "Quantitative Expectation": "finalStock ~ Actual + Quant_Expectation + previous_stock + Month",
    "Qualitative Perception": "finalStock ~ Actual + Qual_Perception + previous_stock + Month",
    "Qualitative Expectation": "finalStock ~ Actual + Qual_Expectation + previous_stock + Month",
}

regressions = {}

data = df_regression[df_regression["participant.round"] == 1]
data = data.rename(
    columns={
        "Quant Perception": "Quant_Perception",
        "Quant Expectation": "Quant_Expectation",
        "Qual Perception": "Qual_Perception",
        "Qual Expectation": "Qual_Expectation",
    },
)

for estimate, formula in models.items():
    model = smf.ols(formula=formula, data=data)
    regressions[estimate] = model.fit()
results = summary_col(
    results=list(regressions.values()),
    stars=True,
    model_names=list(regressions.keys()),
)
results
