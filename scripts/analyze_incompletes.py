"""Convert data to dataframes and create summary in Excel"""

# %%
import json
import os
from pathlib import Path
import re
from typing import List
import warnings

from dotenv import load_dotenv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.logging_config import get_logger

logger = get_logger(__name__)

# * Declare name of file to process
FILE = "all_apps_wide_2024-07-08.csv"

# * Declare name of output file
FINAL_FILE_PREFIX = "processed_"

# * File of incompletes from Exp 1
FILE_EXP_1_PATH = Path(__file__).parents[1] / "data" / "analyze_incompletes_exp1.csv"

# * Declare directory of output file
final_dir = Path(__file__).parents[1] / "data" / "preprocessed"

## Disable warnings
warnings.filterwarnings("ignore")

# turn off warning about df copies
pd.options.mode.chained_assignment = None  # default='warn'


# # Participant fields, player fields, and oTree APPS
participant_fields = [
    "code",
    "label",
    "instructions",
    "reaction_times",
    "periods_survived",
    "task_results",
    "task_results_1",
    "errors_1",
    "task_results_2",
    "errors_2",
    "lossAversion",
    "riskPreferences",
    "wisconsin",
    "remunerated_behavioral",
    "day_1",
    "inflation",
    "round",
    "day_room_number",
    "error",
    "last_room_day",
    "next_room",
    "err_msg",
    "vars_done",
]

APPS = [
    "participant",
    "init",
    "Questionnaire",
    "lossAversion",
    "riskPreferences",
    "wisconsin",
    "timePreferences",
    "Finance",
    "Inflation",
    "Numeracy",
    "task_instructions",
    "task",
    "task_int",
    "redirectapp",
    "sessionResults",
    "redirecttopayment",
]
logger.info("Processing APPS: %s", APPS)

# Define new tables per task fields
FIELDS = [
    "decision",
    "total_price",
    "initial_savings",
    "cashOnHand",
    "finalSavings",
    "finalStock",
    "interestEarned",
    "newPrice",
    "realInterest",
    "responseTime",
    "qualitative_estimate",
    "qualitative_expectation",
    "inf_estimate",
    "inf_expectation",
]
logger.info("Processing task FIELDS: %s", FIELDS)

COLUMNS_FOR_APPS = "participant.code|participant.label|date"
TASK_COLUMNS = "participant.inflation|participant.round|treatment"

# * Filters for testing dates
START_TS = "2024-04-30 00:00:00"
BETWEEN_TS_1 = "2024-05-03 00:00:00"
BETWEEN_TS_2 = "2024-05-03 23:59:59"
BETWEEN_TS_3 = "2024-05-07 00:00:00"
BETWEEN_TS_4 = "2024-06-20 00:00:00"
BETWEEN_TS_5 = "2024-07-02 00:00:00"
BETWEEN_TS_6 = "2024-07-02 23:59:59"

# * Filter participants who did not complete the entire experiment (total tasks complete)
EXP_TASK_COMPLETE = 12

# * Dates to define which treatment group subjects were assigned to
INTERVENTION_1_DATE = "2024-06-20"
INTERVENTION_2_DATE = "2024-07-02"

load_dotenv()
## Convert to json since the env variables get imported as str
LABELS = json.loads(os.getenv("LABELS"))


def remove_failed_tasks(df_to_correct: pd.DataFrame) -> List[str]:
    """Find participants that did not complete one or more rounds of the Savings Game"""
    to_remove = []
    for idx in range(len(df_to_correct)):
        if (
            df_to_correct["participant.task_results_1"].iat[idx] == 0
            or df_to_correct["participant.task_results_2"].iat[idx] == 0
        ):
            to_remove.append(df_to_correct["participant.label"].iat[idx])
    return to_remove


def remove_exp_incomplete(df_to_correct: pd.DataFrame) -> List[str]:
    """Find participants that did not complete experiment"""
    to_remove_dict = df_to_correct["participant.label"].value_counts().to_dict()
    to_remove = []
    for label, count in to_remove_dict.items():
        if count < EXP_TASK_COMPLETE:
            to_remove.append(label)
    return to_remove


def split_df(df_to_split: pd.DataFrame) -> tuple[list, dict]:
    """Generate separate df for each app + for participant info"""
    split_list = []
    split_dict = {}
    for test in APPS:
        split_list.append(test)
        if "task" in test:
            split_dict[test] = df_to_split.filter(
                regex=f"{COLUMNS_FOR_APPS}|{TASK_COLUMNS}|{test}."
            )
        else:
            split_dict[test] = df_to_split.filter(regex=f"{COLUMNS_FOR_APPS}|{test}.")
    return split_list, split_dict


def risk_responses(text: str) -> str:
    """Isolate responses from string"""
    pattern = r"\d{,} - "
    result = re.sub(pattern, "", text)
    return result


def loss_responses(text: str) -> str:
    """Remove trial id and '-' from dict key"""
    pattern_1 = r"\d{,} - "
    pattern_2 = r"\s₮"
    result = re.sub(pattern_1, "", text)
    result = re.sub(pattern_2, "", result)
    return result


def split_df_task(df_to_split: pd.DataFrame) -> tuple[list, dict]:
    """Create separate dataframes for each Savings Game measure"""
    task_split_list = []
    task_split_dict = {}
    for field in FIELDS:
        task_split_list.append(field)
        task_split_dict[field] = df_to_split.filter(
            regex=f"{COLUMNS_FOR_APPS}|{TASK_COLUMNS}|{field}$"
        )
    return task_split_list, task_split_dict


# %%
# * Create initial dataframe with all data
parent_dir = Path(__file__).parents[1]
data_dir = parent_dir / "data"
logger.info("Data from directory %s", data_dir)
complete = pd.read_csv(f"{data_dir}/{FILE}")
logger.info("Initial dataframe size: %s", complete.shape)

# * Remove rows participant.label = NaN
complete = complete[complete["participant.label"].notna()]
logger.info(
    "Total participants who started the experiment: %s",
    complete["participant.label"].nunique(),
)
logger.info("Removing participants who DID finish or had internet connection issues.")

# * Remove columns with all NaN
complete = complete.dropna(axis=1, how="all")

# * Remove questionnable participants
complete = complete[~complete["participant.label"].isin(LABELS)]

# * Convert participant.time_started_utc value to datetime
complete["participant.time_started_utc"] = pd.to_datetime(
    complete["participant.time_started_utc"]
)

# * FILTER BY DATE
logger.info(
    "Filtering test dates: Before %s, between %s and %s, and after %s.",
    START_TS,
    BETWEEN_TS_1,
    BETWEEN_TS_2,
    BETWEEN_TS_3,
)
complete = complete[complete["participant.time_started_utc"] >= START_TS]
complete = complete[
    (complete["participant.time_started_utc"] < BETWEEN_TS_1)
    | (
        (complete["participant.time_started_utc"] > BETWEEN_TS_2)
        & (complete["participant.time_started_utc"] < BETWEEN_TS_3)
    )
    | (
        (complete["participant.time_started_utc"] > BETWEEN_TS_4)
        & (complete["participant.time_started_utc"] < BETWEEN_TS_5)
    )
    | (complete["participant.time_started_utc"] > BETWEEN_TS_6)
]

incomplete_exp_participants = remove_exp_incomplete(complete)
logger.debug(
    "%s participants removed for not completing full experiment: %s",
    len(set(incomplete_exp_participants)),
    incomplete_exp_participants,
)

# * Remove participants who did not finish Savings Game
participants_to_remove = remove_failed_tasks(
    complete[~complete["participant.label"].isin(incomplete_exp_participants)]
)
logger.info(
    "Removing %s participants for not completing task: %s",
    len(set(participants_to_remove)),
    participants_to_remove,
)

complete = complete[
    complete["participant.label"].isin(
        incomplete_exp_participants + participants_to_remove
    )
]
print(complete.shape)

# * Remove subjects with day_1 nans
complete = complete[~complete["participant.day_1"].isna()]

## Convert tasks for day to list object
complete["participant.day_1"] = complete["participant.day_1"].apply(eval)

## Use list comprehension for optimization given small dataset
complete["participant.intervention"] = [
    True if "task_int" in day_tests else False
    for day_tests in complete["participant.day_1"]
]

# * Create date column for easier filtering
complete["date"] = complete["participant.time_started_utc"].dt.normalize()

# * Determine which treatment group each subject was in
complete["treatment"] = [
    (
        "Intervention 1"
        if x < pd.Timestamp(INTERVENTION_1_DATE)
        else ("Intervention 2" if x < pd.Timestamp(INTERVENTION_2_DATE) else "Control")
    )
    for x in complete["date"]
]

# organize rows by participant.label and display corresponding codes
complete = complete.sort_values(
    ["participant.label", "participant.time_started_utc"], ascending=[False, True]
)
participant = complete[
    ["participant.label", "participant.code", "participant.time_started_utc"]
]

participant = participant.sort_values(
    ["participant.label", "participant.time_started_utc"], ascending=[False, True]
)

# * Split into separate dataframes from each app and measure
df_list, df_dict = split_df(complete)


# participant df
df_dict["participant"].drop("participant.task_results", axis=1, inplace=True)
df_dict["participant"].drop("participant._is_bot", axis=1, inplace=True)

# %%
# * Analyze subjects who did not complete experiment
incompletes = complete[
    complete["participant.label"].isin(incomplete_exp_participants)
].dropna(axis=0, how="all")
print(incompletes.shape)

# %%
# * Analyse subjects who (intentionally) failed the SG round
failures = complete[complete["participant.label"].isin(participants_to_remove)].dropna(
    axis=0, how="all"
)

# %%
DEMOGRAPHICS = [
    "Questionnaire.1.player.age",
    "Questionnaire.1.player.gender",
    "Questionnaire.1.player.educationLevel",
    "Questionnaire.1.player.employmentStatus",
]
questionnaire = df_dict["Questionnaire"].copy()
questionnaire = questionnaire[
    ~questionnaire["Questionnaire.1.player.id_in_group"].isna()
]
print(questionnaire.shape)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, col in enumerate(DEMOGRAPHICS):
    sns.histplot(data=questionnaire[col], ax=axes[i], stat="percent")


# %%
questionnaire[DEMOGRAPHICS].describe().T[["mean", "std", "50%"]]

# %%
# * Combine with Exp 1 data
incompletes_exp1 = pd.read_csv(FILE_EXP_1_PATH)
incompletes_exp1["experiment"] = 1
questionnaire["experiment"] = 2

combined = pd.concat(
    [
        incompletes_exp1[["participant.label", "experiment"] + DEMOGRAPHICS],
        questionnaire[["participant.label", "experiment"] + DEMOGRAPHICS],
    ]
)

# %%
combined_melt = pd.melt(
    combined,
    id_vars=["participant.label", "experiment"],
    value_vars=DEMOGRAPHICS,
    var_name="measure",
)

# %%
plot_titles = {
    "Questionnaire.1.player.age": "Age",
    "Questionnaire.1.player.gender": "Gender",
    "Questionnaire.1.player.educationLevel": "Education Level",
    "Questionnaire.1.player.employmentStatus": "Employment",
}
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, col in enumerate(DEMOGRAPHICS):
    print(combined_melt[combined_melt["measure"] == col].shape)
    sns.histplot(
        data=combined[col],
        ax=axes[i],
        stat="percent",
        palette="tab10",
    )
    axes[i].set_xlabel(plot_titles[col], fontsize=20)
    axes[i].set_ylabel("Percent", fontsize=20)
    if i > 0:
        axes[i].set_ylabel(None)
