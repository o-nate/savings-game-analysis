"""Convert data to dataframes and create summary in Excel"""

import json
import os
from pathlib import Path
import re
import time
import warnings

from dotenv import load_dotenv
import pandas as pd
from openpyxl import load_workbook
from tqdm import tqdm

from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Declare name of file to process
FILE = "all_apps_wide_2024-07-08.csv"

# * Declare name of output file
FINAL_FILE_PREFIX = "processed_"

# * Declare directory of output file
final_dir = Path(__file__).parents[1] / "data" / "preprocessed"

warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

# Participant fields, player fields, and oTree APPS
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


def remove_failed_tasks(df_to_correct: pd.DataFrame) -> list[str]:
    """Find participants that did not complete one or more rounds of the Savings Game"""
    to_remove = []
    for idx in range(len(df_to_correct)):
        if (
            df_to_correct["participant.task_results_1"].iat[idx] == 0
            or df_to_correct["participant.task_results_2"].iat[idx] == 0
        ):
            to_remove.append(df_to_correct["participant.label"].iat[idx])
    return to_remove


def remove_exp_incomplete(df_to_correct: pd.DataFrame) -> list[str]:
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


def preprocess_data() -> dict[str, pd.DataFrame]:
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
    logger.info(
        "Removing participants who did not finish or had internet connection issues."
    )

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

    # * Remove participants who did not finish Savings Game
    participants_to_remove = remove_failed_tasks(complete)
    logger.info(
        "Removing %s participants for not completing task: %s",
        len(set(participants_to_remove)),
        participants_to_remove,
    )
    complete = complete[~complete["participant.label"].isin(participants_to_remove)]

    incomplete_exp_participants = remove_exp_incomplete(complete)
    logger.debug(
        "%s participants removed for not completing full experiment: %s",
        len(set(incomplete_exp_participants)),
        incomplete_exp_participants,
    )
    complete = complete[
        ~complete["participant.label"].isin(incomplete_exp_participants)
    ]
    logger.info(
        "Total participants with complete: %s",
        complete["participant.label"].nunique(),
    )

    # TODO Define whether participant was in intervention or control group
    # TODO Can do so without needing to change oTree table in the future by using
    # TODO `participant._index_in_pages` for task app with intervention app,
    # TODO where `intervention` participants have a higher value
    ## Not working currently since task_int not in day_1 list (part of task_1 app)
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
            else (
                "Intervention 2" if x < pd.Timestamp(INTERVENTION_2_DATE) else "Control"
            )
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

    # wisconin df
    df_dict["wisconsin"].drop("session.config.wisconsin_fee", axis=1, inplace=True)

    # task
    df_dict["task"].drop(
        list(
            df_dict["task"].filter(
                regex="participant.task|session|task_int|task_instructions"
            )
        ),
        axis=1,
        inplace=True,
    )

    # # Remove blank rows
    # (i.e. rows for other APPS)

    # for app in tqdm(df_list, desc='Removing blank rows from APPS'):
    for app in df_list:
        if app != "participant":
            df_dict[app].dropna(subset=[f"{app}.1.player.id_in_group"], inplace=True)

    # # `timePreferences`: Distribute delays into individual columns for each round
    new_timePreferences = pd.DataFrame(
        data=df_dict["timePreferences"]["timePreferences.1.player.delay"].tolist()
    )

    new_timePreferences.rename(columns={0: "delay_order"}, inplace=True)

    # convert str to dict
    new_timePreferences.dicts = new_timePreferences.delay_order.apply(json.loads)
    new_timePreferences["dict_list"] = new_timePreferences.dicts

    for n in range(len(new_timePreferences["dict_list"][0]["delay"])):
        new_timePreferences[f"{n}"] = n

    for i in range(len(new_timePreferences["dict_list"])):
        new_timePreferences["dict_list"][i] = new_timePreferences["dict_list"][i][
            "delay"
        ]

    for k in range(len(new_timePreferences["dict_list"])):
        for j in range(len(new_timePreferences["dict_list"][0])):
            new_timePreferences[f"{j}"][k] = new_timePreferences["dict_list"][k][j][
                "delay"
            ]
            new_timePreferences[f"{j}"] = new_timePreferences[f"{j}"]

    new_new = new_timePreferences.iloc[:, 2:]

    for k in range(len(new_new.index)):
        for j in range(1, 1 + len(new_new.columns)):
            df_dict["timePreferences"][f"timePreferences.{j}.player.delay"].iloc[k] = (
                new_new.iloc[k][j - 1]
            )

    # # `riskPreferences`: Convert `raw_responses` to columns

    new_riskPreferences = pd.DataFrame(
        data=df_dict["riskPreferences"][
            "riskPreferences.1.player.raw_responses"
        ].tolist()
    )
    new_riskPreferences.rename(columns={0: "responses"}, inplace=True)

    # remove trial id and '-' from dict key
    new_riskPreferences["responses"] = new_riskPreferences.responses.map(risk_responses)

    # convert str to dict
    new_riskPreferences.dicts = new_riskPreferences.responses.apply(json.loads)
    new_riskPreferences["responses"] = new_riskPreferences.dicts

    # convert dict key:value to columns
    new_riskPreferences = pd.concat(
        [new_riskPreferences, new_riskPreferences["responses"].apply(pd.Series)], axis=1
    )

    # reorder columns
    col_order = []
    for i in range(1, 11):
        i = i * 10
        col_order.append(f"{i}")

    new_riskPreferences = new_riskPreferences[col_order]

    # recombine with riskPreferences df
    for j in range(1, 11):
        k = j * 10
        df_dict["riskPreferences"][f"riskPreferences.player.probability.{k}"] = (
            new_riskPreferences[f"{k}"].values
        )

    df_dict["riskPreferences"].head()

    # # `lossAversion`: Distribute `raw_responses` to columns

    new_lossAversion = pd.DataFrame(
        data=df_dict["lossAversion"]["lossAversion.1.player.raw_responses"].tolist()
    )
    new_lossAversion.rename(columns={0: "responses"}, inplace=True)

    new_lossAversion["responses"] = new_lossAversion.responses.map(loss_responses)

    # convert str to dict
    new_lossAversion.dicts = new_lossAversion.responses.apply(json.loads)
    new_lossAversion["responses"] = new_lossAversion.dicts

    # convert dict key:value to columns
    new_lossAversion = pd.concat(
        [new_lossAversion, new_lossAversion["responses"].apply(pd.Series)], axis=1
    )

    # reorder columns
    col_order = []
    for i in range(2, 8):
        i = i * 100 * 2  # multiply by 2 because double lottery sizes
        col_order.append(f"{i},00")

    new_lossAversion = new_lossAversion[col_order]

    # recombine with lossAversion df
    for j in range(2, 8):
        k = j * 100 * 2  # multiply by 2 because double lottery sizes
        df_dict["lossAversion"][f"lossAversion.player.loss.{k}"] = new_lossAversion[
            f"{k},00"
        ].values

    df_dict["lossAversion"].head()

    # Unpack `wisconsin` dict
    df_wisc = df_dict["wisconsin"].copy()

    # convert str to dict
    df_wisc.dicts = df_wisc["wisconsin.1.player.response_time"].apply(json.loads)
    df_wisc["wisconsin.1.player.response_time"] = df_wisc.dicts

    # convert dict key:value to columns
    df_wisc = pd.concat(
        [df_wisc, df_wisc["wisconsin.1.player.response_time"].apply(pd.Series)], axis=1
    )

    # * Break dicts into columns for each trial
    new = df_wisc.iloc[:, :1]

    new_dfs_dict = {"new": new["participant.code"].reset_index()}
    for col in range(1, 31):
        new[col] = df_wisc[f"{col}"]
        ## Convert list to string
        new[col] = [",".join(map(str, l)) for l in new[col]]
        ## Convert string to dict
        new[col] = new[col].apply(json.loads)
        new_dfs_dict[col] = pd.json_normalize(new[col])

    ## Store general columns to add suffix for trial number
    col_list = list(new[1].iat[0].keys())

    ## Merge dataframes
    new = pd.concat(new_dfs_dict.values(), axis=1)

    ## Add suffixes
    logger.debug("new columns %s", new.columns)
    new.drop("index", axis=1, inplace=True)
    new.columns = ["participant.code"] + [
        f"{c}_{i}" for i in range(1, 31) for c in col_list
    ]

    # concatenate new columns with df_wisc
    df_wisc = df_wisc.merge(new, how="left")
    logger.debug("df_wisc shape %s", df_wisc.shape)

    # remove separated trial number columns
    columns = []
    for col in range(1, 31):
        df_wisc.drop(str(col), axis=1, inplace=True)

    # update original df
    df_dict["wisconsin"] = df_wisc

    # # `task`: Further clean data
    task = df_dict["task"].copy()

    # Drop columns with 'day_'
    task.drop(task.filter(regex="day_").columns, axis=1, inplace=True)

    # Resort by index
    task.sort_index(inplace=True)

    # Convert 'participant.inflation' to list and extract corresponding day's inf sequence
    task["participant.inflation"] = task["participant.inflation"].apply(eval)
    for n in range(len(task.index)):
        round_num = task["participant.round"].iloc[n]
        task["participant.inflation"].iloc[n] = task["participant.inflation"].iloc[n][
            int(round_num) - 1
        ]

    task_df_list, task_df_dict = split_df_task(task)

    # Extract quantity purchased per month
    decision = task_df_dict["decision"].copy()

    # Replace NaNs in decision
    decision.fillna('{"item": [], "quantity": [], "price": []}', inplace=True)

    for month in tqdm(range(1, 121), desc="Extracting decision quantities"):
        col = f"task.{month}.player.decision"
        # Convert str to dict
        decision[col] = decision[col].apply(json.loads)
        ## Convert row value to quantity value
        decision.loc[decision[col].notnull(), col] = decision.loc[
            decision[col].notnull(), col
        ].apply(lambda x: x.get("quantity"))
        ## Extract value from list
        decision.loc[decision[col].notnull(), col] = decision.loc[
            decision[col].notnull(), col
        ].str[0]
        decision[col] = decision[col].fillna(0)

    task_df_dict["decision"] = decision

    ## Add df's to general dict and and list
    final_df_dict = df_dict | task_df_dict
    final_df_list = df_list + task_df_list

    # # Calculate payment results

    final_payments = df_dict["participant"].loc[
        (df_dict["participant"]["participant._current_app_name"] == "redirecttopayment")
    ]
    final_payments = final_payments[
        [
            "participant.label",
            "participant.payoff",
            "participant.task_results_1",
            "participant.task_results_2",
            "participant.day_1",
            "participant.remunerated_behavioral",
        ]
    ]

    # convert from str to list
    # final_payments["participant.day_1"] = final_payments["participant.day_1"].map(eval)
    final_payments["participant.remunerated_behavioral"] = final_payments[
        "participant.remunerated_behavioral"
    ].map(eval)

    # remove non-behavioral task names from lists
    final_payments["participant.day_1"] = final_payments["participant.day_1"].apply(
        lambda x: [i for i in x if i != "debut"]
    )

    # # generate columns for each task with corresponding results
    # final_payments = pd.concat(
    #     [final_payments, final_payments["results_dict1"].apply(pd.Series)], axis=1
    # )

    # * Convert remunerated_behavioral dict to columns
    final_payments = pd.concat(
        [
            final_payments.drop(["participant.remunerated_behavioral"], axis=1),
            final_payments["participant.remunerated_behavioral"].apply(pd.Series),
        ],
        axis=1,
    )

    final_payments.loc["Total"] = final_payments[
        [
            "participant.payoff",
            "participant.task_results_1",
            "participant.task_results_2",
            "wisconsin",
            "risk_preferences",
            "loss_aversion",
        ]
    ].sum()
    final_payments = final_payments.drop(
        [
            "participant.day_1",
        ],
        axis=1,
    )
    final_payments = final_payments.fillna("-")
    final_df_list.append("final_payments")
    final_df_dict["final_payments"] = final_payments

    stats_final_payments = final_payments[
        final_payments["participant.label"] != "-"
    ].describe()
    final_df_list.append("stats_final_payments")
    final_df_dict["stats_final_payments"] = stats_final_payments

    logger.info(
        "Complete. Total participants included: %s", (final_payments.shape[0] - 1)
    )

    if __name__ == "__main__":
        export_results = input("Would you like to export the results to Excel? (y/n):")
        if export_results not in ("y", "n"):
            export_results = input("Please, respond by typing y or n:")
        if export_results == "y":
            ## Excel of performance per session
            timestr = time.strftime("%Y%m%d-%H%M%S")
            logger.info(timestr)
            with pd.ExcelWriter(
                f"{final_dir}/{FINAL_FILE_PREFIX}_{timestr}.xlsx"
            ) as writer:
                participant.to_excel(writer, sheet_name="participant")
                final_df_dict["final_payments"].to_excel(
                    writer, sheet_name="final_payments"
                )
                for df in final_df_dict:
                    final_df_dict[df].to_excel(writer, sheet_name=f"{df}")
                    logger.info("Adding: %s", df)

        logger.info("done")

    return final_df_dict


def main() -> None:
    results = preprocess_data()


if __name__ == "__main__":
    main()
