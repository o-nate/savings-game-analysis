"""Reusable pipeline checkpoints matching scripts/results.py for CSV export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from scripts.utils import constants
from src import calc_opp_costs, decision_patterns, econ_preferences, intervention, knowledge, process_survey
from src.utils import exp_1_patches
from src.utils.constants import ANNUAL_INTEREST_RATE
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_series

# * Mirrors scripts/results.py (keep aligned when that script changes)
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
TREATMENT_GROUPS = [
    "Intervention 1",
    "Intervention 2",
    "Intervention 3",
]


@dataclass
class ResultsExportBundle:
    demographics_combined: pd.DataFrame
    inflation_estimates_combined: pd.DataFrame
    decisions_pre_classify_behavioral_patterns: pd.DataFrame
    decisions_all_enriched: pd.DataFrame
    knowledge_combined: pd.DataFrame
    econ_preferences_combined: pd.DataFrame
    behavioral_combined: pd.DataFrame
    learning_effect_initial: pd.DataFrame
    learning_effect_complete: pd.DataFrame
    treatment_effect: pd.DataFrame


def ensure_databases(con_exp_1: duckdb.DuckDBPyConnection, con_exp_2: duckdb.DuckDBPyConnection) -> None:
    if not table_exists(con_exp_1, "Questionnaire"):
        create_duckdb_database(con_exp_1, experiment=1, initial_creation=True)
    if not table_exists(con_exp_2, "Questionnaire"):
        create_duckdb_database(con_exp_2, experiment=2, initial_creation=True)


def _build_decisions_exp1(con_exp_1: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_questionnaire_1 = con_exp_1.sql("SELECT * FROM Questionnaire").df()
    df_expectations = con_exp_1.sql("SELECT * FROM inf_expectation").df()
    df_perceptions = con_exp_1.sql("SELECT * FROM inf_estimate").df()
    df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con_exp_1, experiment=1)
    df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
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
    df_decisions_1["finalSavings_120"] = (
        df_decisions_1[df_decisions_1["Month"] == 120]
        .groupby("participant.code")["finalSavings"]
        .transform("mean")
    )
    df_decisions_1["finalSavings_120"] = df_decisions_1.groupby("participant.code")[
        "finalSavings_120"
    ].bfill()
    return df_questionnaire_1, df_decisions_1, df_survey_1


def _build_decisions_exp2(con_exp_2: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_questionnaire = con_exp_2.sql("SELECT * FROM Questionnaire").df()
    df_questionnaire = df_questionnaire[
        df_questionnaire["participant.label"] != PARTICIPANT_WITHOUT_BELIEF_DATA
    ]
    df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con_exp_2, experiment=2)
    df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
    df_survey = process_survey.create_survey_df(include_inflation=True)
    df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
    df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
    df_inf_measures["participant.inflation"] = np.where(
        df_inf_measures["participant.inflation"] == "4x30", 430, 1012
    )
    df_inf_measures["Uncertain Expectation"] = process_survey.include_uncertainty_measure(
        df_inf_measures, "Quant Expectation", 1, 0
    )
    df_inf_measures["Average Uncertain Expectation"] = df_inf_measures.groupby(
        "participant.code"
    )["Uncertain Expectation"].transform("mean")
    df_decisions_2 = df_opp_cost.merge(df_inf_measures, how="left")
    df_decisions_2 = process_survey.separate_inflation_bias_by_phase(df_decisions_2)
    df_decisions_2["participant.day"] = df_decisions_2["participant.round"]
    df_decisions_2 = df_decisions_2[df_decisions_2["participant.inflation"] == 430]
    df_decisions_2["finalSavings_120"] = (
        df_decisions_2[df_decisions_2["Month"] == 120]
        .groupby("participant.code")["finalSavings"]
        .transform("mean")
    )
    df_decisions_2["finalSavings_120"] = df_decisions_2.groupby("participant.code")[
        "finalSavings_120"
    ].bfill()
    return df_questionnaire, df_decisions_2


def _mean_expectation_perception_bias(df_decisions_all: pd.DataFrame) -> None:
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


def _enrich_after_classify_subject(df_decisions_all: pd.DataFrame) -> pd.DataFrame:
    df_decisions_all = decision_patterns.classify_subject_decision_patterns(
        data=df_decisions_all,
        estimate_measure="Quant Perception",
        decision_measure="finalStock",
        month=12,
        coherent_decision=0,
        threshold_estimate=ANNUAL_INTEREST_RATE,
        drop_na=True,
    )
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
    df_decisions_all["Quant Perception_consistent_12"] = df_decisions_all[
        "Quant Perception_pattern_12"
    ].str[1]
    df_decisions_all["Quant Perception_consistent_12"] = np.where(
        df_decisions_all["Quant Perception_consistent_12"] == "C", 1, 0
    )
    df_decisions_all["purchase_adaptation_30"] = np.where(
        df_decisions_all["purchase_adaptation_30"] == "A", 1, 0
    )
    return df_decisions_all


def build_export_bundle(
    con_exp_1: duckdb.DuckDBPyConnection,
    con_exp_2: duckdb.DuckDBPyConnection,
) -> ResultsExportBundle:
    ensure_databases(con_exp_1, con_exp_2)

    df_questionnaire_1, df_decisions_1, _df_survey_1 = _build_decisions_exp1(con_exp_1)
    df_questionnaire, df_decisions_2 = _build_decisions_exp2(con_exp_2)

    demographics_combined = pd.concat(
        [df_questionnaire_1[QUESTIONNAIRE_COLS], df_questionnaire[QUESTIONNAIRE_COLS]]
    ).reset_index()
    demographics_combined = demographics_combined.drop(columns="index")

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
    inflation_estimates_combined = pd.concat(
        [df_inflation_estimates_1, df_inflation_estimates_2]
    )

    df_decisions_1["exp"] = 1
    df_decisions_2["exp"] = 2
    df_decisions_all = pd.concat([df_decisions_1, df_decisions_2]).reset_index()
    df_decisions_all = df_decisions_all[
        df_decisions_all["participant.label"] != PARTICIPANT_WITHOUT_BELIEF_DATA
    ]
    df_decisions_all["cum_decision_optimal"] = df_decisions_all.groupby(
        ["participant.code", "participant.inflation"]
    )["qoptimal"].cumsum()
    df_decisions_all["cum_decision_naive"] = df_decisions_all.groupby(
        ["participant.code", "participant.inflation"]
    )["qnaive"].cumsum()

    _mean_expectation_perception_bias(df_decisions_all)

    decisions_pre_classify_behavioral_patterns = df_decisions_all.copy()

    df_decisions_all = _enrich_after_classify_subject(df_decisions_all)
    decisions_all_enriched = df_decisions_all.copy()

    df_behavioral = df_decisions_all[
        (df_decisions_all["phase"] >= "pre")
        & (df_decisions_all["participant.inflation"] == 430)
    ]

    df_knowledge_1 = knowledge.create_knowledge_dataframe(con_exp_1)
    df_econ_preferences_1 = econ_preferences.create_econ_preferences_dataframe(con_exp_1)
    df_knowledge_1 = df_knowledge_1[df_knowledge_1["participant.day"] < 3]
    df_knowledge_1[["participant.round", "exp"]] = 1
    df_knowledge_1 = df_knowledge_1.drop(columns="participant.day")
    df_econ_preferences_1[["participant.round", "exp"]] = 1
    df_econ_preferences_1 = df_econ_preferences_1.drop(columns="participant.day")

    df_knowledge_2 = knowledge.create_knowledge_dataframe(con_exp_2)
    df_econ_preferences_2 = econ_preferences.create_econ_preferences_dataframe(con_exp_2)
    df_knowledge_2[["exp"]] = 2
    df_econ_preferences_2[["exp"]] = 2

    knowledge_combined = pd.concat([df_knowledge_1, df_knowledge_2]).reset_index()
    knowledge_combined = knowledge_combined.drop(columns="index")
    econ_preferences_combined = pd.concat(
        [df_econ_preferences_1, df_econ_preferences_2]
    ).reset_index()
    econ_preferences_combined = econ_preferences_combined.drop(columns="index")

    df_behavioral = combine_series(
        [df_behavioral, knowledge_combined, econ_preferences_combined],
        how="left",
        on=["participant.label", "participant.round", "exp"],
    )
    y_cols = [c for c in df_behavioral.columns if c.endswith("_y")]
    df_behavioral = df_behavioral.drop(columns=y_cols)
    df_behavioral.columns = df_behavioral.columns.str.removesuffix("_x")
    behavioral_combined = df_behavioral.copy()

    df_learn = decisions_all_enriched[
        (decisions_all_enriched["treatment"].isin(["control", "Control"]))
        & (decisions_all_enriched["participant.inflation"] == 430)
        & (decisions_all_enriched["Month"] == 120)
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

    learning_effect_initial, _ = intervention.create_learning_effect_table(
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
    learning_effect_initial = learning_effect_initial.set_index("")

    df_treat = decisions_all_enriched[
        (decisions_all_enriched["participant.inflation"] == 430)
        & (decisions_all_enriched["Month"] == 120)
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

    learning_effect_complete, _ = intervention.create_learning_effect_table(
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
    learning_effect_complete = learning_effect_complete.set_index("")

    return ResultsExportBundle(
        demographics_combined=demographics_combined,
        inflation_estimates_combined=inflation_estimates_combined,
        decisions_pre_classify_behavioral_patterns=decisions_pre_classify_behavioral_patterns,
        decisions_all_enriched=decisions_all_enriched,
        knowledge_combined=knowledge_combined,
        econ_preferences_combined=econ_preferences_combined,
        behavioral_combined=behavioral_combined,
        learning_effect_initial=learning_effect_initial,
        learning_effect_complete=learning_effect_complete,
        treatment_effect=treatment_effect,
    )


def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"
