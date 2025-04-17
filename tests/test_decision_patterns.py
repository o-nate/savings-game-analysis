"""Classify subjects by pattern of perceptions/expectations and decisions"""

from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from src import calc_opp_costs, process_survey

from src.decision_patterns import (
    classify_subject_decision_patterns,
    define_decision_patterns,
)

from src.utils.constants import ANNUAL_INTEREST_RATE, EXP_2_DATABASE
from src.utils.plotting import (
    annotate_2d_histogram,
    create_performance_measures_table,
    visualize_persona_results,
)

from tests.utils import constants

from utils.logging_config import get_logger

# * Set logger
logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE

con = duckdb.connect(DATABASE_FILE, read_only=False)

df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con)

df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
df_opp_cost.head()

df_survey = process_survey.create_survey_df(include_inflation=True)
df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
df_inf_measures["participant.inflation"] = np.where(
    df_inf_measures["participant.inflation"] == "4x30", 430, 1012
)

df_decisions = df_opp_cost.merge(df_inf_measures, how="left")
logger.debug(df_decisions.shape)

# * Store final savings at month t = 120
df_decisions["finalSavings_120"] = (
    df_decisions[df_decisions["Month"] == 120]
    .groupby("participant.code")["finalSavings"]
    .transform("mean")
)
df_decisions["finalSavings_120"] = df_decisions.groupby("participant.code")[
    "finalSavings_120"
].bfill()

logger.info("Testing decision classifier")

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
    threshold_estimate=ANNUAL_INTEREST_RATE,
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
    threshold_estimate=ANNUAL_INTEREST_RATE,
)
logger.debug(df_decisions.shape)

assert df_decisions.shape[0] == constants.TOTAL_ROWS

logger.info("Test complete")
