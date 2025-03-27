"""Tests for process_survey module"""

import logging
import math
import sys

import numpy as np

from utils import constants

from src.process_survey import (
    create_survey_df,
    pivot_inflation_measures,
    calculate_estimate_bias,
    calculate_estimate_sensitivity,
    include_inflation_measures,
    include_uncertainty_measure,
)
from utils.logging_config import get_logger


logger = get_logger(__name__)

logger.info("Testing process_survey module")
data = create_survey_df()

logger.info("Testing results calculated")
data = data[
    (data["participant.code"] == constants.TEST_CREATE_PARTICIPANT_CODE)
    & (data["Month"] == constants.TEST_CREATE_MONTH)
]
for k, v in constants.TEST_CREATE_VALUES.items():
    assert data[data["Measure"] == k]["Estimate"].iat[0] == v


logger.info("Testing pivot function")
data = create_survey_df()
df = pivot_inflation_measures(data)
logger.debug("df shape %s", df.shape)
assert df.shape[0] == constants.TEST_PIVOT_DF_LEN

result = df["Quant Expectation"].mean()
logger.debug("exp mean: %s compare to %s", result, constants.TEST_PIVOT_AVG_QUANT_EXP)
assert math.isclose(
    result,
    constants.TEST_PIVOT_AVG_QUANT_EXP,
    rel_tol=constants.TEST_PIVOT_ERROR_MARGIN,
)

logger.info("Testing estimate measure calculations")
for estimate, actual in zip(["Perception", "Expectation"], ["Actual", "Upcoming"]):
    df[f"{estimate}_bias"] = calculate_estimate_bias(df, f"Quant {estimate}", actual)
    _ = calculate_estimate_sensitivity(
        df, f"Quant {estimate}", actual, f"{estimate}_sensitivity"
    )
    df = df.merge(_, how="left")

logger.info("Testing bias calculation")
assert (
    df[
        (df["participant.code"] == constants.TEST_CALCULATE_BIAS_PARTICIPANT_CODE)
        & (df["Month"] == constants.TEST_CALCULATE_BIAS_MONTH)
    ]["Perception_bias"].values[0]
    == constants.TEST_CALCULATE_BIAS_PERCEPTION
)

assert (
    df[
        (df["participant.code"] == constants.TEST_CALCULATE_BIAS_PARTICIPANT_CODE)
        & (df["Month"] == constants.TEST_CALCULATE_BIAS_MONTH)
    ]["Expectation_bias"].values[0]
    == constants.TEST_CALCULATE_BIAS_EXPECTATION
)

logger.info("Testing sensitivity calculator")
assert (
    df[
        (
            df["participant.code"]
            == constants.TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE
        )
        & (df["Month"] == constants.TEST_CALCULATE_SENSITIVITY_MONTH)
    ]["Perception_sensitivity"].values[0]
    == constants.TEST_CALCULATE_SENSITIVITY_PERCEPTION_VALUE
)
result = df[
    (df["participant.code"] == constants.TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE)
    & (df["Month"] == constants.TEST_CALCULATE_SENSITIVITY_MONTH)
]["Expectation_sensitivity"].values[0]
logger.debug(
    "result:  %s, test value: %s",
    result,
    constants.TEST_CALCULATE_SENSITIVITY_EXPECTATION_VALUE,
)
assert math.isclose(
    result,
    constants.TEST_CALCULATE_SENSITIVITY_EXPECTATION_VALUE,
    rel_tol=constants.TEST_CALCULATE_SENSITIVITY_EXPECTATION_ERROR_MARGIN,
)

logger.info("Testing sensitivity calculator remove nans")
assert (
    df[
        (
            df["participant.code"]
            == constants.TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE_NO_NANS
        )
        & (df["Month"] == constants.TEST_CALCULATE_SENSITIVITY_MONTH)
    ]["Perception_sensitivity"].values[0]
    == constants.TEST_CALCULATE_SENSITIVITY_VALUE_NO_NANS
)

assert (
    df[
        (
            df["participant.code"]
            == constants.TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE_NO_NANS
        )
        & (df["Month"] == constants.TEST_CALCULATE_SENSITIVITY_MONTH)
    ]["Expectation_sensitivity"].values[0]
    == constants.TEST_CALCULATE_SENSITIVITY_VALUE_NO_NANS
)

logger.info("Testing include measures function")
df2 = pivot_inflation_measures(data)
df2 = include_inflation_measures(df2, fill_nans=False)
logger.debug("df2: %s df: %s", df2.shape, df.shape)
assert df2.shape == df.shape
assert (
    df2[
        (
            df2["participant.code"]
            == constants.TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE_NO_NANS
        )
        & (df2["Month"] == constants.TEST_CALCULATE_SENSITIVITY_MONTH)
    ]["Expectation_sensitivity"].values[0]
    != constants.TEST_CALCULATE_SENSITIVITY_VALUE_NO_NANS
)


logger.info(
    """Testing correction for qualitative-quantitative estimate matching 
            (making sure negative qualitative responses have negative quantitative 
            and vice versa)"""
)
df2 = pivot_inflation_measures(data)
count_quant_exp = df2["Quant Expectation"].count()
mean_quant_percp = df2["Quant Perception"].mean()
logger.debug("exp count: %s, mean percp: %s", count_quant_exp, mean_quant_percp)
assert count_quant_exp == constants.TEST_CREATE_QUANT_EXP_COUNT
assert math.isclose(
    mean_quant_percp,
    constants.TEST_CREATE_QUANT_PERCEP_AVG,
    rel_tol=constants.TEST_CALCULATE_SENSITIVITY_EXPECTATION_ERROR_MARGIN,
)

logger.info(
    """Testing corrections for mismatched qualitative, quantitative responses
            (i.e. negative qualitative, positive quantitative)"""
)

test_val = df2[
    (
        df2["participant.code"]
        == constants.TEST_CREATE_CORRECT_NEG_NEG_PARTICIPANT_CODE_PERC
    )
    & (df2["Month"] == constants.TEST_CREATE_CORRECT_NEG_NEG_MONTH_PERC)
]["Quant Perception"].values[0]
logger.debug("test val 1 %s", test_val)
assert test_val < 0

test_val = df2[
    (
        df2["participant.code"]
        == constants.TEST_CREATE_CORRECT_NEG_NEG_PARTICIPANT_CODE_EXP
    )
    & (df2["Month"] == constants.TEST_CREATE_CORRECT_NEG_NEG_MONTH_EXP)
]["Quant Expectation"].values[0]
logger.debug("test val 2 %s", test_val)
assert test_val < 0

test_val = df2[
    (df2["participant.code"] == constants.TEST_CREATE_CORRECT_POS_NEG_PARTICIPANT_CODE)
    & (df2["Month"] == constants.TEST_CREATE_CORRECT_POS_NEG_MONTH)
]["Quant Expectation"].values[0]
logger.debug("test val 3 %s", test_val)
assert test_val > 0

test_val = df2[
    (df2["participant.code"] == constants.TEST_CREATE_CORRECT_ZERO_PARTICIPANT_CODE)
    & (df2["Month"] == constants.TEST_CREATE_CORRECT_ZERO_MONTH)
]["Quant Perception"].values[0]
logger.debug("test val 4 %s", test_val)
assert test_val == 0

test_val = df2[
    (df2["participant.code"] == constants.TEST_CREATE_CORRECT_NEG_POS_PARTICIPANT_CODE)
    & (df2["Month"] == constants.TEST_CREATE_CORRECT_NEG_POS_MONTH)
]["Quant Expectation"].values[0]
logger.debug("test val 5 %s", test_val)
assert test_val < 0

logger.info("Testing uncertainty measure")
df2["Uncertain Expectation"] = include_uncertainty_measure(
    df2, "Quant Expectation", 1, 0
)
test_val = df2[
    (df2["participant.code"] == constants.TEST_UNCERTAINTY_MEASURE_PARTICIPANT_CODE)
    & (df2["Month"] == constants.TEST_UNCERTAINTY_MEASURE_MONTH)
]["Uncertain Expectation"].values[0]
assert test_val == constants.TEST_UNCERTAINTY_VALUE


logger.info("Test complete")
