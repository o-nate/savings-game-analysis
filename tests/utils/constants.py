"""Constants for testing modules"""

# * `preprocess`
MEAN_TRIAL_NUMBER = 30
NCORR_TRIAL_NUMBER_5 = 105

# * `econ_preferences`
LOSS_SWITCHES_PARTICIPANT_CODE = "ub8goyln"
LOSS_SWITCHES_NUMBER = 3
RISK_SAFE_PARTICIPANT_CODE = "0thsjvpw"
RISK_SAFE_NUMBER = 4
RISK_SWITCHES_NUMBER = 4
TIME_PRESENT_PARTICIPANT_CODE = "oqiu2d4t"
TIME_PRESENT_NUMBER = 11
TIME_PRESENT_SWITCHES = 12
WISC_PARTICIPANT_CODE = "m16ef08d"
WISC_N_CORR = 12
WISC_N_PE = 3
WISC_N_SE = 5
DATAFRAME_SHAPE = (154, 10)

# * `intervention`
CHANGE_IN_PERFORMANCE = {
    "Intervention 1": {
        "Initial Total savings": 2161.29,
        "Final Total savings": 2497.76,
        "Initial Over-stocking": 769.30,
        "Final Over-stocking": 708.57,
        "Initial Under-stocking": 688.69,
        "Final Under-stocking": 690.13,
        "Initial Wasteful-stocking": 498.12,
        "Final Wasteful-stocking": 221.41,
    },
    "Intervention 2": {
        "Initial Total savings": 2409.73,
        "Final Total savings": 2670.13,
        "Initial Over-stocking": 796.92,
        "Final Over-stocking": 456.94,
        "Initial Under-stocking": 749.09,
        "Final Under-stocking": 860.30,
        "Initial Wasteful-stocking": 161.69,
        "Final Wasteful-stocking": 130.78,
    },
    "Control": {
        "Initial Total savings": 2017.71,
        "Final Total savings": 1958.00,
        "Initial Over-stocking": 810.69,
        "Final Over-stocking": 1432.16,
        "Initial Under-stocking": 815.44,
        "Final Under-stocking": 499.98,
        "Initial Wasteful-stocking": 473.93,
        "Final Wasteful-stocking": 227.80,
    },
}

# * `knowledge`
SCORE = 1
FIN_LIT_PARTICIPANT_CODE = "xsfez6ci"
NUMERACY_PARTICIPANT_CODE_2B = "39s9z7uj"
NUMERACY_PARTICIPANT_CODE_3 = "k7f2uopj"
NUMERACY_PARTICIPANT_CODE_3_NOT = "g7zm7yjh"
COMPOUND_PARTICIPANT_CODE = "glsbpy7y"
KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL = "9xHTKNJ"
KNOWLEDGE_DATAFRAME_FIN_LIT_SCORE = 1
KNOWLEDGE_DATAFRAME_NUM_SCORE = 0
KNOWLEDGE_DATAFRAME_COMPOUND_SCORE = 0


# * `process_survey`
TEST_CREATE_PARTICIPANT_CODE = "17c9d4zc"
TEST_CREATE_MONTH = 36
TEST_CREATE_VALUES = {
    "Quant Expectation": 10.0,
    "Quant Perception": 3.0,
    "Qual Expectation": 3.0,
    "Qual Perception": 2.0,
}
## Negative qualitative, negative quantitative
TEST_CREATE_CORRECT_NEG_NEG_PARTICIPANT_CODE_EXP = "rvq1b1xn"  # Expectation
TEST_CREATE_CORRECT_NEG_NEG_MONTH_EXP = 48
TEST_CREATE_CORRECT_NEG_NEG_PARTICIPANT_CODE_PERC = "b18fltit"  # Perception
TEST_CREATE_CORRECT_NEG_NEG_MONTH_PERC = 84
## Positive qualitative, negative quantitative
TEST_CREATE_CORRECT_POS_NEG_PARTICIPANT_CODE = "v5ibqv7b"
TEST_CREATE_CORRECT_POS_NEG_MONTH = 12
## Qualitative - no change
TEST_CREATE_CORRECT_ZERO_PARTICIPANT_CODE = "t3lco71m"  # Perception
TEST_CREATE_CORRECT_ZERO_MONTH = 72
## Negative qualitative, positive quantitative
TEST_CREATE_CORRECT_NEG_POS_PARTICIPANT_CODE = "lgvilqkk"  # Expectation
TEST_CREATE_CORRECT_NEG_POS_MONTH = 1


TEST_CREATE_QUANT_EXP_COUNT = 3051.0
TEST_CREATE_QUANT_PERCEP_AVG = 15.546

TEST_PIVOT_DF_LEN = 3388
TEST_PIVOT_AVG_QUANT_EXP = 11.9477
TEST_PIVOT_ERROR_MARGIN = 0.001

TEST_CALCULATE_BIAS_PARTICIPANT_CODE = "01hv4wn4"
TEST_CALCULATE_BIAS_MONTH = 96
TEST_CALCULATE_BIAS_VALUES = {
    "Quant Expectation": 10.0,
    "Quant Perception": 15.0,
    "Actual": 26.85,
    "Upcoming": 55.49,
}
TEST_CALCULATE_BIAS_PERCEPTION = (
    TEST_CALCULATE_BIAS_VALUES["Quant Perception"]
    - TEST_CALCULATE_BIAS_VALUES["Actual"]
)

TEST_CALCULATE_BIAS_EXPECTATION = (
    TEST_CALCULATE_BIAS_VALUES["Quant Expectation"]
    - TEST_CALCULATE_BIAS_VALUES["Upcoming"]
)

TEST_CALCULATE_SENSITIVITY_MONTH = 96
TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE = "k78vuxz0"
TEST_CALCULATE_SENSITIVITY_PERCEPTION_VALUE = 0
TEST_CALCULATE_SENSITIVITY_EXPECTATION_VALUE = -0.3626381
TEST_CALCULATE_SENSITIVITY_EXPECTATION_ERROR_MARGIN = 0.0002

TEST_CALCULATE_SENSITIVITY_PARTICIPANT_CODE_NO_NANS = "3lkzrdod"
TEST_CALCULATE_SENSITIVITY_VALUE_NO_NANS = 0

TEST_UNCERTAINTY_MEASURE_PARTICIPANT_CODE = "t1zyxzdr"
TEST_UNCERTAINTY_MEASURE_MONTH = 12
TEST_UNCERTAINTY_VALUE = 1
