"""Constants for scripts modules"""

from pathlib import Path

from src.utils.constants import EXP_2_DATABASE

DATABASE_FILE = Path(__file__).parents[2] / "data" / EXP_2_DATABASE

QUESTIONNAIRE_MEASURES = [
    "age",
    "gender",
    "educationLevel",
    "employmentStatus",
    "financialStatusIncome",
    "financialStatusSavings_1",
    "financialStatusSavings_2",
    "financialStatusDebt_1",
    "financialStatusDebt_2",
    "stocks",
    "mutualFunds",
    "bonds",
    "savingsAccounts",
    "lifeInsurance",
    "retirementAccounts",
    "crypto",
]

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

PERFORMANCE_MEASURES_OLD_NAMES = ["sreal_%", "early_%", "excess_%"]
PERFORMANCE_MEASURES_NEW_NAMES = ["Savings", "Over-stocking", "Wasteful-stocking"]
PURCHASE_ADAPTATION_OLD_NAME = ["avg_q"]
PURCHASE_ADAPTATION_NEW_NAME = ["Purchase adaptation"]

INFLATION_RESULTS_MEASURES = [
    "Quant Expectation",
    "Quant Perception",
    "Qual Perception_low",
    "Qual Perception_high",
    "Qual Expectation_low",
    "Qual Expectation_high",
    "Avg Qual Perception Accuracy",
    "Avg Qual Expectation Accuracy",
    "Average Uncertain Expectation",
    # "Perception_bias",
    "Perception_bias_high",
    "Perception_bias_low",
    "Perception_sensitivity",
    # "Expectation_bias",
    "Expectation_bias_high",
    "Expectation_bias_low",
    "Expectation_sensitivity",
]

P_VALUE_THRESHOLDS = [0.1, 0.05, 0.01]
DECIMAL_PLACES = 2

PERFORMANCE_MEASURES = PERFORMANCE_MEASURES_OLD_NAMES + ["avg_q_%"]

QUANT_INFLATION_MEASURES = [
    "Perception_sensitivity",
    "avg_perception_bias",
    "Expectation_sensitivity",
    "avg_expectation_bias",
]
QUAL_INFLATION_MEASURES = [
    "Avg Qual Expectation Accuracy",
    "Avg Qual Perception Accuracy",
    "Average Uncertain Expectation",
]
KNOWLEDGE_MEASURES = ["financial_literacy", "numeracy", "compound"]
ECON_PREFERENCE_MEASURES = [
    "lossAversion_choice_count",
    "lossAversion_switches",
    "riskPreferences_choice_count",
    "riskPreferences_switches",
    "timePreferences_choice_count",
    "timePreferences_switches",
    "wisconsin_choice_count",
    "wisconsin_PE",
    "wisconsin_SE",
]
TREATMENTS = ["Intervention 1", "Intervention 2", "Control"]
ADAPTATION_MONTHS = [12, 24, 36, 48]
INDIVIDUAL_CHARACTERISTICS = [
    "financial_literacy",
    "numeracy",
    "compound",
    "wisconsin_choice_count",
    "riskPreferences_choice_count",
    "riskPreferences_switches",
    "lossAversion_choice_count",
    "lossAversion_switches",
    "timePreferences_choice_count",
    "timePreferences_switches",
]

# * Forward selection feature variables
FEATURES = [
    "financial_literacy",
    "numeracy",
    "compound",
    "wisconsin_choice_count",
    "riskPreferences_choice_count",
    # "riskPreferences_switches",
    "lossAversion_choice_count",
    # "lossAversion_switches",
    "timePreferences_choice_count",
    # "timePreferences_switches",
    "age",
    "gender",
    "educationLevel",
    "financialStatusIncome",
    "financialStatusSavings_1",
    "n_switches",
]

# * Mediation analysis dummies
MEDIATION_CONTROL = "control"
MEDIATION_INTERVENTION_1 = "intervention_1"
MEDIATION_INTERVENTION_2 = "intervention_2"
