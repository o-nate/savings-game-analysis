"""Constants for src modules"""

# * Exp data info
EXP_1_FILE = "all_apps_wide_2023-02-27.csv"
EXP_2_FILE = "all_apps_wide_2024-07-08.csv"

# * DuckDB info
EXP_1_DATABASE = "exp1.duckdb"
EXP_2_DATABASE = "exp2.duckdb"

# * Savings Game initial parameters
INITIAL_ENDOWMENT = 863.81
INTEREST_RATE = 0.2277300 / 12
ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100
WAGE = 4.32

# * For column name conversion
EXP_1_COLUMNS_HASH = {
    "participant.intervention": "treatment",
    "phase": "participant.round",
}

# * Define annualized inflation, per 12 months
INF_1012 = [0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79, 0.45, 60.79]
INF_430 = [0.38, 0.47, 26.85, 55.49, 64.18, 0.38, 0.47, 26.85, 55.49, 64.18]

INFLATION_DICT = {
    "participant.inflation": [430 for m in range(40)],  # + [1012 for m in range(19)],
    "participant.round": [1 for m in range(10)]
    + [1 for m in range(10)]
    + [2 for m in range(10)]
    + [2 for m in range(10)],
    "Month": [m * 12 for m in range(1, 11)]
    + [m * 12 if m > 0 else m + 1 for m in range(10)]
    + [m * 12 for m in range(1, 11)]
    + [m * 12 if m > 0 else m + 1 for m in range(10)],
    "Measure": ["Actual" for m in range(10)]
    + ["Upcoming" for m in range(10)]
    + ["Actual" for m in range(10)]
    + ["Upcoming" for m in range(10)],
    "Estimate": INF_430 + INF_430 + INF_430 + INF_430,  # + INF_1012 + INF_1012[1:],
}

# * Economic preferences
CHOICES = {
    "riskPreferences": "probability",
    "lossAversion": "loss.",
    "timePreferences": ".q",
}
TIME_PREFERENCES_ROUNDS = 2

# * Knowledge
QUESTIONS = {
    "financial_literacy": {
        "Finance.1.player.finK_1": 1,
        "Finance.1.player.finK_2": -1,
        "Finance.1.player.finK_9": 1,
    },
    "numeracy": {"Numeracy.1.player.num_2b": 20, "Numeracy.1.player.num_3": 30},
    "compound": {
        "Inflation.1.player.infCI_1": 1100,
        "Inflation.1.player.infCI_2": 2,
        "Inflation.1.player.infCI_3": 2,
        "Inflation.1.player.infCI_4": 32000,
    },
}

# * Stats analysis
BONFERRONI_ALPHA = 0.05

# * Decision patterns parameters
PERSONAS = ["AC", "AI", "IC", "II"]
DECISION_PATTERN_MEASURES = [
    "Quant Perception",
    "Quant Expectation",
    "Qual Expectation",
]
QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_12 = 2
QUALITATIVE_EXPECTATION_THRESHOLD_MONTH_36 = 1
