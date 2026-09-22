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
INFLATION_SEQUENCES = {430: INF_430, 1012: INF_1012}
INFLATION_SEQUENCE_LABELS = {430: "4x30", 1012: "10x12"}


def _inflation_schedule(sequence: int, rates: list[float]) -> dict[str, list]:
    """Realized inflation for one sequence, repeated for rounds 1 and 2.

    Actual is the rate over the 12 months ending at each survey month.
    Upcoming at month t is the rate over the following 12 months, so the
    upcoming value at month 1 is the first actual rate.
    """
    actual_months = [month * 12 for month in range(1, 11)]
    upcoming_months = [1] + actual_months[:-1]
    schedule = {
        "participant.inflation": [],
        "participant.round": [],
        "Month": [],
        "Measure": [],
        "Estimate": [],
    }
    for round_number in (1, 2):
        measures = (("Actual", actual_months), ("Upcoming", upcoming_months))
        for measure, months in measures:
            for month, rate in zip(months, rates):
                schedule["participant.inflation"].append(sequence)
                schedule["participant.round"].append(round_number)
                schedule["Month"].append(month)
                schedule["Measure"].append(measure)
                schedule["Estimate"].append(rate)
    return schedule


def _combine_schedules(*schedules: dict[str, list]) -> dict[str, list]:
    combined = {key: [] for key in schedules[0]}
    for schedule in schedules:
        for key, values in schedule.items():
            combined[key].extend(values)
    return combined


INFLATION_DICT = _combine_schedules(
    *(
        _inflation_schedule(sequence, rates)
        for sequence, rates in INFLATION_SEQUENCES.items()
    )
)

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
