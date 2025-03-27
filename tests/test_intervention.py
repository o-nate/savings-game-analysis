"""Test of intervention module"""

import logging
import math

from pathlib import Path

import duckdb

from src.calc_opp_costs import calculate_opportunity_costs
from src.discontinuity import purchase_discontinuity
from src.intervention import calculate_change_in_measure
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

from tests.utils.constants import CHANGE_IN_PERFORMANCE

# * Logging settings
logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(DATABASE_FILE, read_only=False)
if table_exists(con, "Inflation") == False:
    create_duckdb_database(con, initial_creation=True)


# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

logger.info("Testing intervention module")

df_int = con.sql("SELECT * FROM task_int").df()

df_results = calculate_opportunity_costs()

df_results = purchase_discontinuity(
    df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
)

questions = ["intro_1", "q", "confirm"]
cols = [c for c in df_int.columns if any(q in c for q in questions)]

# * Compare impact of intervention
measures = [
    "finalSavings",
    "early",
    "late",
    "excess",
]

data_df = df_results[df_results["month"] == 120].copy()
logging.debug(data_df.shape)
data_df = data_df.merge(df_int[["participant.label", "date"] + cols], how="left")

# * Rename measures
data_df.rename(
    columns={
        "finalSavings": "Total savings",
        "early": "Over-stocking",
        "late": "Under-stocking",
        "excess": "Wasteful-stocking",
    },
    inplace=True,
)

measures = ["Total savings", "Over-stocking", "Under-stocking", "Wasteful-stocking"]

# * Measure intervention impact
for treat in ["Intervention 1", "Intervention 2", "Control"]:
    for m in measures:
        before, after, _ = calculate_change_in_measure(
            data_df[data_df["treatment"] == treat], m
        )
        assert math.isclose(
            before, CHANGE_IN_PERFORMANCE[treat][f"Initial {m}"], rel_tol=0.05
        )
        assert math.isclose(
            after, CHANGE_IN_PERFORMANCE[treat][f"Final {m}"], rel_tol=0.05
        )

logger.info("Test complete")
