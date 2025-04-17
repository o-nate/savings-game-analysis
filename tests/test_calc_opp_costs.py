"""Tests for opportunity costs calculation module"""

import duckdb

from scripts.utils import constants
from src.calc_opp_costs import calculate_opportunity_costs
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

TABLE_NAME = "strategies"
TEST_PARTICIPANT_CODE = "17c9d4zc"

logger.debug("Check db")
con = duckdb.connect(constants.EXP_2_DATABASE_FILE, read_only=False)

if table_exists(con, TABLE_NAME):
    df_opp_cost = con.sql(f"SELECT * FROM {TABLE_NAME}").df()
else:
    logger.debug("Table does not exist")
    df_opp_cost = calculate_opportunity_costs(con, experiment=2)
    create_duckdb_database(con, {TABLE_NAME: df_opp_cost})
    logger.info("%s table added to database", TABLE_NAME)

logger.info("Initiating tests")

logger.info("Testing late cost makes sense at month 1")
assert (
    df_opp_cost[df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE]["late"].iat[0]
    == 0
)

logger.info("Testing costs and savings add to maximum")
total_costs = (
    df_opp_cost[
        (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
        & (df_opp_cost["month"] == 120)
    ][["early", "late", "excess"]]
    .sum(axis=1)
    .values[0]
)
savings = df_opp_cost[
    (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
    & (df_opp_cost["month"] == 120)
]["sreal"].iat[0]
max_savings = df_opp_cost[
    (df_opp_cost["participant.code"] == TEST_PARTICIPANT_CODE)
    & (df_opp_cost["month"] == 120)
]["soptimal"].iat[0]

logger.debug(
    "total costs (%s) + savings (%s) %s = max savings (%s)",
    total_costs,
    savings,
    total_costs + savings,
    max_savings,
)

assert total_costs + savings == max_savings


logger.info("Test complete")
