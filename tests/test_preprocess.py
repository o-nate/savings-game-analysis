"""Tests for preprocess module"""

from pathlib import Path

import duckdb

from src.preprocess import preprocess_data
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists

from tests.utils import constants
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(DATABASE_FILE, read_only=False)
if not table_exists(con, "Inflation"):
    create_duckdb_database(con, initial_creation=True)

# * Constants for testing

logger.info("Testing preprocess module")

logger.info(
    "Testing conversion of nested dicts/lists to columns for Wisconsin Card Sorting Task"
)
df = con.sql("SELECT * FROM wisconsin").df()
final_df_dict = preprocess_data()
assert (
    final_df_dict["wisconsin"]["trial_number_30"].mean() == constants.MEAN_TRIAL_NUMBER
)
assert df["trial_number_30"].mean() == constants.MEAN_TRIAL_NUMBER
logger.info("Trial numbers correct")


assert (
    final_df_dict["wisconsin"].value_counts("correct_5")[True]
    == constants.NCORR_TRIAL_NUMBER_5
)
assert df.value_counts("correct_5")[True] == constants.NCORR_TRIAL_NUMBER_5
logger.info("Number of correct guesses correct")
