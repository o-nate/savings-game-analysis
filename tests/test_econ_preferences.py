"""Test economic preferences module"""

from pathlib import Path

import duckdb

from src.econ_preferences import (
    count_preference_choices,
    count_switches,
    count_wisconsin_errors,
    create_econ_preferences_dataframe,
)
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger
from tests.utils import constants

logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / "database.duckdb"
con = duckdb.connect(DATABASE_FILE, read_only=False)
if table_exists(con, "Inflation") == False:
    create_duckdb_database(con, initial_creation=True)

logger.info("Testing economic preferences module")

logger.info("Testing: Risk preferences")
df = con.sql("SELECT * FROM riskPreferences").df()
df["n_safe"] = count_preference_choices(df, "riskPreferences")
df["n_switches"] = count_switches(df, "riskPreferences")
result = df[df["participant.code"] == constants.RISK_SAFE_PARTICIPANT_CODE][
    "n_safe"
].values[0]
assert result == constants.RISK_SAFE_NUMBER
result = df[df["participant.code"] == constants.RISK_SAFE_PARTICIPANT_CODE][
    "n_switches"
].values[0]
assert result == constants.RISK_SWITCHES_NUMBER


logger.info("Testing switch count: Loss aversion")
df = con.sql("SELECT * FROM lossAversion").df()
df["n_switches"] = count_switches(df, "lossAversion")
result = df[df["participant.code"] == constants.LOSS_SWITCHES_PARTICIPANT_CODE][
    "n_switches"
].values[0]
assert result == constants.LOSS_SWITCHES_NUMBER


logger.info("Testing: Time preferences")
df = con.sql("SELECT * FROM timePreferences").df()
df["n_present"] = count_preference_choices(df, "timePreferences")
result = df[df["participant.code"] == constants.TIME_PRESENT_PARTICIPANT_CODE][
    "n_present"
].values[0]
logger.debug("tpref result %s", result)
assert result == constants.TIME_PRESENT_NUMBER
df["n_switches"] = count_switches(df, "timePreferences")
result = df[df["participant.code"] == constants.TIME_PRESENT_PARTICIPANT_CODE][
    "n_switches"
].values[0]
logger.debug("tpref result switches %s", result)
assert result == constants.TIME_PRESENT_SWITCHES


logger.info("Testing Wisconsin Card Sorting Task")
df = con.sql("SELECT * FROM wisconsin").df()
df["n_corr"] = count_preference_choices(df, "wisconsin")
logger.debug(df[["participant.code", "n_corr"]].head())
result = df[df["participant.code"] == constants.WISC_PARTICIPANT_CODE]["n_corr"].values[
    0
]
logger.debug("wisc result %s", result)
assert result == constants.WISC_N_CORR

df["n_PE"] = count_wisconsin_errors(df, "perseverative")
df["n_SE"] = count_wisconsin_errors(df, "set-loss")
result = df[df["participant.code"] == constants.WISC_PARTICIPANT_CODE]["n_PE"].values[0]
logger.debug("wisc result pe %s", result)
assert result == constants.WISC_N_PE

result = df[df["participant.code"] == constants.WISC_PARTICIPANT_CODE]["n_SE"].values[0]
logger.debug("wisc result se %s", result)
assert result == constants.WISC_N_SE

logger.info("Testing creat dataframe")
df_econ_preferences = create_econ_preferences_dataframe()
result = df_econ_preferences.shape
logger.debug("Shape %s vs %s", result, constants.DATAFRAME_SHAPE)
assert result == constants.DATAFRAME_SHAPE


logger.info("Tests complete")
