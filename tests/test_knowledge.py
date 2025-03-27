"""Test knowledge measures module"""

from pathlib import Path

import duckdb

from src.knowledge import count_correct_responses, create_knowledge_dataframe
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

from tests.utils import constants

logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(DATABASE_FILE, read_only=False)
if table_exists(con, "Inflation") == False:
    create_duckdb_database(con, initial_creation=True)

logger.info("Testing knowledge measures")

df = con.sql("SELECT * FROM Finance").df()
measure = "financial_literacy"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.FIN_LIT_PARTICIPANT_CODE][
    measure
].values[0]
assert result == constants.SCORE


df = con.sql("SELECT * FROM Numeracy").df()
measure = "numeracy"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_2B][
    measure
].values[0]
assert result == constants.SCORE

result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_3][
    measure
].values[0]
assert result == constants.SCORE

result = df[df["participant.code"] == constants.NUMERACY_PARTICIPANT_CODE_3_NOT][
    measure
].values[0]
assert result != constants.SCORE


df = con.sql("SELECT * FROM Inflation").df()
measure = "compound"
df[measure] = count_correct_responses(df, measure)
result = df[df["participant.code"] == constants.COMPOUND_PARTICIPANT_CODE][
    measure
].values[0]
assert result == constants.SCORE

logger.info("Testing knowledge dataframe creation")
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["financial_literacy"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_FIN_LIT_SCORE
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["numeracy"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_NUM_SCORE
df = create_knowledge_dataframe()
result_fin = df[
    df["participant.label"] == constants.KNOWLEDGE_DATAFRAME_PARTICIPANT_LABEL
]["compound"].values[0]
assert result_fin == constants.KNOWLEDGE_DATAFRAME_COMPOUND_SCORE

logger.info("Testing complete")
