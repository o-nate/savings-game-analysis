"""Create DuckDB database to persist the results instead of regenerating them each time"""

from pathlib import Path

import duckdb
import pandas as pd

from src.preprocess import preprocess_data
from utils.logging_config import get_logger

logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[2] / "data" / "database.duckdb"


def create_duckdb_database(
    db_connection: duckdb.DuckDBPyConnection,
    initial_creation: bool = False,
    data_dict: dict[str, pd.DataFrame] = None,
) -> None:
    # con = duckdb.connect(database=DATABASE_FILE, read_only=False)

    if initial_creation:
        logger.debug("initial creation??????????????????????")
        data_dict = preprocess_data()

    for name, df in data_dict.items():
        logger.debug("creating table %s", name)
        data_for_table = df.copy()
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM data_for_table;
        """
        db_connection.sql(query=create_table_query)
    logger.info("Database ready!")


def table_exists(db_connection: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    """
    Check if a table exists in a DuckDB database.

    Args:
        db_connection: An open DuckDB connection
        table_name: Name of the table to check

    Returns:
        bool: True if the table exists, False otherwise
    """
    try:
        # Query the information schema
        result = db_connection.execute(
            """
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_name = ?
            )
        """,
            [table_name],
        ).fetchone()[0]
        if result is False:
            logger.info("Table %s does not exist.", table_name)
        return bool(result)

    except Exception as e:
        print(f"Error checking table existence: {e}")
        return False


def main() -> None:
    con = duckdb.connect(DATABASE_FILE, read_only=False)
    if table_exists(con, "decision"):
        logger.info("Database exists!")
    else:
        final_df_dict = preprocess_data()
        create_duckdb_database(db_connection=con, data_dict=final_df_dict)
    if table_exists(con, "strategies"):
        logger.info("Strategies table exists!")
    else:
        logger.info("Strategies table does not exist.")


if __name__ == "__main__":
    main()
