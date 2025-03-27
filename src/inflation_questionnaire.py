"""Process responses from inflation questionnaire"""

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from functools import reduce

from src import calc_opp_costs
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(DATABASE_FILE, read_only=False)


def calculate_inflation_score(data: pd.DataFrame) -> pd.Series:
    """Generate inflation behavior score based on answers to how individuals changed
    their behavior when faced with inflation in real life

    Args:
        data (pd.DataFrame): DataFrame with real-life behavioral change responses

    Returns:
        pd.Series: Series with score for each individual
    """
    return (
        data["Inflation.1.player.inf_checking"] * -1
        + data["Inflation.1.player.inf_stock"]
        + data["Inflation.1.player.inf_quantity"]
    )


# %%
if table_exists(con, "Inflation") == False:
    create_duckdb_database(con, initial_creation=True)
df = con.sql("SELECT * FROM Inflation").df()
df.describe().T

# %%
# sns.pairplot(df[[c for c in df.columns if any(bc in c for bc inf_behavior_cols)]])


# %%
df["inflation_score"] = calculate_inflation_score(df)

sns.histplot(data=df, x="inflation_score")

inf_behavior_cols = [
    "Inflation.1.player.inf_food",
    "Inflation.1.player.inf_housing",
    "Inflation.1.player.inf_other",
    "Inflation.1.player.inf_quantity",
    "Inflation.1.player.inf_stock",
    "Inflation.1.player.inf_checking",
    "Inflation.1.player.inf_savings",
    "inflation_score",
]
# %%
df2 = calc_opp_costs.calculate_opportunity_costs()
df2 = df2.merge(df[["participant.label"] + inf_behavior_cols], how="left")

for c in inf_behavior_cols:
    sns.jointplot(
        data=df2[(df2["month"] == 120) & (df2["participant.round"] == 1)],
        x=c,
        y="sreal_%",
        # hue="participant.round",
        kind="reg",
    )
    plt.show()


# %%
def main() -> None:
    """Run script"""


if __name__ == "__main__":
    main()
