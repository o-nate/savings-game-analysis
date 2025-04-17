"""
Present results from experiment 1:
Inflation and behavior, an experimental analysis
"""

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import calc_opp_costs, discontinuity, intervention, econ_preferences, knowledge

from src.process_survey import (
    create_survey_df,
    include_inflation_measures,
    include_uncertainty_measure,
    pivot_inflation_measures,
)
from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.database import create_duckdb_database, table_exists

from src.utils.helpers import combine_series, export_plot
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

## Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

con = duckdb.connect(constants.EXP_1_DATABASE_FILE, read_only=False)

# ! Export plots
export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
FILE_PATH = Path(__file__).parents[1] / "results"

# %% [markdown]
## Descriptive statistics: Subjects
if table_exists(con, "Questionnaire") == False:
    create_duckdb_database(con, experiment=1, initial_creation=True)
df_questionnaire = con.sql("SELECT * FROM Questionnaire").df()

df_questionnaire[
    [
        m
        for m in df_questionnaire
        if any(qm in m for qm in constants.QUESTIONNAIRE_MEASURES)
    ]
].describe().T[["mean", "std", "min", "50%", "max"]]


# %% [markdown]
## Behavior in the Savings Game
### Overall performance
df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con, experiment=1)
calc_opp_costs.plot_savings_and_stock(
    df_opp_cost, col="phase", row="participant.inflation", palette="tab10"
)


# %% [markdown]
### Performance measures: Over- and wasteful-stocking and purchase adaptation
# TODO update discontinuity function to include 10x12
#  df_measures = discontinuity.purchase_discontinuity(
#     df_opp_cost, constants.DECISION_QUANTITY, constants.WINDOW
# )

# ## Set avg_q and avg_q_% as month=33 value
# df_pivot_measures = pd.pivot_table(
#     df_measures[df_measures["month"] == 33][["participant.code", "avg_q", "avg_q_%"]],
#     index="participant.code",
# )
# df_pivot_measures.reset_index(inplace=True)
# df_measures = df_measures[[m for m in df_measures.columns if "avg_q" not in m]].merge(
#     df_pivot_measures, how="left"
# )

# ## Rename columns for results table
# df_measures.rename(
#     columns={
#         k: v
#         for k, v in zip(
#             constants.PERFORMANCE_MEASURES_OLD_NAMES
#             + constants.PURCHASE_ADAPTATION_OLD_NAME,
#             constants.PERFORMANCE_MEASURES_NEW_NAMES
#             + constants.PURCHASE_ADAPTATION_NEW_NAME,
#         )
#     },
#     inplace=True,
# )
# df_measures[(df_measures["month"] == 120) & (df_measures["phase"] == "pre")].describe()[
#     constants.PERFORMANCE_MEASURES_NEW_NAMES + constants.PURCHASE_ADAPTATION_NEW_NAME
# ].T[["mean", "std", "min", "50%", "max"]]
