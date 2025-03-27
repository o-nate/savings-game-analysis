"""Present results from both experiments"""

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

# DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(constants.DATABASE_FILE, read_only=False)

# ! Export plots
export_all_plots = input("Export all plots? (y/n) ").lower() == "y"
FILE_PATH = Path(__file__).parents[1] / "results"

# %% [markdown]
## Descriptive statistics: Subjects
if table_exists(con, "Questionnaire") == False:
    create_duckdb_database(con, initial_creation=True)
df_questionnaire = con.sql("SELECT * FROM Questionnaire").df()

df_questionnaire[
    [
        m
        for m in df_questionnaire
        if any(qm in m for qm in constants.QUESTIONNAIRE_MEASURES)
    ]
].describe().T[["mean", "std", "min", "50%", "max"]]
