"""Analyze wasteful-stocking and errors in instructions questions"""

# %%
import logging
from pathlib import Path
import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

from scripts.utils import constants

from src import (
    calc_opp_costs,
    discontinuity,
    intervention,
    econ_preferences,
    knowledge,
    process_survey,
)

from src.stats_analysis import (
    create_bonferroni_correlation_table,
    create_pearson_correlation_matrix,
    run_forward_selection,
    run_treatment_forward_selection,
)
from src.utils.constants import EXP_2_DATABASE
from src.utils.database import create_duckdb_database, table_exists
from src.utils.helpers import combine_series, export_plot
from utils.logging_config import get_logger

logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

## Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

DATABASE_FILE = Path(__file__).parents[1] / "data" / EXP_2_DATABASE
con = duckdb.connect(DATABASE_FILE, read_only=False)

# %%
df_opp_cost = calc_opp_costs.calculate_opportunity_costs(con)

# %%
if table_exists(con, "task_instructions") == False:
    create_duckdb_database(con, initial_creation=True)
df = con.sql("SELECT * FROM task_instructions").df()
df2 = df_opp_cost[
    (df_opp_cost["month"] == 120) & (df_opp_cost["participant.round"] == 1)
]
df3 = df2[["participant.label", "excess_%", "early_%", "finalSavings_%"]].merge(
    df[["participant.label"] + [c for c in df.columns if "error" in c]], how="left"
)

# %% [markdown]
## Plot correlation matrix
# Compute the correlation matrix
corr = df3[[c for c in df3.columns if "label" not in c]].corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 220, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,
    mask=mask,
    cmap=cmap,
    vmax=0.3,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
