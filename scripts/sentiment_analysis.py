"""Sentiment analysis of post-experiment survey responses"""

# %%
from typing import Type

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pingouin import mediation_analysis
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
from tqdm.auto import tqdm
from transformers import pipeline, Pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

from scripts.utils import constants

from src import (
    calc_opp_costs,
    discontinuity,
    intervention,
    econ_preferences,
    knowledge,
    process_survey,
)
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

logger = get_logger(__name__)
logger.debug(constants.DATABASE_FILE)
con = duckdb.connect(constants.DATABASE_FILE, read_only=False)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)

MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

if table_exists(con, "sessionResults") == False:
    create_duckdb_database(con, initial_creation=True)
df_responses = con.sql("SELECT * FROM sessionResults").df()

for q in ["q2", "q3"]:
    print(len(df_responses[df_responses[f"sessionResults.1.player.{q}"].isna()]))

# %%
# Clean data
for q in ["q2", "q3"]:
    df_responses[f"sessionResults.1.player.{q}"].fillna("", inplace=True)
    print(len(df_responses[df_responses[f"sessionResults.1.player.{q}"].isna()]))

# %%
# Use a pipeline as a high-level helper
sentiment_task = pipeline("sentiment-analysis", model=MODEL, tokenizer=MODEL)

# %%
example = df_responses["sessionResults.1.player.q3"].iat[0]
print(example)

# %%
sentiment_task(example)


# # %%
# def analyze_sentiment(
#     text: str, sentiment_analyzer: Type[Pipeline]
# ) -> list[dict[str, str | float]]:
#     return sentiment_analyzer(text)


# %%
# Analyze sentiments of both open-ended survey questions
tqdm.pandas()
for q in ["q2", "q3"]:
    df_responses[f"sentiment_{q}"] = df_responses[
        f"sessionResults.1.player.{q}"
    ].progress_apply(lambda text: sentiment_task(text))
    s = df_responses.pop(f"sentiment_{q}").explode()
    df_responses = df_responses.join(
        pd.DataFrame(s.tolist(), index=s.index).rename(
            columns={"label": f"label_{q}", "score": f"score_{q}"}
        )
    )
    df_responses.value_counts(f"label_{q}")

# %%
df_responses[
    [c for c in df_responses.columns if any(q in c for q in ["q2", "q3"])]
].head()
