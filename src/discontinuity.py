"""
Identify change in purchase behavior before and after inflation phase changes

***Measure of discontinuity***
measure = mean([mean(after_430) - mean(before_430)], [mean(after_1012) - mean(before_1012)]),
where before_430 is mean(quantity months 28-30) / `decision quantity` at start of month 28
and after_430 is mean(quantity months 31-33) / `decision quantity` at start of month 2
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.calc_opp_costs import calculate_opportunity_costs
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

# * Define phase change months to analyze
CHANGE_430 = 30
CHANGE_1012 = 12
CHANGE_1012_2 = 36

# * Define discontinuity measure for graph
Y = "avg_q"


def purchase_discontinuity(
    df: pd.DataFrame,
    decision_quantity: str,
    window: int,
) -> pd.DataFrame:
    """Generates dataframe with column for change in purchase behavior before
    and after inflation phase changes"""
    ## Set `decision_quantity` to previous month's
    ## The `decision_quantity` at month=1 for each participant should be NaN
    ## since the decision at month=1 is based on `decision_quantity` at month=t-1
    final_df = df.copy()
    final_df[decision_quantity] = final_df.groupby("participant.code")[
        decision_quantity
    ].shift(1)

    ## Remove all `decision quantitys` except at first phase change (and third from 10x12)
    final_df[decision_quantity] = np.where(
        final_df["month"] == CHANGE_430 - (window - 1),
        final_df[decision_quantity],
        np.nan,
    )

    ## Forward fill `decision quantity` with value at start of first month in window
    final_df[decision_quantity] = final_df.groupby("participant.code")[
        decision_quantity
    ].ffill()

    ## Take rolling average of quantity purchased with window and then
    final_df["avg_q"] = final_df.groupby("participant.code")["decision"].transform(
        lambda x: x.rolling(window).mean()
    )

    ## Calculate percentage of initial decision_quantity at start of window
    final_df["avg_q_%"] = final_df["avg_q"] / final_df[decision_quantity] * 100

    return final_df


def main() -> None:
    """Run script"""
    df_opp_cost = calculate_opportunity_costs(con)
    df_disc = df_opp_cost[df_opp_cost.columns.to_list()[:13]].copy()
    print(df_disc.head())

    df_disc = purchase_discontinuity(
        df_disc, decision_quantity=DECISION_QUANTITY, window=WINDOW
    )

    print("-------------------------------------------------------------")
    print(f"{df_disc['participant.label'].nunique()} participants included")
    print(df_disc.head())
    print("-------------------------------------------------------------")
    print(
        df_disc[df_disc["month"] == 120]
        .groupby("participant.round")[["avg_q", "avg_q_%"]]
        .describe()
        .T
    )
    logger.debug(df_disc.shape)
    ## Assign starting month
    scatter_dict = {
        430: {
            "month": CHANGE_430,
            "title": "4x30 (Pre-treatment)",
            "phase": "pre",
        },
        "430_2": {
            "month": CHANGE_430,
            "title": "4x30 (Post-treatment)",
            "phase": "post",
        },
    }

    sequence_layout = [
        430,
        "430_2",
    ]  ## Define which side each sequence should be on

    fig, axes = plt.subplots(ncols=len(sequence_layout), sharey=True, figsize=(10, 5))

    ## Define a custom palette with only black
    custom_palette = sns.color_palette(["black"])

    ## Loop through scatter_dict to generate dataframes for before & after figures for each sequence
    for seq, v in scatter_dict.items():
        before_after = [v["month"], (v["month"] + WINDOW)]
        if seq == 430 or seq == 1012:
            v["data"] = df_disc[
                (df_disc["participant.inflation"] == seq)
                & df_disc["month"].isin(before_after)
                & (df_disc["phase"] == v["phase"])
            ]
        elif seq == "430_2":
            v["data"] = df_disc[
                (df_disc["participant.inflation"] == 430)
                & df_disc["month"].isin(before_after)
                & (df_disc["phase"] == v["phase"])
            ]
        else:
            v["data"] = df_disc[
                (df_disc["participant.inflation"] == 1012)
                & df_disc["month"].isin(before_after)
                & (df_disc["phase"] == v["phase"])
            ]

    ## Loop through sequences and generate connected scatter plots
    for i, seq in enumerate(sequence_layout):
        data = scatter_dict[seq]["data"]
        start = scatter_dict[seq]["month"]
        end = start + WINDOW
        sns.violinplot(
            ax=axes[i],
            data=data,
            x="month",
            y=Y,
            split=True,
            palette="viridis",
            alpha=0.25,
            inner=None,
            dodge=True,
        )
        sns.stripplot(
            ax=axes[i],
            data=data,
            x="month",
            y=Y,
            jitter=0.015,
            palette="viridis",
            alpha=0.4,
            legend=None,
        )
        sns.pointplot(
            ax=axes[i],
            data=data,
            x="month",
            y=Y,
            estimator=np.mean,
            errorbar=None,
            dodge=0.0,
            palette=custom_palette,
        )
        ## Plot lines with connected points
        x1 = np.array(
            data["month"][data["month"] == start] - start
        )  ## Must shift leftward manually
        x2 = np.array(
            data["month"][data["month"] == end] - (end - 1)
        )  ## Must shift leftward manually
        y1 = np.array(data[Y][data["month"] == start])
        y2 = np.array(data[Y][data["month"] == end])
        for n, m in enumerate(x1):
            axes[i].plot(
                [m, x2[n]],
                [y1[n], y2[n]],
                ## Define color
                color="grey",
                alpha=0.15,
            )
        ## Line for means
        before = data[(data["month"] == start)][Y].mean()
        after = data[(data["month"] == end)][Y].mean()
        logger.debug("before: %s, after %s", before, after)
        axes[i].plot([0, 1], [before, after], color="black")

        ## Add title and labels
        title = scatter_dict[seq]["title"]
        axes[i].set_title(title)

        ## Change x-axis labels
        x_labels = ["Before" if m < end else "After" for m in data["month"].values]
        axes[i].set_xticklabels(x_labels)

        axes[i].set_xlabel("")  ## Remove x-axis subtitles
        axes[i].set_ylabel("")  ## Remove y-axis subtitles

    ## Common title
    fig.suptitle(
        f"Change in average quantity purchased in {WINDOW}-month window before and after inflation phase change"
    )
    ## Common x-axis title
    fig.text(0.5, 0.04, "Inflation phase change", ha="center")
    ## Common y-axis title
    fig.text(0.04, 0.5, "Average quantity purchased", va="center", rotation="vertical")
    plt.show()


if __name__ == "__main__":
    main()
