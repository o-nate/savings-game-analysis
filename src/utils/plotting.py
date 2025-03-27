"""Plotting functions"""

from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.axes import Axes

from src import calc_opp_costs, process_survey

from src.utils.constants import INTEREST_RATE
from utils.logging_config import get_logger

# * Set logger
logger = get_logger(__name__)

# * Pandas settings
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# * Decimal rounding
pd.set_option("display.float_format", lambda x: "%.2f" % x)


def visualize_persona_results(
    data: pd.DataFrame, persona_horizon: str, measures: list[str], **kwargs
) -> plt.Figure:
    grouped_df = (
        data.dropna()
        .groupby([persona_horizon, "treatment", "participant.round"])[measures]
        .describe()[[(m, me) for m in measures for me in ["count", "mean"]]]
    ).reset_index()

    # Flatten column names
    grouped_df.columns = [
        f"{col[0]}_{col[1]}" if col[1] != "" else col[0] for col in grouped_df.columns
    ]

    # Pivot to get participant.round as columns for easier difference calculation
    values = [f"{m}_mean" for m in measures] + [f"{measures[-1]}_count"]
    pivot_df = grouped_df.pivot_table(
        index=[persona_horizon, "treatment"],
        columns="participant.round",
        values=values,
    ).reset_index()

    # Calculate the difference between Round 2 and Round 1
    for measure in measures:
        pivot_df[f"{measure}_diff"] = (
            pivot_df[(f"{measure}_mean", 2)] - pivot_df[(f"{measure}_mean", 1)]
        )
        if measure == measures[-1]:
            pivot_df["count_diff"] = (
                pivot_df[(f"{measure}_count", 2)] - pivot_df[(f"{measure}_count", 1)]
            )

    # Plot configuration for three columns (Round 1, Round 2, Difference)
    fig, axs = plt.subplots(len(measures) + 1, 3, **kwargs)
    fig.suptitle(
        "Mean and Count Results by Participant Round and Difference", fontsize=16
    )

    round1_df = grouped_df[grouped_df["participant.round"] == 1]
    round2_df = grouped_df[grouped_df["participant.round"] == 2]

    for idx, measure in enumerate(measures):

        # Round 1 plot
        sns.barplot(
            data=round1_df,
            x=persona_horizon,
            y=f"{measure}_mean",
            hue="treatment",
            ax=axs[idx, 0],
            dodge=True,
            legend=False,
        )
        axs[idx, 0].set_title(f"{measure}: Round 1")
        axs[idx, 0].set_ylabel("Mean")

        # Round 2 plot
        sns.barplot(
            data=round2_df,
            x=persona_horizon,
            y=f"{measure}_mean",
            hue="treatment",
            ax=axs[idx, 1],
            dodge=True,
            legend=False,
        )
        axs[idx, 1].set_title(f"{measure}: Round 2")
        axs[idx, 1].set_ylabel("Mean")

        # Difference plot
        sns.barplot(
            data=pivot_df,
            x=persona_horizon,
            y=f"{measure}_diff",
            hue="treatment",
            ax=axs[idx, 2],
            dodge=True,
            legend=False,
        )
        axs[idx, 2].set_title(f"{measure}: Difference (Round 2 - Round 1)")
        axs[idx, 2].set_ylabel("Difference")

        if idx == len(measures) - 1:
            # Round 1 plot
            sns.barplot(
                data=round1_df,
                x=persona_horizon,
                y=f"{measure}_count",
                hue="treatment",
                ax=axs[idx + 1, 0],
                dodge=True,
                legend=True,
            )
            axs[idx + 1, 0].set_title("Count: Round 1")
            axs[idx + 1, 0].set_ylabel("Count")

            # Round 2 plot
            sns.barplot(
                data=round2_df,
                x=persona_horizon,
                y=f"{measure}_count",
                hue="treatment",
                ax=axs[idx + 1, 1],
                dodge=True,
                legend=False,
            )
            axs[idx + 1, 1].set_title("Count: Round 2")
            axs[idx + 1, 1].set_ylabel("Count")

            # Difference plot
            sns.barplot(
                data=pivot_df,
                x=persona_horizon,
                y="count_diff",
                hue="treatment",
                ax=axs[idx + 1, 2],
                dodge=True,
                legend=False,
            )
            axs[idx + 1, 2].set_title("Count: Difference (Round 2 - Round 1)")
            axs[idx + 1, 2].set_ylabel("Difference")

    # Extract legend handles and labels from one of the plots
    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles and labels:
        # Create a common legend
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=len(labels),
            title="Treatment",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_2d_histogram(
    df: pd.DataFrame,
    pattern_cols: list[Tuple[str, Any]],
    plot_title: str,
    **kwargs: dict[str, Any],
) -> Axes:
    """
    Create a 2D histogram with cell count labels using Seaborn heatmap.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be plotted.
    pattern_cols : list
        List of two column tuples to use for x and y axes.
        Each tuple should contain (column_name, round_number).
    plot_title : str
        Title of the plot to be displayed.
    **kwargs : dict
        Additional keyword arguments passed to sns.heatmap():

        Commonly used kwargs include:
        - annot (bool): If True, write the data value in each cell.
        - cbar (bool): If True, plot a colorbar.
        - cmap (str): Colormap name (e.g., "Blues", "YlGnBu").
        - ax (matplotlib.axes.Axes): Axes to plot on.
        - fmt (str): Format of the annotations (default 'd' for integers).
        - linewidths (float): Width of the lines that will divide each cell.
        - linecolor (str): Color of the lines dividing cells.
        - square (bool): If True, make the plot square.
        - vmin (float): Minimum value of the colormap.
        - vmax (float): Maximum value of the colormap.

    Returns:
    --------
    matplotlib.axes.Axes
        The axes object containing the 2D histogram heatmap.

    Example:
    --------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = plot_2d_histogram(
        pattern_changes,
        [("perception_pattern_12", 1.0), ("perception_pattern_12", 2.0)],
        "Perception Pattern Changes",
        annot=True,
        cbar=False,
        cmap="Blues",
        ax=axs[0]
    )
    """
    # Ensure fmt is set to "d" for integer formatting
    kwargs.setdefault("fmt", "d")

    # Create cross-tabulation to get the counts
    crosstab = pd.crosstab(df[pattern_cols[0]], df[pattern_cols[1]])

    # Create the heatmap
    ax = sns.heatmap(crosstab, fmt="d", **kwargs)  # Integer format

    # Set titles and labels
    ax.set_title(plot_title, fontsize=14)
    ax.set_xlabel(f"Round {pattern_cols[0][1]}", fontsize=12)
    ax.set_ylabel(f"Round {pattern_cols[1][1]}", fontsize=12)

    return ax


def annotate_2d_histogram(
    ax: Axes,
    x_col: Tuple[str, Any],
    y_col: Tuple[str, Any],
    data: pd.DataFrame,
    **kwargs: dict[str, Any],
) -> Axes:
    """
    Annotate each cell in a 2D histogram with the actual number of observations.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes containing the 2D histogram to be annotated
    x_col : tuple
        The column and value for the x-axis
    y_col : tuple
        The column and value for the y-axis
    data : pandas.DataFrame
        The original DataFrame used to create the histogram
    **kwargs : dict, optional
        Additional keyword arguments passed to ax.text() for customizing annotations.
        Commonly used kwargs include:
        - fontweight (str): Font weight of the annotation. Default is 'bold'.
            Example: fontweight='bold', fontweight='normal'
        - color (str): Color of the annotation text. Default is 'darkred'.
            Example: color='black', color='red', color='#FF0000'
        - fontsize (int): Size of the annotation text. Default is 8.
            Example: fontsize=8, fontsize=10, fontsize=12

    Returns:
    --------
    matplotlib.axes.Axes
        The annotated axes
    """
    # Extract the specific column values
    x_data = data.loc[:, x_col]
    y_data = data.loc[:, y_col]

    # Create cross-tabulation to get exact counts
    crosstab = pd.crosstab(x_data, y_data)

    # Iterate through the cross-tabulation
    for (x_val, y_val), count in crosstab.stack().items():
        if count > 0:
            ax.text(
                x_val,
                y_val,
                str(count),
                ha="center",
                va="center",
                **kwargs,
            )

    return ax


def main() -> None:

    df_opp_cost = calc_opp_costs.calculate_opportunity_costs()

    df_opp_cost = df_opp_cost.rename(columns={"month": "Month"})
    df_opp_cost.head()

    df_survey = process_survey.create_survey_df(include_inflation=True)
    df_inf_measures = process_survey.pivot_inflation_measures(df_survey)
    df_inf_measures = process_survey.include_inflation_measures(df_inf_measures)
    df_inf_measures["participant.inflation"] = np.where(
        df_inf_measures["participant.inflation"] == "4x30", 430, 1012
    )

    df_decisions = df_inf_measures.merge(df_opp_cost, how="left")

    # * Store final savings at month t = 120
    df_decisions["finalSavings_120"] = (
        df_decisions[df_decisions["Month"] == 120]
        .groupby("participant.code")["finalSavings"]
        .transform("mean")
    )
    df_decisions["finalSavings_120"] = df_decisions.groupby("participant.code")[
        "finalSavings_120"
    ].bfill()

    df_decisions.head()

    # * Classify subjects as Rational-Accurate, Rational-Pessimitic,
    # * Irrational-MoneyIllusion, Irrational-DeathAverse

    MAX_RATIONAL_STOCK = 15
    PERSONAS = [
        "Rational & Accurate",
        "Rational & Pessimistic",
        "Irrational & Money Illusioned",
        "Irrational & Death Averse",
    ]  # ["RA", "RP", "IM", "ID"]
    ANNUAL_INTEREST_RATE = ((1 + INTEREST_RATE) ** 12 - 1) * 100

    df_personas = df_decisions[df_decisions["Month"].isin([1, 12])]
    df_personas["previous_expectation"] = df_personas.groupby("participant.code")[
        "Quant Expectation"
    ].shift(1)

    _, axs = plt.subplots(3, 5, figsize=(30, 20))
    axs = axs.flatten()

    # _, axs2 = plt.subplots(3, 5, figsize=(30, 20))
    # axs2 = axs2.flatten()

    for max_stock in list(range(MAX_RATIONAL_STOCK)):
        data = df_personas.copy()

        CONDITIONS = [
            # Rational and accurate
            (data["finalStock"] <= max_stock)
            & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
            # Rational and pessimistic
            (data["finalStock"] > max_stock)
            & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
            # Irrational and money illusioned
            (data["finalStock"] <= max_stock)
            & (data["previous_expectation"] > ANNUAL_INTEREST_RATE),
            # Irrational and death averse
            (data["finalStock"] > max_stock)
            & (data["previous_expectation"] <= ANNUAL_INTEREST_RATE),
        ]

        data[f"persona_horizon_{max_stock}"] = np.select(
            condlist=CONDITIONS, choicelist=PERSONAS, default=np.nan
        )

        # * Add column for persona based on max_stock to track how distribution changes
        df_personas = df_personas.merge(data, how="left")

        data = data[data["Month"].isin([12])]

        print(data.value_counts(f"persona_horizon_{max_stock}"))

    MEASURES = ["previous_expectation", "finalStock", "finalSavings_120"]

    print(
        df_personas.dropna()
        .groupby(["persona_horizon_0", "treatment", "participant.round"])[MEASURES]
        .describe()[[(m, me) for m in MEASURES for me in ["count", "mean"]]]
    )

    _ = visualize_persona_results(df_personas, "persona_horizon_0", MEASURES)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    main()
