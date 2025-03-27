"""Conduct multivariate analyses on treatment effects of stock-expectation relationship"""

from typing import Any

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn as sns
import matplotlib.pyplot as plt

from src import calc_opp_costs, process_survey

from utils.logging_config import get_logger


# * Logging settings
logger = get_logger(__name__)


def analyze_treatment_effects(data: pd.DataFrame, month: int) -> dict[str, Any]:
    """
    Analyze the effect of treatment on Quant Expectation and finalStock at month,
    comparing participant rounds 1 and 2. Provide data visualizations as well.

    Args:
        data (pd.DataFrame): Must contain columns:
        'participant.code',
        'participant.label',
        'participant.round',
        'Month',
        'Quant Expectation',
        'finalStock'
        month (str): Month of measurement

    Returns:
        dict[str, Any]: Dictionary with results
    """

    # Filter for Month
    df_analysis = data[data["Month"].isin([max(1, month - 12), month])].copy()

    # Set expectation as previous estimate
    df_analysis["previous_expectation"] = df_analysis.groupby("participant.code")[
        "Quant Expectation"
    ].shift(1)
    df_analysis["Quant Expectation"] = df_analysis["previous_expectation"]

    # Create summary statistics function
    def get_summary_stats(group):
        return pd.Series(
            {
                "Quant_Exp_Mean": group["Quant Expectation"].mean(),
                "Quant_Exp_SD": group["Quant Expectation"].std(),
                "finalStock_Mean": group["finalStock"].mean(),
                "finalStock_SD": group["finalStock"].std(),
                "Count": len(group),
            }
        )

    # Get summary statistics by treatment and round
    summary_stats = df_analysis.groupby(["treatment", "participant.round"]).apply(
        get_summary_stats
    )

    # Prepare data for analysis
    df_clean = df_analysis.dropna(subset=["Quant Expectation", "finalStock"])
    df_clean = df_clean.rename(
        columns={"Quant Expectation": "Quant_Expectation", "participant.round": "round"}
    )

    # MANOVA with round interaction
    manova = MANOVA.from_formula(
        "Quant_Expectation + finalStock ~ C(treatment) * C(round)",
        data=df_clean,
    )
    manova_results = manova.mv_test()

    # Separate ANOVAs with effect sizes and interactions
    def run_anova_with_effects(dv):
        # Run ANOVA with interaction
        model = sm.OLS.from_formula(f"{dv} ~ treatment * C(round)", data=df_clean)
        results = model.fit()

        # Calculate effect sizes
        aov_table = sm.stats.anova_lm(results, typ=2)
        ss_total = aov_table["sum_sq"].sum()

        effect_sizes = {
            "treatment": aov_table["sum_sq"][0] / ss_total,
            "round": aov_table["sum_sq"][1] / ss_total,
            "interaction": (
                aov_table["sum_sq"][2] / ss_total if len(aov_table) > 2 else None
            ),
        }

        # Run Tukey's HSD for treatment within each round
        tukey_results = {}
        for round_num in [1, 2]:
            round_data = df_clean[df_clean["round"] == round_num]
            tukey = pairwise_tukeyhsd(round_data[dv], round_data["treatment"])
            tukey_results[f"Round_{round_num}"] = tukey

        return {
            "anova_results": results,
            "effect_sizes": effect_sizes,
            "tukey_results": tukey_results,
        }

    quant_results = run_anova_with_effects("Quant_Expectation")
    stock_results = run_anova_with_effects("finalStock")

    # Calculate correlations within each treatment-round combination
    correlations = df_clean.groupby(["treatment", "round"]).apply(
        lambda x: x["Quant_Expectation"].corr(x["finalStock"])
    )

    def create_visualizations(data):
        data = data.rename(
            columns={
                "Quant_Expectation": "Quant Expectation",
                "round": "participant.round",
            }
        )

        fig = plt.figure(figsize=(20, 15))

        # Use tab10 palette name for seaborn plots
        palette = "tab10"
        # For manual coloring in scatter plots, get the actual colors
        tab10_colors = plt.cm.tab10.colors

        # 1. Box plots by treatment and round
        plt.subplot(3, 2, 1)
        sns.boxplot(
            data=data,
            x="treatment",
            y="Quant Expectation",
            hue="participant.round",
            palette=palette,
        )
        plt.title("Quantitative Expectations by Treatment and Round")
        plt.xticks(rotation=45)

        plt.subplot(3, 2, 2)
        sns.boxplot(
            data=data,
            x="treatment",
            y="finalStock",
            hue="participant.round",
            palette=palette,
        )
        plt.title("Final Stock by Treatment and Round")
        plt.xticks(rotation=45)

        # 2. Interaction plots
        plt.subplot(3, 2, 3)
        sns.pointplot(
            data=data,
            x="treatment",
            y="Quant Expectation",
            hue="participant.round",
            errorbar="se",
            palette=palette,
        )
        plt.title("Treatment-Round Interaction (Quant Expectation)")
        plt.xticks(rotation=45)

        plt.subplot(3, 2, 4)
        sns.pointplot(
            data=data,
            x="treatment",
            y="finalStock",
            hue="participant.round",
            errorbar="se",
            palette=palette,
        )
        plt.title("Treatment-Round Interaction (Final Stock)")
        plt.xticks(rotation=45)

        # 3. Scatter plots by round with matching regression lines
        def plot_scatter_with_matched_reglines(data, round_num, ax):
            # Get the number of treatments
            n_treatments = len(data["treatment"].unique())

            # Create plot for each treatment with tab10 colors
            for i, treatment in enumerate(sorted(data["treatment"].unique())):
                treatment_data = data[
                    (data["participant.round"] == round_num)
                    & (data["treatment"] == treatment)
                ]

                # Plot scatter points
                sns.scatterplot(
                    data=treatment_data,
                    x="Quant Expectation",
                    y="finalStock",
                    color=tab10_colors[i],
                    label=treatment,
                    alpha=0.5,
                    ax=ax,
                )

                # Add regression line with same color
                sns.regplot(
                    data=treatment_data,
                    x="Quant Expectation",
                    y="finalStock",
                    scatter=False,
                    color=tab10_colors[i],
                    line_kws={"linestyle": "-"},
                    ax=ax,
                )

            ax.set_title(f"Relationship between Variables (Round {round_num})")

        # Create scatter plots with matched regression lines
        ax5 = plt.subplot(3, 2, 5)
        plot_scatter_with_matched_reglines(data, 1, ax5)

        ax6 = plt.subplot(3, 2, 6)
        plot_scatter_with_matched_reglines(data, 2, ax6)

        plt.tight_layout()
        return fig

    visualizations = create_visualizations(df_clean)

    return {
        "summary_stats": summary_stats,
        "manova_results": manova_results,
        "quant_expectation": quant_results,
        "final_stock": stock_results,
        "correlations": correlations,
        "visualizations": visualizations,
    }


def interpret_results(results):
    """
    Generate a formatted interpretation of the analysis results
    """
    interpretation = {
        "Summary": "Statistical analysis results:",
        "MANOVA": f"MANOVA p-value: {results['manova_results'].results['Intercept']['stat']['Value'].iat[0]:.4f}",
        "Quant_Effects": {
            "Treatment": f"Effect size (η²): {results['quant_expectation']['effect_sizes']['treatment']:.4f}",
            "Round": f"Effect size (η²): {results['quant_expectation']['effect_sizes']['round']:.4f}",
            "Interaction": f"Effect size (η²): {results['quant_expectation']['effect_sizes']['interaction']:.4f}",
        },
        "Stock_Effects": {
            "Treatment": f"Effect size (η²): {results['final_stock']['effect_sizes']['treatment']:.4f}",
            "Round": f"Effect size (η²): {results['final_stock']['effect_sizes']['round']:.4f}",
            "Interaction": f"Effect size (η²): {results['final_stock']['effect_sizes']['interaction']:.4f}",
        },
        "Correlations": results["correlations"].to_dict(),
    }
    return interpretation


def create_statistical_visualizations(results: dict) -> plt.Figure:
    """
    Create visualizations for statistical effects analysis using results from analyze_treatment_effects

    Args:
        results (dict): Results dictionary from analyze_treatment_effects containing:
            - quant_expectation: ANOVA results for quantitative expectations
            - final_stock: ANOVA results for final stock
            - manova_results: MANOVA test results

    Returns:
        plt.Figure: Figure containing four subplots of statistical visualizations
    """
    fig = plt.figure(figsize=(20, 15))

    # 1. Effect Sizes Comparison
    # Extract Pillai's trace values from the stat DataFrames
    manova_effects = [
        results["manova_results"]
        .results["C(treatment)"]["stat"]
        .loc["Pillai's trace", "Value"],
        results["manova_results"]
        .results["C(round)"]["stat"]
        .loc["Pillai's trace", "Value"],
        results["manova_results"]
        .results["C(treatment):C(round)"]["stat"]
        .loc["Pillai's trace", "Value"],
    ]
    manova_effects = np.array(manova_effects) * 100  # Convert to percentage

    quant_effects = [
        results["quant_expectation"]["effect_sizes"]["treatment"] * 100,
        results["quant_expectation"]["effect_sizes"]["round"] * 100,
        (
            results["quant_expectation"]["effect_sizes"]["interaction"] * 100
            if results["quant_expectation"]["effect_sizes"]["interaction"]
            else 0
        ),
    ]
    stock_effects = [
        results["final_stock"]["effect_sizes"]["treatment"] * 100,
        results["final_stock"]["effect_sizes"]["round"] * 100,
        (
            results["final_stock"]["effect_sizes"]["interaction"] * 100
            if results["final_stock"]["effect_sizes"]["interaction"]
            else 0
        ),
    ]

    effect_sizes = pd.DataFrame(
        {
            "Effect": ["Treatment", "Round", "Interaction"],
            "MANOVA (Pillai)": manova_effects,
            "Quant_Exp": quant_effects,
            "Final_Stock": stock_effects,
        }
    )

    ax1 = plt.subplot(2, 2, 1)
    effect_sizes_melted = pd.melt(
        effect_sizes,
        id_vars=["Effect"],
        var_name="Analysis",
        value_name="Variance_Explained",
    )

    sns.barplot(
        data=effect_sizes_melted,
        x="Effect",
        y="Variance_Explained",
        hue="Analysis",
        ax=ax1,
    )

    ax1.set_title("Variance Explained (%) by Different Effects", pad=20)
    ax1.set_ylabel("Variance Explained (%)")
    ax1.tick_params(axis="x", rotation=45)

    # 2. Treatment Effects on Final Stock
    summary_stats = results["summary_stats"]
    treatments = summary_stats.index.get_level_values("treatment").unique()
    rounds = [1, 2]

    ax2 = plt.subplot(2, 2, 2)

    for treatment in treatments:
        stock_values = [
            summary_stats.loc[(treatment, round_num)]["finalStock_Mean"]
            for round_num in rounds
        ]
        ax2.plot(rounds, stock_values, "o-", label=treatment, linewidth=2)

    ax2.set_title("Treatment Effects on Final Stock Across Rounds", pad=20)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Final Stock Value")
    ax2.legend()
    ax2.grid(True)

    # 3. Treatment Effects on Quant Expectations
    ax3 = plt.subplot(2, 2, 3)

    for treatment in treatments:
        quant_values = [
            summary_stats.loc[(treatment, round_num)]["Quant_Exp_Mean"]
            for round_num in rounds
        ]
        ax3.plot(rounds, quant_values, "o-", label=treatment, linewidth=2)

    ax3.set_title(
        "Treatment Effects on Quantitative Expectations Across Rounds", pad=20
    )
    ax3.set_xlabel("Round")
    ax3.set_ylabel("Quant Expectation Value")
    ax3.legend()
    ax3.grid(True)

    # 4. Interaction Effects Visualization
    ax4 = plt.subplot(2, 2, 4)
    control_treatment = treatments[0]  # Assuming first treatment is control

    for treatment in treatments[1:]:  # Skip control
        stock_diff = [
            summary_stats.loc[(treatment, round_num)]["finalStock_Mean"]
            - summary_stats.loc[(control_treatment, round_num)]["finalStock_Mean"]
            for round_num in rounds
        ]
        ax4.plot(
            rounds,
            stock_diff,
            "o-",
            label=f"{treatment} - Control (Stock)",
            linewidth=2,
        )

    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax4.set_title("Treatment Effects Relative to Control (Final Stock)", pad=20)
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Difference from Control")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig


def create_detailed_effects_table(results: dict) -> pd.DataFrame:
    """
    Create a formatted table of statistical effects using results from analyze_treatment_effects

    Args:
        results (dict): Results dictionary from analyze_treatment_effects containing:
            - manova_results: MANOVA test results
            - quant_expectation: ANOVA results for quantitative expectations
            - final_stock: ANOVA results for final stock

    Returns:
        pd.DataFrame: Formatted table of statistical effects
    """

    def format_pvalue(p: float) -> str:
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    # Extract MANOVA Pillai's trace
    manova_pillai = [
        results["manova_results"]
        .results["C(treatment)"]["stat"]
        .loc["Pillai's trace", "Value"],
        results["manova_results"]
        .results["C(round)"]["stat"]
        .loc["Pillai's trace", "Value"],
        results["manova_results"]
        .results["C(treatment):C(round)"]["stat"]
        .loc["Pillai's trace", "Value"],
    ]

    # Extract coefficients and p-values from ANOVA results
    quant_params = results["quant_expectation"]["anova_results"].params
    quant_pvals = results["quant_expectation"]["anova_results"].pvalues

    stock_params = results["final_stock"]["anova_results"].params
    stock_pvals = results["final_stock"]["anova_results"].pvalues

    def format_beta(param: float, pval: float) -> str:
        stars = format_pvalue(pval)
        return f"{param:.2f}{stars}"

    effects_data = {
        "Effect": ["Treatment", "Round", "Interaction"],
        "MANOVA_Pillai": [f"{p:.4f}" for p in manova_pillai],
        "Quant_Exp_Beta": [
            format_beta(quant_params["treatment[T.1]"], quant_pvals["treatment[T.1]"]),
            format_beta(quant_params["C(round)[T.2]"], quant_pvals["C(round)[T.2]"]),
            format_beta(
                quant_params["treatment[T.1]:C(round)[T.2]"],
                quant_pvals["treatment[T.1]:C(round)[T.2]"],
            ),
        ],
        "Final_Stock_Beta": [
            format_beta(stock_params["treatment[T.1]"], stock_pvals["treatment[T.1]"]),
            format_beta(stock_params["C(round)[T.2]"], stock_pvals["C(round)[T.2]"]),
            format_beta(
                stock_params["treatment[T.1]:C(round)[T.2]"],
                stock_pvals["treatment[T.1]:C(round)[T.2]"],
            ),
        ],
        "Significance": ["*p<.05", "**p<.01", "***p<.001"],
    }

    return pd.DataFrame(effects_data).set_index("Effect")


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
    df_decisions.head()

    # Month 12
    results = analyze_treatment_effects(df_decisions, 12)

    interpretation = interpret_results(results)
    print(interpretation)

    _ = create_statistical_visualizations(results)
    plt.show()

    # effects_table = create_detailed_effects_table(results)
    # print(effects_table)

    # Month 36
    results = analyze_treatment_effects(df_decisions, 36)

    interpretation = interpret_results(results)
    print(interpretation)

    _ = create_statistical_visualizations(results)
    plt.show()

    # effects_table = create_detailed_effects_table(results)
    # print(effects_table)


if __name__ == "__main__":
    main()
