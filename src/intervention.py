"""Script to analyze intervention's effect"""

import logging
import sys

from pathlib import Path
from typing import List, Tuple

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns

from src.calc_opp_costs import calculate_opportunity_costs
from src.discontinuity import purchase_discontinuity
from src.utils.database import create_duckdb_database, table_exists
from utils.logging_config import get_logger

# * Logging settings
logger = get_logger(__name__)

DATABASE_FILE = Path(__file__).parents[1] / "data" / "database.duckdb"


# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3

"""Print comparison of performance before and after intervention with p values"""


def calculate_change_in_measure(
    data: pd.DataFrame, measure_impacted: str, display_results: bool = False
) -> Tuple[float, float, float]:
    """Calculate change in performance measure

    Args:
        data (pd.DataFrame): DataFrame with performance measures and participant labels
        measure_impacted (str): Measure to calculate change for
        display_results (bool, optional): Toggle whether results are printed directly.
        Defaults to False.

    Returns:
        Tuple[float, float, float]: Mean values before and after and associate p-value
    """
    before = data[(data["phase"] == "pre")][measure_impacted]
    after = data[(data["phase"] == "post")][measure_impacted]
    p_value = stats.wilcoxon(before, after, zero_method="zsplit", nan_policy="raise")[1]
    if display_results:
        print(f"Initial {measure_impacted}: {before.mean()}")
        print(f"Final {measure_impacted}: {after.mean()}")
        print(
            f"Change in {measure_impacted}:",
            after.mean() - before.mean(),
        )
        print(f"p value for change in {measure_impacted}:\t{p_value}")
    return before.mean(), after.mean(), p_value


def create_learning_effect_table(
    data: pd.DataFrame,
    measures: List[str],
    p_value_threshold: List[float],
    decimal_places: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate table to show the change in performance measures between Savings Game
    rounds.

    Args:
        data (pd.DataFrame): DataFrame with performance measures, rounds, and
        participant labels
        measures (List[str]): Measures to calculate change for
        p_value_threshold (List[float]): List of p-values that correspond to stars
        added on results
        decimal_places (int, optional): Decimal place to round to. Defaults to 2.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrame with results and DataFrame with
        pivoted aggregate data
    """
    ## Create pivot table to calculate difference pre- and post-treatment
    df_pivot = pd.pivot_table(
        data[["participant.label", "phase", "treatment"] + measures],
        index=["participant.label", "treatment"],
        columns=["phase"],
    )
    df_pivot.reset_index(inplace=True)
    header_column = {"": [m for i in measures for m in [i, "(std)"]]}
    results_columns = {"Session 1": [], "Session 2": [], "Change in performance": []}
    dict_for_dataframe = header_column | results_columns
    for m in measures:
        df_pivot[f"Change in {m}"] = df_pivot[(m, "post")] - df_pivot[(m, "pre")]
        before, after, p_value = calculate_change_in_measure(data, m)

        ## Add difference
        diff = str(round(after - before, decimal_places))
        for pval in p_value_threshold:
            diff += "*" if p_value <= pval else ""
        logger.debug(
            "measure: %s, after: %s, before: %s, diff: %s, pval: %s",
            m,
            after,
            before,
            diff,
            p_value,
        )
        dict_for_dataframe["Session 1"].append(before)
        dict_for_dataframe["Session 2"].append(after)
        dict_for_dataframe["Change in performance"].append(diff)

        ## Add standard deviation
        standard_deviation = str(
            round(
                df_pivot[(m, "pre")].std(),
                decimal_places,
            )
        )
        dict_for_dataframe["Session 1"].append(f"({standard_deviation})")
        standard_deviation = str(
            round(
                df_pivot[(m, "post")].std(),
                decimal_places,
            )
        )
        dict_for_dataframe["Session 2"].append(f"({standard_deviation})")
        standard_deviation = str(
            round(
                df_pivot[f"Change in {m}"].std(),
                decimal_places,
            )
        )
        dict_for_dataframe["Change in performance"].append(f"({standard_deviation})")
    return pd.DataFrame(dict_for_dataframe), df_pivot


def create_diff_in_diff_table(
    data: pd.DataFrame,
    measures: List[str],
    treatments: List[str],
    p_value_threshold: List[float],
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Generate table to show difference-in-difference results between treatments

    Args:
        data (pd.DataFrame): DataFrame with performance measures, rounds,
        treatment groups, and participant labels
        measures (List[str]): List of measures to calculate change for
        treatments (List[str]): List of treatment group names
        p_value_threshold (List[float]): List of p-values that correspond to stars
        added on results
        decimal_places (int, optional): Decimal place to round to. Defaults to 2.s

    Returns:
        pd.DataFrame: DataFrame with diff-in-diff results
    """
    ## Create pivot table to calculate difference pre- and post-treatment
    df_pivot = pd.pivot_table(
        data[["participant.label", "phase", "treatment"] + measures],
        index=["participant.label", "treatment"],
        columns=["phase"],
    )
    df_pivot.reset_index(inplace=True)
    header_column = {"": [m for i in measures for m in [i, "(std)"]]}
    results_columns = {t: [] for t in treatments}
    dict_for_dataframe = header_column | results_columns
    for m in measures:
        df_pivot[f"Change in {m}"] = df_pivot[(m, "post")] - df_pivot[(m, "pre")]
        for treat in treatments:
            before, after, p_value = calculate_change_in_measure(
                data[data["treatment"] == treat], m
            )

            ## Add difference
            diff = str(round(after - before, decimal_places))
            for pval in p_value_threshold:
                diff += "*" if p_value <= pval else ""
            logger.debug(
                "measure: %s, treatment: %s, after: %s, before: %s, diff: %s, pval: %s",
                m,
                treat,
                after,
                before,
                diff,
                p_value,
            )
            dict_for_dataframe[treat].append(diff)

            ## Add standard deviation
            standard_deviation = str(
                round(
                    df_pivot[df_pivot["treatment"] == treat][f"Change in {m}"].std(),
                    decimal_places,
                )
            )
            dict_for_dataframe[treat].append(f"({standard_deviation})")
    return pd.DataFrame(dict_for_dataframe)


def measure_feedback_impact(
    data: pd.DataFrame, measures_impacted: List[str], error_feedback: List[str]
) -> None:
    """Print comparison between those who are convinced by intervention feedback
    and not across measures

    Args:
        data (pd.DataFrame): DataFrame with performance measures, rounds,
        treatment groups, and participant labels
        measures_impacted (List[str]): Measure to calculate change for
        error_feedback (List[str]): List of pieces of feedback to analyze
    """
    for m in measures_impacted:
        for error in error_feedback:
            convinced_response = 9 if error == "convinced" else 3
            field = (
                error if error == "convinced" else f"task_int.1.player.confirm_{error}"
            )
            before = data[
                (data["phase"] == "pre") & (data[field] == convinced_response)
            ][m]
            after = data[
                (data["phase"] == "post") & (data[field] == convinced_response)
            ][m]
            p_value = stats.wilcoxon(
                before, after, zero_method="zsplit", nan_policy="raise"
            )[1]
            print(f"p value for convinced of {error}, {m}:\t{p_value}")
            print(
                "Change in measure:",
                data[(data["phase"] == "post") & (data[field] == convinced_response)][
                    m
                ].mean()
                - data[(data["phase"] == "pre") & (data[field] == convinced_response)][
                    m
                ].mean(),
            )
            ## Not very convinced
            before = data[
                (data["phase"] == "pre") & (data[field] < convinced_response)
            ][m]
            after = data[
                (data["phase"] == "post") & (data[field] < convinced_response)
            ][m]
            p_value = stats.wilcoxon(
                before, after, zero_method="zsplit", nan_policy="raise"
            )[1]
            print(f"p value for not convinced of {error}, {m}:\t{p_value}")
            print(
                "Change in measure:",
                data[(data["phase"] == "post") & (data[field] < 3)][m].mean()
                - data[(data["phase"] == "pre") & (data[field] < 3)][m].mean(),
            )


def main() -> None:
    """Run script"""
    con = duckdb.connect(DATABASE_FILE, read_only=False)
    if table_exists(con, "task_int") == False:
        create_duckdb_database(con, initial_creation=True)
    df_int = con.sql("SELECT * FROM task_int").df()

    df_results = calculate_opportunity_costs()
    logging.debug(df_results.shape)

    df_results = purchase_discontinuity(
        df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
    )

    questions = ["intro_1", "q", "confirm"]
    cols = [c for c in df_int.columns if any(q in c for q in questions)]
    logging.debug(cols)

    # TODO Link mistakes participants made to their questions and responses
    # * Compare impact of intervention
    measures = [
        "finalSavings",
        "early",
        "late",
        "excess",
    ]

    data_df = df_results[df_results["month"] == 120].copy()
    logging.debug(data_df.shape)
    data_df = data_df.merge(df_int[["participant.label", "date"] + cols], how="left")

    # * Rename mistakes
    data_df.rename(
        columns={
            "finalSavings": "Total savings",
            "early": "Over-stocking",
            "late": "Under-stocking",
            "excess": "Wasteful-stocking",
        },
        inplace=True,
    )

    measures = ["Total savings", "Over-stocking", "Under-stocking", "Wasteful-stocking"]

    data_df["convinced"] = data_df[[c for c in data_df.columns if "confirm" in c]].sum(
        axis=1
    )
    logging.debug([c for c in data_df.columns if "confirm" in c])
    print(data_df.head())

    # * Measure learning effect
    learning_effect = create_learning_effect_table(
        data_df,
        measures=measures,
        p_value_threshold=[0.1, 0.05, 0.01],
    )
    print("\nlearning effect")
    print(learning_effect)

    # * Measure intervention impact
    diff_results = create_diff_in_diff_table(
        data_df,
        measures=measures,
        treatments=["Intervention 1", "Intervention 2", "Control"],
        p_value_threshold=[0.1, 0.05, 0.01],
    )
    print("\ndiff in diff")
    print(diff_results)

    # * Measure impact of intervention feedback
    measure_feedback = input("Measure impact of intervention feedback? (y/n):")
    if measure_feedback not in ["y", "n"]:
        measure_feedback = input("Please respond with 'y' or 'n':")
    elif measure_feedback == "n":
        pass
    else:
        ## Fully convinced
        measure_feedback_impact(data_df, measures, ["convinced"])

        ## Mostly convinced
        errors = ["early", "excess"]
        measure_feedback_impact(data_df, measures, errors)

    graph_data = input("Plot intervention data? (y/n):")
    if graph_data not in ("y", "n"):
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        df_melted = data_df.melt(
            id_vars=[
                "participant.code",
                "participant.label",
                "date",
                "participant.round",
                "treatment",
                "convinced",
                "task_int.1.player.confirm_early",
                "task_int.1.player.confirm_late",
                "task_int.1.player.confirm_excess",
            ],
            value_vars=measures,
            var_name="Measure",
            value_name="Result",
        )

        ## Convert to dummy variables for plotting
        df_melted["convinced"] = [
            True if answer == 9 else False for answer in df_melted["convinced"]
        ]
        df_melted["task_int.1.player.confirm_early"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_early"]
        ]
        df_melted["task_int.1.player.confirm_late"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_late"]
        ]
        df_melted["task_int.1.player.confirm_excess"] = [
            True if answer == 3 else False
            for answer in df_melted["task_int.1.player.confirm_excess"]
        ]

        # * Plots by date
        sns.catplot(
            data=df_melted,
            x="Measure",
            y="Result",
            col="treatment",
            hue="participant.round",
            kind="violin",
            split=True,
        )

        # * Plots by being convinced overall
        sns.catplot(
            data=df_melted,
            x="Measure",
            y="Result",
            col="convinced",
            hue="participant.round",
            kind="violin",
            split=True,
        )

        for m in ["early", "late", "excess"]:
            sns.catplot(
                data=df_melted,
                x="Measure",
                y="Result",
                col=f"task_int.1.player.confirm_{m}",
                row="treatment",
                hue="participant.round",
                kind="violin",
                split=True,
            )

        plt.show()

    graph_data = input("Plot general response data? (y/n):")
    if graph_data not in ("y", "n"):
        graph_data = input("Please respond with 'y' or 'n':")
    if graph_data == "y":
        data = df_int.melt(
            id_vars=[
                "participant.code",
                "participant.label",
            ],
            value_vars=cols,
            var_name="Measure",
            value_name="Result",
        )

        g = sns.FacetGrid(data, row="Measure")
        g.map(plt.hist, "Result")

        # h = sns.FacetGrid(
        #     data=data[data["Measure"].isin(cols)],
        #     col="Month",
        #     height=2.5,
        #     col_wrap=3,
        #     hue="Measure",
        # )
        plt.show()


if __name__ == "__main__":
    main()
