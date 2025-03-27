"""Statistical analysis of data"""

import logging
from typing import List, Tuple, Type

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, pointbiserialr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src import econ_preferences, knowledge

from src.calc_opp_costs import calculate_opportunity_costs
from src.utils import constants
from src.discontinuity import purchase_discontinuity
from src.utils.helpers import combine_series
from utils.logging_config import get_logger

logger = get_logger(__name__)

# * Output settings
pd.set_option("display.max_rows", None, "display.max_columns", None)

# * Define `decision quantity` measure
DECISION_QUANTITY = "cum_decision"

# * Define purchase window, i.e. how many months before and after inflation phase change to count
WINDOW = 3


def create_pearson_correlation_matrix(
    data: pd.DataFrame,
    p_values: List[float],
    include_stars: bool = True,
    display: bool = False,
    decimal_places: int = 2,
) -> pd.DataFrame:
    """Calculate pearson correlation and p-values and add asterisks
    to relevant values in table

    Args:
        data (pd.DataFrame): Data to correlate
        p_values (List[float]): p value thresholds for stars
        include_stars (bool, optional): Include star for p values. Defaults to True.
        display (bool, optional): Display p values for each correlation. Defaults to False.
        decimal_places (int, optional): Decimal places to include. Defaults to 2.

    Returns:
        pd.DataFrame: Correlation matrix using Pearson correlation with p values stars
    """

    rho = data.corr()
    pval = data.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    cols = data.columns.to_list()
    if display:
        print(f"P-values benchmarks: {p_values}")
        for c in cols:
            print(c)
            print(f"{c} p-values: \n{pval[c]}")
    if include_stars:
        p = pval.applymap(lambda x: "".join(["*" for t in p_values if x <= t]))
        return rho.round(decimal_places).astype(str) + p
    return rho.round(decimal_places)


def create_bonferroni_correlation_table(
    data: pd.DataFrame,
    measures_1: List[str],
    measures_2: List[str],
    correlation: str,
    decimal_places: int = 4,
    filtered_results: bool = True,
) -> Tuple[pd.DataFrame, List[float]]:
    """Create table with Bonferroni-corrected correlations

    Args:
        data (pd.DataFrame): DataFrame with each subjects' individual characteristic
        and task measures
        measures_1 (List[str]): first list of measures to correlate
        measures_2 (List[str]): second list of measures to correlate
        correlation (str): test to apply (pearson, pointbiserial)
        decimal_places (int): decimal places to round to. Defaults to 4
        filtered_results (bool): toggle whether to return only results that pass
        Bonferroni correction or all results. Defaults to True.

    Raises:
        ValueError: Raised to indicate a selection of either `pearson` or `pointbiserial`
        is required.

    Returns:
        Tuple[pd.DataFrame, List[float]]: Tuple of DataFrame with correlation and p-values
        and list of raw p values
    """
    raw_pvals = []
    if correlation == "pearson":
        test = "Pearson correlation"
    elif correlation == "pointbiserial":
        test = "Point bi-serial"
    else:
        raise ValueError("""Please, indicate either `pearson` or `pointbiserial`.""")
    print(f"Applying {test}")
    corr_dict = {"measure": [], "task_measure": [], "correlation": [], "p_value": []}

    ## Create df with correlations to then apply a Bonferroni correction
    for m1 in measures_1:
        for m2 in measures_2:
            if correlation == "pearson":
                ## Pearson correlation for each measure test
                corr = pearsonr(data[m1], data[m2])
            else:
                ## Point bi-serial correlation for each measure test
                corr = pointbiserialr(data[m1], data[m2])
            corr_dict["measure"].append(m1)
            corr_dict["task_measure"].append(m2)
            corr_dict["correlation"].append(corr[0])
            corr_dict["p_value"].append(corr[1])
            raw_pvals.append(corr[1])

    df_corr = pd.DataFrame(corr_dict)

    ## Bonferroni correction
    rejected, _, _, alpha_corrected = multipletests(
        raw_pvals,
        alpha=constants.BONFERRONI_ALPHA,
        method="bonferroni",
        is_sorted=False,
        returnsorted=False,
    )
    print(
        f"Reject null hypothesis for {np.sum(rejected)} of {len(df_corr)} tests.\t",
        f"Corrected alpha: {alpha_corrected}",
    )

    pd.set_option("display.float_format", lambda x: f"%.{decimal_places}f" % x)
    if filtered_results:
        return df_corr[df_corr["p_value"] < alpha_corrected]
    return df_corr


def run_forward_selection(
    data: pd.DataFrame, response: str, categoricals: List[str]
) -> Type[sm.regression.linear_model.RegressionResultsWrapper]:
    """Conduct forward selection of feature variables. The algorithm's objective is
    to maximize adjusted R^2.

    Args:
        data (pd.DataFrame): DataFrame with a column for each feature variable and the
        dependent variable
        response (str): Name of the dependent variable
        categoricals (List[str]): List of column names for variables that are categorical
        (not continuous)

    Returns:
        model (sm.regression.linear_model.RegressionResultsWrapper): OLS regression
        model with maximized adjusted R^2
    """
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            if candidate in categoricals:
                formula = (
                    f"""{response} ~ {" + ".join(selected + [f"C({candidate})"])} + 1"""
                )
            else:
                formula = f"""{response} ~ {" + ".join(selected + [candidate])} + 1"""
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        logger.debug("best: %s", best_candidate)
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            if best_candidate in categoricals:
                best_candidate = f"C({best_candidate})"
            selected.append(best_candidate)
            current_score = best_new_score
    formula = f"""{response} ~ {" + ".join(selected)} + 1"""
    model = smf.ols(formula, data).fit()
    return model


def run_treatment_forward_selection(
    data: pd.DataFrame,
    response: str,
    categoricals: List[str],
    treatment: str = "treatment",
) -> Type[sm.regression.linear_model.RegressionResultsWrapper]:
    """Conduct forward selection of feature variables of treatment regression.
    The algorithm's objective is to maximize adjusted R^2.

    Args:
        data (pd.DataFrame): DataFrame with a column for each feature variable and the
        dependent variable
        response (str): Name of the dependent variable
        categoricals (List[str]): List of column names for variables that are categorical
        (not continuous)

    Returns:
        model (sm.regression.linear_model.RegressionResultsWrapper): OLS regression
        model with maximized adjusted R^2
    """
    remaining = set(data.columns)
    remaining.remove(response)
    remaining.remove(treatment)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            if candidate in categoricals:
                formula = f"""{response} ~ C({treatment}) / {" + ".join(selected + [f"C({candidate})"])} + 1"""
            else:
                formula = f"""{response} ~ C({treatment}) / {" + ".join(selected + [candidate])} + 1"""
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            if best_candidate in categoricals:
                best_candidate = f"C({best_candidate})"
            selected.append(best_candidate)
            current_score = best_new_score
    formula = f"""{response} ~ C({treatment}) / {" + ".join(selected)} + 1"""
    model = smf.ols(formula, data).fit()
    return model


def main() -> None:
    """Run script"""
    df_results = calculate_opportunity_costs()
    logger.debug(df_results.shape)

    df_results = purchase_discontinuity(
        df_results, decision_quantity=DECISION_QUANTITY, window=WINDOW
    )

    # * Forward selection
    df_knowledge = knowledge.create_knowledge_dataframe()
    df_econ_preferences = econ_preferences.create_econ_preferences_dataframe()
    df_individual_char = combine_series(
        [df_results, df_knowledge, df_econ_preferences],
        how="left",
        on="participant.label",
    )
    logger.debug("indiv shape %s", df_individual_char.shape)

    data = df_individual_char[
        (df_individual_char["month"] == 120) & (df_individual_char["phase"] == "pre")
    ].copy()

    print(
        data[
            [
                "sreal",
                "financial_literacy",
                "numeracy",
                "compound",
                "wisconsin_choice_count",
                "riskPreferences_choice_count",
            ]
        ].head()
    )

    model = run_forward_selection(
        data=data[
            [
                "sreal",
                "financial_literacy",
                "numeracy",
                "compound",
                "wisconsin_choice_count",
                "riskPreferences_choice_count",
            ]
        ],
        response="sreal",
        categoricals=["financial_literacy", "numeracy", "compound"],
    )
    print(model.summary())

    model = run_treatment_forward_selection(
        data=data[
            [
                "treatment",
                "sreal",
                "financial_literacy",
                "numeracy",
                "compound",
                "wisconsin_choice_count",
                "riskPreferences_choice_count",
            ]
        ],
        response="sreal",
        categoricals=["financial_literacy", "numeracy", "compound"],
    )
    print(model.summary())


if __name__ == "__main__":
    main()
