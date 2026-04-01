"""One-off generator for notebooks/savings_game_analysis.ipynb — run from repo root: uv run python scripts/_gen_educational_notebook.py"""

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "savings_game_analysis.ipynb"


def md(text: str):
    return new_markdown_cell(text)


def code(text: str, *, solution: bool = False):
    c = new_code_cell(text)
    if solution:
        c.setdefault("metadata", {})["tags"] = ["solution"]
    return c


cells = [
    md(
        """# Savings game results (educational)

This notebook is a simplified companion to [`scripts/results.py`](../scripts/results.py). It uses **pre-exported CSVs** (see [`documentation/csv_export.md`](../documentation/csv_export.md); for Gist, student copies, and regenerating this file, see [`documentation/educational_notebook.md`](../documentation/educational_notebook.md)) and focuses on the **4×30 inflation sequence** (`participant.inflation == 430`).

**Instructor:** keep the full notebook (including cells tagged `solution`). **Students (Colab):** use the generated `savings_game_analysis_student.ipynb`, which omits solution cells.

### Data from GitHub Gist

The notebook defaults to **`USE_GIST_URLS = True`** with raw URLs in `URLS` below. For local development without downloading from Gist, set **`USE_GIST_URLS = False`** to read from `data/csv_export/combined/` relative to the repository root. To use your own Gist, replace the URLs (see [educational_notebook.md](../documentation/educational_notebook.md)).

Sections 4–5 need `knowledge_combined.csv` and `econ_preferences_combined.csv` in addition to `decisions_all_enriched.csv`."""
    ),
    code(
        """from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# True loads from URLS below; False uses local CSVs under data/csv_export/combined (repo root)
USE_GIST_URLS = True

# Raw URLs per file (GitHub Gist → Raw). Filenames must match keys.
URLS = {
    "decisions_all_enriched": "https://gist.githubusercontent.com/o-nate/bfaf589551d366a2c1bd1640a9920397/raw/39f894a4b3ae2002d4aea2db504b413766f84bdb/decisions_all_enriched.csv",
    "knowledge_combined": "https://gist.githubusercontent.com/o-nate/bfaf589551d366a2c1bd1640a9920397/raw/39f894a4b3ae2002d4aea2db504b413766f84bdb/knowledge_combined.csv",
    "econ_preferences_combined": "https://gist.githubusercontent.com/o-nate/bfaf589551d366a2c1bd1640a9920397/raw/39f894a4b3ae2002d4aea2db504b413766f84bdb/econ_preferences_combined.csv",
    "treatment_effect": "https://gist.githubusercontent.com/o-nate/bfaf589551d366a2c1bd1640a9920397/raw/39f894a4b3ae2002d4aea2db504b413766f84bdb/treatment_effect.csv",
}

def _local_data_dir() -> Path:
    cwd = Path.cwd()
    for base in (cwd, cwd.parent, cwd.parent.parent):
        p = base / "data" / "csv_export" / "combined"
        if p.is_dir():
            return p
    return cwd / "data" / "csv_export" / "combined"


def load_csv(name: str) -> pd.DataFrame:
    if USE_GIST_URLS:
        return pd.read_csv(URLS[name], low_memory=False)
    path = _local_data_dir() / f"{name}.csv"
    return pd.read_csv(path, low_memory=False)


def filter_4x30_sequence(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["participant.inflation"] == 430].copy()"""
    ),
    md(
        """### Load tables

We load the enriched decisions panel and intervention summary CSVs. The decisions file is large; loading may take a minute on Colab."""
    ),
    code(
        """decisions_df = load_csv("decisions_all_enriched")
decisions_filtered = filter_4x30_sequence(decisions_df)

print(decisions_filtered.shape, "rows (4×30 sequence)")"""
    ),
    md(
        """## 1. Savings-game inflation

The column **`Actual`** holds the **realized inflation** in the game for that month (see the “Actual” measure in `process_survey`). Filter to the 4×30 sequence, then average across participants for each `Month`.

### Step 1 — Actual inflation

Plot the cross-participant mean of **`Actual`** over **`Month`**.

**Exercise:** Build the aggregated series and plot it."""
    ),
    code(
        """# Your code: groupby Month, mean Actual, plot

"""
    ),
    code(
        """# Solution — Step 1 (Actual only)
mean_actual = decisions_filtered.groupby("Month")["Actual"].mean().dropna()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_actual.index, mean_actual.values, color="tab:blue", label="Mean Actual")
ax.set_xlabel("Month")
ax.set_ylabel("Inflation rate (%)")
ax.set_title("Savings game: mean realized inflation")
ax.legend()
plt.tight_layout()
plt.show()""",
        solution=True,
    ),
    md(
        """### Step 2 — Actual, expectations, and perceptions (one plot)

Add **`Quant Expectation`** and **`Quant Perception`** (same aggregation: mean by `Month`) on the **same axes** as **`Actual`**, so the figure shows **three** series.

**Exercise:** Plot all three lines on one chart with a legend."""
    ),
    code(
        """# Your code: three series, one axes

"""
    ),
    code(
        """# Solution — Step 2 (three series)
mean_actual = decisions_filtered.groupby("Month")["Actual"].mean().dropna()
mean_expectation = decisions_filtered.groupby("Month")["Quant Expectation"].mean().dropna()
mean_perception = decisions_filtered.groupby("Month")["Quant Perception"].mean().dropna()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(mean_actual.index, mean_actual.values, color="tab:blue", label="Actual")
ax.plot(mean_expectation.index, mean_expectation.values, color="tab:orange", label="Quant Expectation")
ax.plot(mean_perception.index, mean_perception.values, color="tab:green", label="Quant Perception")
ax.set_xlabel("Month")
ax.set_ylabel("Rate (%)")
ax.set_title("Savings game: mean inflation beliefs and realized inflation")
ax.legend()
plt.tight_layout()
plt.show()""",
        solution=True,
    ),
    md(
        """## 2. Performance: savings and stock vs benchmarks

The helper **`plot_savings_and_stock`** below is adapted from [`calc_opp_costs.plot_savings_and_stock`](../src/calc_opp_costs.py) (same logic as [`scripts/results.py`](../scripts/results.py)). It needs **`PLOT_MELT_COLS`** for `pandas.melt` id columns, plus the strategy stock columns, savings columns, and display names. Use **`phase == "pre"`** on the 4×30 panel.

**Exercise:** Filter to the pre phase, then call `plot_savings_and_stock` with `month_col="Month"` and the strategy lists defined below."""
    ),
    code(
        '''# Columns for melt id_vars (matches calc_opp_costs.PLOT_MELT_COLS)
PLOT_MELT_COLS = [
    "participant.code",
    "participant.label",
    "treatment",
    "phase",
    "participant.inflation",
]

# Benchmark vs naïve vs participant average (matches results.py performance plot)
PLOT_STRATEGY_STOCK_COLS = ["sgoptimal", "sgnaive", "finalStock"]
PLOT_STRATEGY_SAVINGS_COLS = ["soptimal", "snaive", "sreal"]
PLOT_STRATEGY_NAMES = ["Benchmark", "Naïve", "Average"]


def plot_savings_and_stock(
    data: pd.DataFrame,
    month_col: str,
    strategy_stock_cols: list[str],
    strategy_savings_cols: list[str],
    strategy_names: list[str],
    ax: plt.Axes = None,
    set_ylim: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot average performance versus optimal and naive strategies."""
    id_variables = PLOT_MELT_COLS + [month_col]
    value_variables = ["participant.inflation"] + strategy_stock_cols
    df_stock = data.melt(
        id_vars=id_variables,
        var_name="Strategy",
        value_vars=value_variables,
        value_name="Stock",
    )

    value_variables = ["participant.inflation"] + strategy_savings_cols
    df_savings = data.melt(
        id_vars=id_variables,
        var_name="Strategy",
        value_vars=value_variables,
        value_name="Savings",
    )

    dfts = pd.concat([df_stock, df_savings], axis=1, join="inner")

    dfts = dfts.drop_duplicates()

    dfts = dfts.loc[:, ~dfts.columns.duplicated()].copy()

    dfts["Strategy"] = dfts["Strategy"].replace(strategy_stock_cols, strategy_names)

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    sns.barplot(
        data=dfts,
        x=month_col,
        y="Stock",
        hue="Strategy",
        ax=ax,
        estimator="mean",
        errorbar=None,
        palette=kwargs.get("palette", "tab10"),
    )

    ax2 = ax.twinx()

    sns.lineplot(
        data=dfts,
        x=month_col,
        y="Savings",
        hue="Strategy",
        ax=ax2,
        legend=None,
        errorbar=None,
        palette=kwargs.get("palette", "tab10"),
    )

    ax.set_ylabel("Quantity in stock", labelpad=20, fontsize=kwargs.get("fontsize", 14))
    ax2.set_ylabel(
        "Savings balance (₮)", labelpad=20, fontsize=kwargs.get("fontsize", 14)
    )

    if set_ylim:
        ax2.set_ylim(0, dfts["Savings"].max() + 500)

    return ax
''',
    ),
    code(
        """# Your code: filter to phase == "pre", then call plot_savings_and_stock

""",
    ),
    code(
        """# Solution
perf = decisions_filtered[(decisions_filtered["phase"] == "pre")]

fig, ax = plt.subplots(figsize=(12, 7))
plot_savings_and_stock(
    perf,
    month_col="Month",
    strategy_stock_cols=PLOT_STRATEGY_STOCK_COLS,
    strategy_savings_cols=PLOT_STRATEGY_SAVINGS_COLS,
    strategy_names=PLOT_STRATEGY_NAMES,
    palette="tab10",
    ax=ax,
    set_ylim=True,
    fontsize=16,
)
ax.set_xlabel("Period", fontsize=16)
ax.legend(loc="upper center", fontsize=16)
ax.set_xticks(ax.get_xticks()[0:120:12])
plt.tight_layout()
plt.show()""",
        solution=True,
    ),
    md(
        """## 3. OLS: overall performance on inflation beliefs

This mirrors **OLS regressions: Overall performance measures on inflation measures** in [`scripts/results.py`](../scripts/results.py) (before the appendix). At `Month == 120` and `phase == "pre"`, regress each of total performance (`sreal_%`), over-stocking (`early_%`), and wasteful-stocking (`excess_%`) on sensitivity and average bias for expectations and perceptions."""
    ),
    code(
        """# Same specification as results.py: avg_* bias columns are precomputed in the export
df_reg_inf = decisions_filtered.rename(
    columns={
        "Mean Perception Bias": "avg_perception_bias",
        "Mean Expectation Bias": "avg_expectation_bias",
        "sreal_%": "sreal_percent",
        "early_%": "early_percent",
        "excess_%": "excess_percent",
    },
)
panel_120 = df_reg_inf[(df_reg_inf["phase"] == "pre") & (df_reg_inf["Month"] == 120)]

regressions_inf = {}
for m in ["sreal_percent", "early_percent", "excess_percent"]:
    regressions_inf[m] = smf.ols(
        formula=(
            f"{m} ~ Expectation_sensitivity + avg_expectation_bias + "
            "Perception_sensitivity + avg_perception_bias"
        ),
        data=panel_120,
    ).fit()

summary_col(
    results=list(regressions_inf.values()),
    stars=True,
    model_names=list(regressions_inf.keys()),
)"""
    ),
    md(
        """## 4. OLS / Logit: behavioral correlates (condensed)

This mirrors **OLS/Logistic regression of performance and decision patterns on behavioral variables (Condensed)** in [`scripts/results.py`](../scripts/results.py). We merge [`knowledge_combined.csv`](../../data/csv_export/combined/knowledge_combined.csv) and [`econ_preferences_combined.csv`](../../data/csv_export/combined/econ_preferences_combined.csv) into the decisions panel, encode pattern dummies, then fit OLS for `sreal_percent` and logit models for a subset of binary outcomes (`LOGIT_COLS` in `results.py`)."""
    ),
    code(
        """LOGIT_COLS = [
    "purchase_adaptation_30",
    "perception_consistent_12",
    "decision_pattern_30_perception_accuracy_SN",
    "decision_pattern_30_perception_accuracy_SA",
    "decision_pattern_30_perception_accuracy_IN",
    "decision_pattern_30_perception_accuracy_IA",
    "perception_pattern_12_AC",
    "perception_pattern_12_AI",
    "perception_pattern_12_IC",
]

knowledge_df = load_csv("knowledge_combined")
econ_df = load_csv("econ_preferences_combined")

df_behavioral = decisions_filtered[decisions_filtered["phase"] >= "pre"].copy()
df_behavioral = df_behavioral.merge(
    knowledge_df, on=["participant.label", "participant.round", "exp"], how="left"
)
df_behavioral = df_behavioral.merge(
    econ_df, on=["participant.label", "participant.round", "exp"], how="left"
)
_y_drop = [c for c in df_behavioral.columns if c.endswith("_y")]
df_behavioral = df_behavioral.drop(columns=_y_drop)
df_behavioral.columns = df_behavioral.columns.str.removesuffix("_x")

df_regress = pd.get_dummies(
    df_behavioral,
    columns=["decision_pattern_30_perception_accuracy"],
    drop_first=False,
    dtype=int,
)
df_regress = pd.get_dummies(
    df_regress,
    columns=["Quant Perception_pattern_12"],
    drop_first=False,
    dtype=int,
)
df_regress = df_regress.rename(
    columns={
        "sreal_%": "sreal_percent",
        "Quant Perception_consistent_12": "perception_consistent_12",
        "Quant Perception_pattern_12_AC": "perception_pattern_12_AC",
        "Quant Perception_pattern_12_AI": "perception_pattern_12_AI",
        "Quant Perception_pattern_12_IC": "perception_pattern_12_IC",
    },
)
df_regress["n_switches"] = df_regress[
    ["lossAversion_switches", "riskPreferences_switches", "timePreferences_switches"]
].sum(axis=1)

pre_120 = df_regress[(df_regress["phase"] == "pre") & (df_regress["Month"] == 120)]

regressions_beh = {}
for m in ["sreal_percent"] + LOGIT_COLS[:2] + LOGIT_COLS[3:5]:
    formula = (
        f"{m} ~ C(financial_literacy) + C(numeracy) + C(compound) + n_switches"
        " + wisconsin_choice_count + lossAversion_choice_count + riskPreferences_choice_count"
        " + timePreferences_choice_count"
    )
    if m in LOGIT_COLS:
        regressions_beh[m] = smf.logit(formula=formula, data=pre_120).fit()
    else:
        regressions_beh[m] = smf.ols(formula=formula, data=pre_120).fit()

info_dict = {
    "Pseudo R-squared": lambda x: (
        "%#8.3f" % x.prsquared if hasattr(x, "prsquared") else ""
    ),
}
summary_col(
    results=list(regressions_beh.values()),
    stars=True,
    model_names=list(regressions_beh.keys()),
    info_dict=info_dict,
)"""
    ),
    md(
        """## 5. Intervention summaries

Precomputed treatment-effect table (from `intervention` in the main pipeline). Read with `index_col=0` because the first column was the index when exported."""
    ),
    code(
        """def load_summary(name: str) -> pd.DataFrame:
    if USE_GIST_URLS:
        return pd.read_csv(URLS[name], index_col=0)
    return pd.read_csv(_local_data_dir() / f"{name}.csv", index_col=0)


from IPython.display import display

display(load_summary("treatment_effect"))""",
    ),
]

nb = new_notebook()
nb["cells"] = cells
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {
    "name": "python",
    "version": "3.10.0",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.write(nb, OUT)
print("Wrote", OUT)
