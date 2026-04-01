# Combined CSV export (`scripts/results_pipeline.py` and `scripts/export_analysis_csvs.py`)

## Purpose

These scripts rebuild the **combined experiment 1 + experiment 2** analysis tables that [`scripts/results.py`](../scripts/results.py) constructs in pandas, and write them to CSV so you can load them in a Jupyter notebook and run a lighter analysis without stepping through the full notebook-style script.

Exports are **not** raw DuckDB dumps per experiment. They are **merged checkpoints** (demographics, inflation knowledge estimates, longitudinal decisions, knowledge and preference tasks, behavioral merges, and intervention summary tables).

## Components

### `scripts/results_pipeline.py`

- Defines `ResultsExportBundle`, a dataclass holding every export dataframe.
- **`build_export_bundle(con_exp_1, con_exp_2)`** runs the same logical sequence as `results.py`: opens both DuckDB databases, ensures tables exist (via `create_duckdb_database` when needed), builds per-experiment decision panels, concatenates and filters participants, applies mean bias columns, then branches into:
  - a **pre–behavioral-classification** copy of `df_decisions_all` (before `decision_patterns.classify_subject_decision_patterns`);
  - an **enriched** `df_decisions_all` after perception/purchase-pattern steps and binary recodes used for the behavioral block;
  - combined **knowledge** and **econ preference** tables;
  - **`behavioral_combined`** after joining the behavioral slice to knowledge and econ preferences on `participant.label`, `participant.round`, and `exp`, with `_x` / `_y` suffix cleanup;
  - **learning effect** tables from `intervention.create_learning_effect_table` (two measure lists: “initial” and “complete”, matching the two calls in `results.py`);
  - **treatment effect** from `intervention.create_diff_in_diff_table`.

Constants at the top of this file intentionally **mirror** `results.py`; if you change filters or thresholds in the main results script, update the pipeline here so exports stay aligned.

### `scripts/export_analysis_csvs.py`

- Command-line entry point: connects to the experiment 1 and experiment 2 database files defined in [`scripts/utils/constants.py`](../scripts/utils/constants.py), calls `build_export_bundle`, and writes CSVs.
- **Large tables** are written with UTF-8, comma separation, and **no index** (`index=False`).
- **Summary tables** (learning and treatment effects) keep the dataframe **index** in the CSV (`index=True`), consistent with `set_index("")` in the analysis script.

## Outputs

Default output directory: `data/csv_export/combined/` (created if missing).

| File | Contents |
|------|----------|
| `demographics_combined.csv` | Pooled questionnaire columns from both experiments |
| `inflation_estimates_combined.csv` | Inflation knowledge (`infK_*`) measures with an `experiment` column |
| `decisions_pre_classify_behavioral_patterns.csv` | Combined decisions panel before behavioral pattern classification |
| `decisions_all_enriched.csv` | Same panel after classification and recodes used for behavioral analysis |
| `knowledge_combined.csv` | Pooled literacy / numeracy / compound scores with `exp` |
| `econ_preferences_combined.csv` | Pooled preference-task aggregates with `exp` |
| `behavioral_combined.csv` | Behavioral slice merged with knowledge and econ preferences |
| `learning_effect_initial.csv` | Learning-effect summary (first measure set) |
| `learning_effect_complete.csv` | Learning-effect summary (extended measure set) |
| `treatment_effect.csv` | Diff-in-diff style treatment summary |

Unless `--skip-static` is passed, the exporter also copies **`France HICP_20251112134158.csv`** from `data/` into the output directory when that file is present (for HICP plots in a notebook).

## How to run

From the repository root, with project dependencies available (for example via `uv`):

```bash
uv run python -m scripts.export_analysis_csvs
```

Options:

- **`--output-dir PATH`** — write CSVs somewhere other than `data/csv_export/combined/`.
- **`--skip-static`** — do not copy the France HICP CSV.

DuckDB opens the databases with **write** access because opportunity-cost and related logic may materialize cached tables. If you see a lock error on `data/exp1.duckdb` or `data/exp2.duckdb`, close other processes using those files (another Python session, Jupyter kernel, etc.) and run again.

## Using the CSVs in Jupyter

Typical pattern:

```python
import pandas as pd

out = "data/csv_export/combined"
decisions = pd.read_csv(f"{out}/decisions_all_enriched.csv")
behavioral = pd.read_csv(f"{out}/behavioral_combined.csv")
```

For summary tables that were saved with an index column, either read with `index_col=0` or inspect the first column after `read_csv`.

## Maintenance

When [`scripts/results.py`](../scripts/results.py) changes upstream steps (filters, merges, or intervention lists), update [`scripts/results_pipeline.py`](../scripts/results_pipeline.py) in the same way so exported tables remain comparable to the main analysis.

---

**Educational notebook:** to create, host on Gist, and build student copies of [`notebooks/savings_game_analysis.ipynb`](../notebooks/savings_game_analysis.ipynb), see [educational_notebook.md](educational_notebook.md).
