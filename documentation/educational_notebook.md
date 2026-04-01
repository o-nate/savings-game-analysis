# Educational notebook (`savings_game_analysis.ipynb`)

This page covers **creating and distributing** the slim teaching notebook that loads combined CSV exports. For what the exporter writes and how to run `export_analysis_csvs`, see [csv_export.md](csv_export.md).

The instructor notebook [`notebooks/savings_game_analysis.ipynb`](../notebooks/savings_game_analysis.ipynb) loads the CSVs needed for teaching: `decisions_all_enriched.csv`, `knowledge_combined.csv`, `econ_preferences_combined.csv`, and `treatment_effect.csv`. Analysis is restricted to **4×30** (`participant.inflation == 430`).

## GitHub Gist

The generated notebook defaults to **`USE_GIST_URLS = True`** with raw URLs in the first code cell. To refresh data after re-exporting CSVs, upload from `data/csv_export/combined/` to a [public Gist](https://gist.github.com/) (step 1: `uv run python -m scripts.export_analysis_csvs`), copy each file’s **Raw** URL into `URLS` in [`scripts/_gen_educational_notebook.py`](../scripts/_gen_educational_notebook.py), and regenerate the `.ipynb` files.

For offline or repo-local runs, set **`USE_GIST_URLS = False`** in that script before regenerating (or change it in the notebook only).

Large files (especially `decisions_all_enriched.csv`) may hit Gist size limits; see the notebook intro for workarounds.

## Student notebook (no solution cells)

The instructor notebook includes code cells tagged `solution`. To produce a **student** copy without those cells (for Colab):

```bash
uv run python scripts/build_student_notebook.py
```

Writes [`notebooks/savings_game_analysis_student.ipynb`](../notebooks/savings_game_analysis_student.ipynb) by default. Use `--input` / `--output` to override paths.

## Regenerating the instructor notebook

The file [`scripts/_gen_educational_notebook.py`](../scripts/_gen_educational_notebook.py) rebuilds `notebooks/savings_game_analysis.ipynb` from Python (keeps cell tags consistent). Run:

```bash
uv run python scripts/_gen_educational_notebook.py
```

After regenerating, rebuild the student notebook if you distribute it.

## Maintenance

When [`scripts/results.py`](../scripts/results.py) or [`scripts/results_pipeline.py`](../scripts/results_pipeline.py) change in ways that affect exported columns or teaching scope, update [`scripts/_gen_educational_notebook.py`](../scripts/_gen_educational_notebook.py) and the notebooks as needed, then regenerate the instructor and student `.ipynb` files.
