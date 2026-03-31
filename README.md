# Savings Game Analysis


## Virtual environment

To run this project you need a virtual environment named `savings-game-analysis` (or any name you prefer) with the required libraries installed.

### uv

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. From the repository root, create the virtual environment and install the locked dependencies plus the **editable** project package (so `src` and `utils` imports resolve):

   ```bash
   uv sync
   ```

   Python version follows [`.python-version`](.python-version) (currently 3.10). The project is declared in [`pyproject.toml`](pyproject.toml) with a setuptools build backend; [`uv.lock`](uv.lock) pins the full dependency tree.

You can run scripts with `uv run` without activating the environment (for example, `uv run python src/stats_analysis.py`). To activate the environment manually: **Linux / macOS:** `source .venv/bin/activate`; **Windows (cmd):** `.venv\Scripts\activate.bat`; **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`.

Alternatively, `uv pip install -e .` after `uv venv` performs the same editable install as `uv sync`.

### conda

To create the virtual environment and install libraries from Conda:

```bash
conda env create --file environment.yml
```

Activate it, then install the project in editable mode (see below).

## Editable install (`src` imports)

If you used **conda**, activate that environment and run:

```bash
pip install -e .
```

If you used **uv**, `uv sync` (or `uv pip install -e .`) already performs this step.
