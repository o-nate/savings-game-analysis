---
name: Holt-Laury CARA gamma
overview: Add a count-based Holt-Laury -> CARA gamma estimator to src/econ_preferences.py, then feed per-participant gamma interval endpoints and midpoint into the Experiment-2 regression in scripts/results.py as a low/mid/high sensitivity set alongside quant_expectation_t1 + round.
todos:
  - id: estimator
    content: Add HL constants + count-based CARA gamma estimator (cara_gamma_intervals with brentq root-finder and explicit gamma->0 branch) to src/econ_preferences.py
    status: completed
  - id: merge-gamma
    content: In scripts/results.py exp-2 block, compute per-participant risk_gamma_low/high/mid from riskPreferences_choice_count (exp==2) and merge on participant.label into df_regress_exp2
    status: completed
  - id: regression-specs
    content: Replace single summary_col with low/mid/high sensitivity specs per outcome (early_percent, excess_percent), each ~ quant_expectation_t1 + round + gamma_col
    status: completed
  - id: verify
    content: Lint-check edited files and present a diff/summary for review
    status: completed
isProject: false
---

## Confirmed decisions

- Count method only (no switch-based gamma).
- Column names: `risk_gamma_low`, `risk_gamma_high`, `risk_gamma_mid`.
- Regression: interval-sensitivity specs -- run each outcome 3x (low / mid / high), each `~ quant_expectation_t1 + round + <gamma_col>`, side-by-side via `summary_col`.

## Verified inputs (Step 1)

- Safe-choice count already exists: `riskPreferences_choice_count` (= number of Option A / safe choices, k in 0..10) in [src/econ_preferences.py](src/econ_preferences.py). Reuse; do not recompute.
- Inconsistency already exists: `riskPreferences_switches`. Left untouched.
- HL payoffs (from [savings-game/riskPreferences/stimuli.csv](savings-game/riskPreferences/stimuli.csv)): Option A (safe) high 2.00 / low 1.60; Option B (risky) high 3.85 / low 0.10; `p` = P(high payoff) = 0.10..1.00 over the 10 rows, same `p` for both options.
- Merge key: `participant.label` (one HL measurement per participant; constant across exp-2 rounds).

## Edit 1: estimator in [src/econ_preferences.py](src/econ_preferences.py)

Add module constants + one public function (private helpers underscore-prefixed). Uses `scipy.optimize.brentq`.

- Normalized CARA `u(x) = (1 - e^{-gamma x}) / gamma` so it -> `x` as gamma -> 0; explicit risk-neutral branch when `abs(gamma) < eps` returns expected value. This is the explicit gamma->0 handling and avoids the degeneracy of `-e^{-gamma x}` (which collapses to a constant at gamma=0).
- Gap `EU_B(gamma) - EU_A(gamma)` is strictly monotone in gamma; root-find with `brentq` over a safe bracket `[-50, 50]` (payoffs <= 3.85, no overflow). If both options dominate across the bracket, return `+inf` (B dominates, e.g. p=1.0) or `-inf` (A dominates, p->0).

```python
from scipy.optimize import brentq

HL_OPTION_A = (2.00, 1.60)   # safe (high, low), from stimuli.csv
HL_OPTION_B = (3.85, 0.10)   # risky (high, low)
HL_HIGH_PROBS = [i / 10 for i in range(1, 11)]  # P(high) per row, 0.1..1.0

def _cara_eu(gamma, high, low, p):
    if abs(gamma) < 1e-9:
        return p * high + (1 - p) * low
    return (1 - (p * np.exp(-gamma * high) + (1 - p) * np.exp(-gamma * low))) / gamma

def _indifference_gap(gamma, p):
    return _cara_eu(gamma, *HL_OPTION_B, p) - _cara_eu(gamma, *HL_OPTION_A, p)

def _solve_gamma_at_p(p, bracket=(-50.0, 50.0)):
    lo, hi = bracket
    g_lo, g_hi = _indifference_gap(lo, p), _indifference_gap(hi, p)
    if g_lo > 0 and g_hi > 0:   # B dominates for all gamma (p == 1.0)
        return np.inf
    if g_lo < 0 and g_hi < 0:   # A dominates for all gamma (p -> 0)
        return -np.inf
    return brentq(_indifference_gap, lo, hi, args=(p,))
```

Public function (the single estimator), count-based:

```python
def cara_gamma_intervals(safe_counts: pd.Series) -> pd.DataFrame:
    """CARA gamma interval (low, high, midpoint) per Holt-Laury safe-choice count k.
    k safe choices => indifference bracket between rows k and k+1:
    gamma in [gamma*(p_k), gamma*(p_{k+1})]. Open sides -> NaN (all-risky lower side,
    all-safe upper side, and the p=1 dominance row); midpoint NaN unless both finite.
    """
    edges = [_solve_gamma_at_p(p) for p in HL_HIGH_PROBS]  # gamma*(0.1..1.0)
    rows = {}
    for k in range(11):
        low = edges[k - 1] if k >= 1 else -np.inf   # gamma*(p_k)
        high = edges[k] if k <= 9 else np.inf       # gamma*(p_{k+1})
        low = low if np.isfinite(low) else np.nan
        high = high if np.isfinite(high) else np.nan
        mid = (low + high) / 2 if np.isfinite(low) and np.isfinite(high) else np.nan
        rows[k] = (low, high, mid)
    out = safe_counts.round().map(rows)  # NaN counts -> NaN triple
    return pd.DataFrame(
        out.tolist(), index=safe_counts.index,
        columns=["risk_gamma_low", "risk_gamma_high", "risk_gamma_mid"],
    )
```

Censoring outcome: `k=0` -> (NaN, finite, NaN); `k=1..8` -> both finite + midpoint; `k=9` -> (finite, NaN, NaN); `k=10` -> (NaN, NaN, NaN).

## Edit 2: integrate into [scripts/results.py](scripts/results.py) (block at lines 1530-1574)

`econ_preferences` is already imported. After `df_regress_exp2` is built and renamed:

- Pull per-participant safe counts for exp 2 from `df_econ_preferences_all`, compute gamma, merge on `participant.label`:

```python
risk_counts = (
    df_econ_preferences_all.loc[
        df_econ_preferences_all["exp"] == 2,
        ["participant.label", "riskPreferences_choice_count"],
    ].drop_duplicates("participant.label")
)
risk_counts[["risk_gamma_low", "risk_gamma_high", "risk_gamma_mid"]] = (
    econ_preferences.cara_gamma_intervals(risk_counts["riskPreferences_choice_count"])
)
df_regress_exp2 = df_regress_exp2.merge(
    risk_counts.drop(columns="riskPreferences_choice_count"),
    on="participant.label", how="left",
)
```

- Replace the single-loop `summary_col` with low/mid/high sensitivity specs per outcome:

```python
gamma_specs = {"low": "risk_gamma_low", "mid": "risk_gamma_mid", "high": "risk_gamma_high"}
results_tables = {}
for m in ["early_percent", "excess_percent"]:
    regressions = {}
    for label, col in gamma_specs.items():
        model = smf.ols(f"{m} ~ quant_expectation_t1 + round + {col}", data=df_regress_exp2)
        regressions[f"{m} ({label})"] = model.fit()
    results_tables[m] = summary_col(
        results=list(regressions.values()), stars=True,
        model_names=list(regressions.keys()),
    )
results_tables["early_percent"]
# results_tables["excess_percent"]
```

Each spec drops its own censored participants via listwise NaN handling (low drops all-risky; high drops all-safe/k=9-10; mid keeps k=1..8). `quant_expectation_t1 + round` retained in every spec.

## Scope / non-goals

- No changes to `create_econ_preferences_dataframe`, `count_switches`, `count_preference_choices`, or any unrelated block.
- No new dependency (`scipy` already used in results.py).
- Will show a diff/summary before finalizing.
