# Claim Denial Risk — Prediction + GenAI Explanations

Predicts whether a hospital claim will be **denied** before it is submitted, so a
front-end review team (which can only inspect the **top 25%** of daily volume)
spends its limited time on the riskiest claims. For the top-10 riskiest current
claims it also produces a short, plain-English, LLM-generated explanation an
analyst can act on in seconds.

## High-level approach

1. **Framing.** The team can review only the top 25% of claims, so this is a
   *ranking / triage* problem, not a balanced-accuracy problem. The primary
   metric is **denial capture at the top 25%** — of all claims that will be
   denied, how many land in the 25% we flag. False negatives (a denial we don't
   flag → silent rework) cost more than false positives (an extra review), so we
   optimise recall within the review budget.
2. **Features.** Only pre-submission fields are used. On top of the raw columns
   we engineer transparent, domain-informed features: `auth_gap` (auth required
   but not on file), `referral_gap`, `late_submission`, `payment_ratio`,
   `readiness_deficits` (count of front-end gaps), and `log_total_billed`.
   `claim_id`, `is_denied`, `denial_reason`, and `split` are never used as
   inputs.
3. **Models.** A `LogisticRegression` baseline is compared against
   `RandomForest` and `GradientBoosting`, plus a transparent non-ML rule
   baseline (rank by number of readiness deficits). We select on validation
   denial-capture@25% (tie-break PR-AUC) and report a single unbiased read on
   the provided **test** split.
4. **Explanations.** Per-claim `top_risk_factors` come from an interpretable
   risk-driver table (each flag's learned denial-rate lift over the base rate).
   These drivers ground an LLM prompt that returns a 2–3 sentence explanation
   with one concrete action, hedged as a risk estimate.

## Project layout

```
denial-risk-model/
├── data/                       # claims_history.csv, current_claims.csv
├── src/
│   ├── config.py               # paths, column groups, constants
│   ├── data.py                 # load + feature engineering + splits
│   ├── model.py                # preprocessing + estimator pipelines
│   ├── evaluate.py             # metrics (capture@25%) + plots
│   ├── risk_drivers.py         # interpretable per-claim risk factors
│   ├── llm.py                  # prompt template + API/offline generator
│   ├── train.py                # train, compare, select, persist
│   ├── predict.py              # score current_claims -> predictions CSV
│   ├── explain.py              # LLM explanations for top-N claims
│   └── eda.py                  # data findings + plots for the write-up
├── outputs/                    # model.pkl, metrics.json, predictions, plots
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

# 1) (optional) data findings + EDA plots
python -m src.eda

# 2) train + compare models, evaluate on the test split, persist best model
python -m src.train --data_path data/claims_history.csv --models logreg rf gbm --seed 42

# 3) score current claims -> outputs/predictions_current_claims.csv
python -m src.predict --model_path outputs/model.pkl \
    --data_path data/current_claims.csv \
    --output outputs/predictions_current_claims.csv

# 4) LLM explanations for the top-10 riskiest current claims
python -m src.explain --predictions outputs/predictions_current_claims.csv \
    --data_path data/current_claims.csv --top_n 10
```

Everything is reproducible with `--seed 42`.

## Testing / verifying the code

An automated `pytest` suite verifies the pipeline's key invariants:

```bash
pip install -r requirements.txt   # includes pytest
python -m pytest                  # runs tests/ (config in pytest.ini)
```

Expected: **24 passing tests in a few seconds.** What they check:

| Area | What is verified | File |
|---|---|---|
| Feature engineering | `auth_gap` / `referral_gap` logic, zero-billed safe `payment_ratio`, `late_submission`, `readiness_deficits`, no NaNs | `tests/test_data.py` |
| No data leakage | `is_denied`, `denial_reason`, `split`, `claim_id` never enter model inputs; split used as-is | `tests/test_data.py` |
| Metric correctness | capture @ top-25% on synthetic perfect/worst/partial rankings; `k = ceil(fraction·n)` | `tests/test_evaluate.py` |
| Model & scoring | probabilities in [0,1]; prediction schema, sort order, valid tiers; `predicted_denial` matches the operating threshold | `tests/test_model_predict.py` |
| Risk drivers | learned lift > 1 for known drivers; per-claim factors grounded in the row | `tests/test_model_predict.py` |
| Reproducibility | training twice with `--seed 42` yields identical test metrics | `tests/test_model_predict.py` |
| GenAI grounding | prompt uses only given fields; explanation cites the claim, names the driver, gives an action, hedges as a risk estimate, invents no dollar amounts; low-risk claim behaves sensibly | `tests/test_llm.py` |

Beyond the unit tests, you can **verify the end-to-end results manually**:

* `outputs/metrics.json` — test-set ROC-AUC / PR-AUC and denial capture @ top 25%.
* `outputs/predictions_current_claims.csv` — 500 rows, sorted by `denial_probability`, tiers, drivers, top-10 explanations.
* `outputs/top10_explanations.json` — the exact prompt + output per claim, plus the low-risk sanity check.
* `writeup/writeup.pdf` — the 2–3 page write-up.

## Predictions output

`outputs/predictions_current_claims.csv`, sorted highest→lowest denial
probability, with:

| column | meaning |
|---|---|
| `claim_id` | from current_claims.csv |
| `denial_probability` | model probability of denial (0–1) |
| `predicted_denial` | 0/1 at the operating threshold (top-25% cut from validation) |
| `risk_tier` | High / Medium / Low (see thresholds below) |
| `top_risk_factors` | 2–3 grounded drivers for this claim |
| `explanation` | LLM plain-English explanation (top 10 populated) |

### Threshold & risk-tier definitions

* **Operating threshold** = the score that flags the top 25% of claims,
  learned on the **validation** split (matches the team's review capacity).
* **Risk tiers** (cut on the validation score distribution):
  * **High** = riskiest **10%**
  * **Medium** = next **15%** (so High+Medium ≈ the 25% the team can review)
  * **Low** = remaining **75%**

## GenAI / LLM notes

> **No live LLM API was used for this submission** (no API key was available in
> the environment). Per the assignment's stated fallback — *"If you do not have
> API access, write the prompts and include two or three manually drafted example
> outputs; we are evaluating your prompt design"* — the graded artifacts here are
> the **prompt design** (`src/llm.py`) and the **example outputs** produced by a
> deterministic, fully-grounded offline generator that fills the exact same prompt
> template. The code path calls a real provider automatically the moment a key is
> present (see below). Example outputs to review: `outputs/top10_explanations.json`
> (prompt + output per claim, plus the low-risk sanity check) and the `explanation`
> column of `outputs/predictions_current_claims.csv`.

* The prompt template and system prompt live in `src/llm.py`; the exact prompt
  used for each claim is saved to `outputs/top10_explanations.json`.
* If `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`) is set, that provider is called;
  otherwise a **grounded offline template generator** runs so the pipeline is
  fully reproducible without network access. In both cases the explanation is
  built only from the claim's real field values and the model's risk drivers.
* `explain.py` also runs the prompt on the **lowest-risk** claim as a sanity
  check — it correctly reports low risk and gives a light completeness nudge
  instead of inventing problems.
