"""Central configuration: column groups, feature-engineering constants, paths.

Kept dependency-free so every other module can import it cheaply.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root, i.e. the parent of ``src/``)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

DEFAULT_HISTORY = DATA_DIR / "claims_history.csv"
DEFAULT_CURRENT = DATA_DIR / "current_claims.csv"
DEFAULT_MODEL = OUTPUT_DIR / "model.pkl"
DEFAULT_METRICS = OUTPUT_DIR / "metrics.json"
DEFAULT_PREDICTIONS = OUTPUT_DIR / "predictions_current_claims.csv"

# ---------------------------------------------------------------------------
# Column semantics (taken from the assessment's column reference)
# ---------------------------------------------------------------------------
TARGET = "is_denied"
ID_COL = "claim_id"

# Columns that must never be fed to the model.
LEAK_COLS = [TARGET, "denial_reason", "split", ID_COL]

# Raw predictors available *before* submission.
CATEGORICAL_COLS = ["payer_id", "payer_type", "visit_type"]
NUMERIC_COLS = [
    "total_billed",
    "expected_payment",
    "num_procedures",
    "num_diagnoses",
    "days_to_submit",
]
BINARY_COLS = [
    "prior_auth_required",
    "has_prior_auth",
    "is_in_network",
    "missing_documentation_flag",
    "eligibility_verified",
    "referral_required",
    "referral_present",
]

# ---------------------------------------------------------------------------
# Engineered features (all derived from pre-submission fields only)
# ---------------------------------------------------------------------------
ENGINEERED_NUMERIC = [
    "auth_gap",            # prior auth required but not on file
    "referral_gap",        # referral required but not present
    "payment_ratio",       # expected_payment / total_billed
    "unbilled_amount",     # total_billed - expected_payment
    "late_submission",     # days_to_submit above timely-filing risk band
    "readiness_deficits",  # count of the known documentation/eligibility gaps
    "log_total_billed",    # damp the long right tail of charges
]

# Operational capacity: the review team can inspect the top 25% of claims.
REVIEW_CAPACITY = 0.25

# Risk-tier cut points expressed as *top fraction of the score distribution*.
# High = riskiest 10%, Medium = next 15% (so High+Medium ~= the 25% the team
# can actually review), Low = everything else. See README for rationale.
RISK_TIER_HIGH_FRAC = 0.10
RISK_TIER_MEDIUM_FRAC = 0.25

SEED = 42
