"""Data loading, validation, and feature engineering.

All feature engineering uses *only* information available before a claim is
submitted, so the same ``add_engineered_features`` function is applied to both
the historical and the current claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_claims(path) -> pd.DataFrame:
    """Load a claims CSV and coerce the expected dtypes."""
    df = pd.read_csv(path)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with domain-informed engineered features.

    Every feature below is a deterministic transform of pre-submission fields
    (no target, no denial_reason), so it is safe for both train and scoring.
    """
    out = df.copy()

    # Authorization gap: the payer requires prior auth but none is on file.
    out["auth_gap"] = (
        (out["prior_auth_required"] == 1) & (out["has_prior_auth"] == 0)
    ).astype(int)

    # Referral gap: referral required but not present.
    out["referral_gap"] = (
        (out["referral_required"] == 1) & (out["referral_present"] == 0)
    ).astype(int)

    # Contractual payment ratio and the amount the hospital does not expect back.
    denom = out["total_billed"].replace(0, np.nan)
    out["payment_ratio"] = (out["expected_payment"] / denom).fillna(0.0)
    out["unbilled_amount"] = out["total_billed"] - out["expected_payment"]

    # Timely-filing risk: submissions in the slow tail deny noticeably more.
    out["late_submission"] = (out["days_to_submit"] >= 25).astype(int)

    # A single "claim readiness" deficit counter the model (and analysts) can
    # reason about: how many known front-end problems does this claim carry?
    out["readiness_deficits"] = (
        out["auth_gap"]
        + out["referral_gap"]
        + out["missing_documentation_flag"]
        + (out["eligibility_verified"] == 0).astype(int)
        + (out["is_in_network"] == 0).astype(int)
    )

    # Charges have a long right tail; log-scale stabilises linear models.
    out["log_total_billed"] = np.log1p(out["total_billed"].clip(lower=0))

    return out


def feature_columns() -> list[str]:
    """The full ordered list of model input columns after engineering."""
    return (
        config.NUMERIC_COLS
        + config.BINARY_COLS
        + config.ENGINEERED_NUMERIC
        + config.CATEGORICAL_COLS
    )


def split_frames(df: pd.DataFrame):
    """Split the history frame using the provided ``split`` column (as-is)."""
    frames = {}
    for name in ["train", "validation", "test"]:
        frames[name] = df[df["split"] == name].reset_index(drop=True)
    return frames


def make_xy(df: pd.DataFrame):
    """Return (X, y) where X holds only the allowed model input columns."""
    X = df[feature_columns()].copy()
    y = df[config.TARGET].astype(int) if config.TARGET in df.columns else None
    return X, y
