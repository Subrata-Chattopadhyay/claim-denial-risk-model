"""Tests for data loading, feature engineering, and leakage prevention."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.data import add_engineered_features, feature_columns, make_xy, split_frames


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": ["A", "B", "C"],
            "payer_id": ["P001", "P002", "P003"],
            "payer_type": ["Commercial", "Medicaid MCO", "BCBS"],
            "visit_type": ["Outpatient", "Inpatient", "Emergency"],
            "total_billed": [1000.0, 0.0, 5000.0],
            "expected_payment": [800.0, 0.0, 2500.0],
            "num_procedures": [1, 2, 3],
            "num_diagnoses": [1, 2, 3],
            "prior_auth_required": [1, 1, 0],
            "has_prior_auth": [0, 1, 0],
            "is_in_network": [1, 0, 1],
            "days_to_submit": [5, 30, 25],
            "missing_documentation_flag": [0, 1, 0],
            "eligibility_verified": [1, 0, 1],
            "referral_required": [1, 0, 1],
            "referral_present": [0, 0, 1],
        }
    )


def test_auth_and_referral_gap_logic():
    eng = add_engineered_features(_toy_frame())
    # Row A: auth required, not on file -> auth_gap=1; referral required, absent -> 1
    assert eng.loc[0, "auth_gap"] == 1
    assert eng.loc[0, "referral_gap"] == 1
    # Row B: auth required AND on file -> auth_gap=0
    assert eng.loc[1, "auth_gap"] == 0
    # Row C: no referral gap (present) and no auth requirement
    assert eng.loc[2, "auth_gap"] == 0
    assert eng.loc[2, "referral_gap"] == 0


def test_payment_ratio_handles_zero_billed():
    eng = add_engineered_features(_toy_frame())
    # Row B has total_billed == 0 -> ratio must be 0, not NaN/inf.
    assert eng.loc[1, "payment_ratio"] == 0.0
    assert np.isfinite(eng["payment_ratio"]).all()
    # Row A: 800/1000
    assert abs(eng.loc[0, "payment_ratio"] - 0.8) < 1e-9


def test_late_submission_and_readiness_deficits():
    eng = add_engineered_features(_toy_frame())
    assert eng.loc[0, "late_submission"] == 0  # 5 days
    assert eng.loc[1, "late_submission"] == 1  # 30 days
    # Row B deficits: auth_gap(0)+referral_gap(0)+missing_doc(1)+elig_unverified(1)+oon(1)=3
    assert eng.loc[1, "readiness_deficits"] == 3


def test_engineered_columns_present_and_no_nan(engineered_history):
    for col in config.ENGINEERED_NUMERIC:
        assert col in engineered_history.columns
    assert engineered_history[config.ENGINEERED_NUMERIC].isnull().sum().sum() == 0


def test_no_leakage_in_feature_columns():
    cols = feature_columns()
    for leak in config.LEAK_COLS:
        assert leak not in cols, f"leak column {leak} must not be a model input"


def test_make_xy_excludes_target_and_id(engineered_history):
    X, y = make_xy(engineered_history)
    assert config.TARGET not in X.columns
    assert config.ID_COL not in X.columns
    assert "denial_reason" not in X.columns
    assert set(y.unique()).issubset({0, 1})


def test_split_frames_uses_provided_split(history_df):
    frames = split_frames(add_engineered_features(history_df))
    assert set(frames) == {"train", "validation", "test"}
    total = sum(len(f) for f in frames.values())
    assert total == len(history_df)  # no rows lost, splits are as-is


def test_current_has_same_engineered_columns(current_df, engineered_history):
    eng_cur = add_engineered_features(current_df)
    for col in config.ENGINEERED_NUMERIC:
        assert col in eng_cur.columns
    # Current data must not contain leakage columns at all.
    for leak in [config.TARGET, "denial_reason", "split"]:
        assert leak not in current_df.columns
