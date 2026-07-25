"""Tests for model building, risk drivers, scoring schema, and reproducibility."""
from __future__ import annotations

import numpy as np

from src import config
from src.data import add_engineered_features, make_xy, split_frames
from src.model import build_model
from src.predict import assign_tier, score_claims
from src.risk_drivers import RiskDrivers
from src.train import train


def test_model_trains_and_outputs_valid_probabilities(engineered_history):
    frames = split_frames(engineered_history)
    X_tr, y_tr = make_xy(frames["train"])
    X_va, _ = make_xy(frames["validation"])
    pipe = build_model("logreg")
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_va)[:, 1]
    assert proba.shape[0] == len(X_va)
    assert np.all((proba >= 0.0) & (proba <= 1.0))


def test_risk_drivers_learn_lift(engineered_history):
    frames = split_frames(engineered_history)
    drivers = RiskDrivers().fit(frames["train"])
    # Auth gap is the strongest known driver -> lift should exceed 1.
    assert drivers.lift_["auth_gap"] > 1.0
    table = drivers.global_table()
    assert (table["lift_vs_base"] >= 0).all()


def test_top_factors_are_grounded_in_row(engineered_history):
    frames = split_frames(engineered_history)
    drivers = RiskDrivers().fit(frames["train"])
    # Build a row with a known auth gap.
    row = frames["train"].iloc[0].copy()
    row["prior_auth_required"] = 1
    row["has_prior_auth"] = 0
    row["auth_gap"] = 1
    factors = drivers.top_factors_for_row(row, n=3)
    keys = {f["key"] for f in factors}
    assert "auth_gap" in keys
    assert len(factors) <= 3


def test_assign_tier_thresholds():
    thr = {"high": 0.7, "medium": 0.5}
    assert assign_tier(0.9, thr) == "High"
    assert assign_tier(0.6, thr) == "Medium"
    assert assign_tier(0.1, thr) == "Low"


def test_prediction_schema_and_sorting(trained_bundle, current_df):
    preds = score_claims(trained_bundle, current_df)
    expected_cols = [
        "claim_id",
        "denial_probability",
        "predicted_denial",
        "risk_tier",
        "top_risk_factors",
        "explanation",
    ]
    assert list(preds.columns) == expected_cols
    # One row per current claim.
    assert len(preds) == len(current_df)
    # Sorted highest -> lowest probability.
    probs = preds["denial_probability"].to_numpy()
    assert np.all(np.diff(probs) <= 1e-12)
    # Probabilities in range; predicted_denial binary; tiers valid.
    assert preds["denial_probability"].between(0, 1).all()
    assert set(preds["predicted_denial"].unique()).issubset({0, 1})
    assert set(preds["risk_tier"].unique()).issubset({"High", "Medium", "Low"})


def test_predicted_denial_matches_operating_threshold(trained_bundle, current_df):
    preds = score_claims(trained_bundle, current_df)
    thr = trained_bundle["operating_threshold"]
    expected = (preds["denial_probability"] >= thr).astype(int)
    assert (preds["predicted_denial"] == expected).all()


def test_training_is_reproducible():
    m1 = train(config.DEFAULT_HISTORY, models=["logreg"], seed=config.SEED)
    m2 = train(config.DEFAULT_HISTORY, models=["logreg"], seed=config.SEED)
    assert abs(m1["test"]["pr_auc"] - m2["test"]["pr_auc"]) < 1e-9
    assert abs(m1["test"]["roc_auc"] - m2["test"]["roc_auc"]) < 1e-9
