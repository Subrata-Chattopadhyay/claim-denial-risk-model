"""Tests for the evaluation metrics — especially capture @ top 25%."""
from __future__ import annotations

import numpy as np

from src.evaluate import capture_at_topk, evaluate


def test_capture_perfect_ranking():
    # 10 claims, 3 denials, all ranked at the very top -> reviewing top 30%
    # captures 100% of denials.
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    res = capture_at_topk(y_true, y_score, fraction=0.30)
    assert res["k"] == 3
    assert res["denials_caught"] == 3
    assert res["denial_capture"] == 1.0
    assert res["precision"] == 1.0


def test_capture_worst_ranking():
    # Denials ranked at the bottom -> top 30% captures none.
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    res = capture_at_topk(y_true, y_score, fraction=0.30)
    assert res["denials_caught"] == 0
    assert res["denial_capture"] == 0.0


def test_capture_fraction_maps_to_k():
    y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    y_score = np.linspace(1, 0, 10)
    res = capture_at_topk(y_true, y_score, fraction=0.25)
    # ceil(0.25 * 10) = 3 flagged.
    assert res["k"] == 3


def test_evaluate_bundle_keys_and_ranges():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    # Scores correlated with truth so AUC > 0.5.
    y_score = 0.3 * y_true + rng.random(200) * 0.7
    m = evaluate(y_true, y_score)
    for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "capture_at_top25"]:
        assert key in m
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert 0.0 <= m["pr_auc"] <= 1.0
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)
