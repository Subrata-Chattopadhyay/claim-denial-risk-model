"""Train and compare denial-risk models; persist the best pipeline.

Model selection is driven by the operational constraint: the review team can
only inspect the top 25% of claims, so we select on **validation denial capture
at the top 25%** (tie-broken by PR-AUC). The winning pipeline is bundled with
its fitted risk-driver table and risk-tier thresholds and written to disk.

Example:
    python -m src.train --data_path data/claims_history.csv \
        --models logreg rf gbm --seed 42
"""
from __future__ import annotations

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from . import config, evaluate as ev
from .data import add_engineered_features, load_claims, make_xy, split_frames
from .model import AVAILABLE_MODELS, build_model
from .risk_drivers import RiskDrivers


def compute_tier_thresholds(scores: np.ndarray) -> dict:
    """Score cut points for High/Medium tiers from a reference distribution."""
    high = float(np.quantile(scores, 1 - config.RISK_TIER_HIGH_FRAC))
    medium = float(np.quantile(scores, 1 - config.RISK_TIER_MEDIUM_FRAC))
    return {"high": high, "medium": medium}


def rule_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    """A transparent non-ML baseline: count of front-end readiness deficits."""
    eng = add_engineered_features(df)
    return eng["readiness_deficits"].to_numpy(dtype=float)


def train(data_path, models, seed=config.SEED) -> dict:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    raw = load_claims(data_path)
    eng = add_engineered_features(raw)
    frames = split_frames(eng)
    X_tr, y_tr = make_xy(frames["train"])
    X_va, y_va = make_xy(frames["validation"])
    X_te, y_te = make_xy(frames["test"])

    results = {}

    # Transparent rule baseline (validation) for comparison.
    base_val = ev.capture_at_topk(
        y_va, rule_baseline_scores(frames["validation"]), config.REVIEW_CAPACITY
    )
    results["rule_baseline"] = {
        "val_denial_capture_top25": base_val["denial_capture"],
        "val_precision_top25": base_val["precision"],
    }

    trained = {}
    leaderboard = []
    for name in models:
        pipe = build_model(name)
        pipe.fit(X_tr, y_tr)
        val_scores = pipe.predict_proba(X_va)[:, 1]
        val_metrics = ev.evaluate(y_va, val_scores)
        trained[name] = pipe
        row = {
            "model": name,
            "val_denial_capture_top25": val_metrics["capture_at_top25"]["denial_capture"],
            "val_precision_top25": val_metrics["capture_at_top25"]["precision"],
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        leaderboard.append(row)
        print(
            f"[{name}] val capture@25%={row['val_denial_capture_top25']:.3f} "
            f"precision@25%={row['val_precision_top25']:.3f} "
            f"PR-AUC={row['val_pr_auc']:.3f} ROC-AUC={row['val_roc_auc']:.3f}"
        )

    # Select best: denial capture @25%, tie-break PR-AUC.
    best = max(
        leaderboard,
        key=lambda r: (r["val_denial_capture_top25"], r["val_pr_auc"]),
    )
    best_name = best["model"]
    best_pipe = trained[best_name]
    print(f"\nSelected model: {best_name}")

    # Refit thresholds on validation scores; final unbiased read on TEST.
    val_scores = best_pipe.predict_proba(X_va)[:, 1]
    tiers = compute_tier_thresholds(val_scores)
    test_scores = best_pipe.predict_proba(X_te)[:, 1]
    # Operating threshold = the top-25% cut learned on validation.
    op_threshold = ev.capture_at_topk(y_va, val_scores, config.REVIEW_CAPACITY)["threshold"]
    test_metrics = ev.evaluate(y_te, test_scores, threshold=op_threshold)

    # Fit interpretable risk drivers on the TRAIN split only.
    drivers = RiskDrivers().fit(frames["train"])

    # Plots on the test split.
    plot_paths = ev.plot_roc_pr(y_te, test_scores, prefix="test")
    plot_paths.append(ev.plot_capture_curve(y_te, test_scores, prefix="test"))

    # Persist the bundle.
    bundle = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "risk_drivers": drivers,
        "tier_thresholds": tiers,
        "operating_threshold": float(op_threshold),
        "feature_columns": list(X_tr.columns),
        "seed": seed,
    }
    with open(config.DEFAULT_MODEL, "wb") as fh:
        pickle.dump(bundle, fh)

    metrics = {
        "selected_model": best_name,
        "leaderboard": leaderboard,
        "rule_baseline": results["rule_baseline"],
        "tier_thresholds": tiers,
        "operating_threshold": float(op_threshold),
        "test": test_metrics,
        "plots": plot_paths,
    }
    ev.save_metrics(metrics, config.DEFAULT_METRICS)

    print("\n=== TEST metrics (selected model) ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved model -> {config.DEFAULT_MODEL}")
    print(f"Saved metrics -> {config.DEFAULT_METRICS}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train denial-risk classifier.")
    p.add_argument("--data_path", default=str(config.DEFAULT_HISTORY))
    p.add_argument(
        "--models",
        nargs="+",
        default=AVAILABLE_MODELS,
        help="Subset of: logreg rf gbm",
    )
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data_path, args.models, seed=args.seed)
