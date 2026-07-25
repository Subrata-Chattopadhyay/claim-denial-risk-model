"""Evaluation metrics and plots.

The review team can only inspect the top 25% of claims by risk score, so the
head-line metric is **denial capture at the top 25%** (recall among the claims
we would actually flag). We also report threshold-independent ranking metrics
(ROC-AUC, PR-AUC / average precision) and precision at the review cut.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from . import config


def capture_at_topk(y_true, y_score, fraction: float) -> dict:
    """Metrics when we flag the riskiest ``fraction`` of claims.

    Returns the score threshold at that cut, plus precision, recall (denial
    capture) and the share of all denials caught.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_score)
    k = max(1, int(np.ceil(fraction * n)))
    order = np.argsort(-y_score)
    flagged = np.zeros(n, dtype=int)
    flagged[order[:k]] = 1
    threshold = float(np.sort(y_score)[::-1][k - 1])

    total_denials = int(y_true.sum())
    caught = int(y_true[flagged == 1].sum())
    return {
        "fraction": fraction,
        "k": k,
        "threshold": threshold,
        "precision": precision_score(y_true, flagged, zero_division=0),
        "recall": recall_score(y_true, flagged, zero_division=0),
        "denials_caught": caught,
        "total_denials": total_denials,
        "denial_capture": caught / total_denials if total_denials else 0.0,
    }


def evaluate(y_true, y_score, threshold: float | None = None) -> dict:
    """Full metric bundle for a set of predictions."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    top25 = capture_at_topk(y_true, y_score, config.REVIEW_CAPACITY)
    thr = threshold if threshold is not None else top25["threshold"]
    y_pred = (y_score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "base_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "capture_at_top25": top25,
    }


def save_metrics(metrics: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _ensure_plotdir():
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_roc_pr(y_true, y_score, prefix: str = "test") -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_plotdir()
    paths = []

    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="grey")
    ax[0].set_xlabel("False positive rate")
    ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC curve")
    ax[0].legend(loc="lower right")

    ax[1].plot(rec, prec, label=f"PR (AP={ap:.3f})")
    ax[1].axhline(np.mean(y_true), ls="--", color="grey", label="base rate")
    ax[1].set_xlabel("Recall (denial capture)")
    ax[1].set_ylabel("Precision")
    ax[1].set_title("Precision-Recall curve")
    ax[1].legend(loc="upper right")
    fig.tight_layout()
    p = config.PLOTS_DIR / f"{prefix}_roc_pr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(str(p))
    return paths


def plot_capture_curve(y_true, y_score, prefix: str = "test") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_plotdir()
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    total = y_sorted.sum()
    fracs = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    capture = np.cumsum(y_sorted) / total

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fracs, capture, label="model")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="random")
    ax.axvline(config.REVIEW_CAPACITY, color="red", ls=":", label="25% review capacity")
    idx = int(np.ceil(config.REVIEW_CAPACITY * len(y_sorted))) - 1
    ax.scatter([fracs[idx]], [capture[idx]], color="red", zorder=5)
    ax.annotate(
        f"{capture[idx]*100:.0f}% of denials\ncaught in top 25%",
        (fracs[idx], capture[idx]),
        textcoords="offset points",
        xytext=(10, -30),
    )
    ax.set_xlabel("Fraction of claims reviewed (highest risk first)")
    ax.set_ylabel("Fraction of denials caught")
    ax.set_title("Denial-capture (gains) curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = config.PLOTS_DIR / f"{prefix}_capture_curve.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return str(p)
