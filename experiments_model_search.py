"""One-off experiment: can a stronger model / better threshold beat logreg?

Compares several models on the validation split (capture@25%, PR-AUC) and
reports TEST-set F1 at two thresholds:
  * capacity threshold (top-25% cut, learned on validation) -- the operating point
  * F1-optimal threshold (learned on validation)            -- best achievable F1
Nothing here is committed to the pipeline; it just answers the question.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src import config
from src.data import add_engineered_features, load_claims, make_xy, split_frames
from src.evaluate import capture_at_topk
from src.model import _preprocessor  # reuse identical preprocessing

warnings.filterwarnings("ignore")


def best_f1_threshold(y_true, y_score):
    """Threshold on the score that maximises F1 (scanned over candidates)."""
    thr_grid = np.unique(y_score)
    best_t, best_f1 = 0.5, -1.0
    for t in thr_grid:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def make(name):
    if name == "logreg":
        est, scale = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=42), True
    elif name == "logreg_C0.3":
        est, scale = LogisticRegression(max_iter=2000, class_weight="balanced",
                                        C=0.3, random_state=42), True
    elif name == "rf_tuned":
        est, scale = RandomForestClassifier(n_estimators=600, min_samples_leaf=8,
                                            max_features="sqrt",
                                            class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=42), False
    elif name == "gbm_tuned":
        est, scale = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                                max_depth=3, subsample=0.9,
                                                random_state=42), False
    elif name == "hist_gbm":
        est, scale = HistGradientBoostingClassifier(learning_rate=0.05,
                                                    max_depth=3, max_iter=400,
                                                    l2_regularization=1.0,
                                                    random_state=42), False
    else:
        raise ValueError(name)
    return Pipeline([("prep", _preprocessor(scale=scale)), ("clf", est)])


def calibrated(name):
    base = make(name)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main():
    df = add_engineered_features(load_claims(config.DEFAULT_HISTORY))
    f = split_frames(df)
    X_tr, y_tr = make_xy(f["train"])
    X_va, y_va = make_xy(f["validation"])
    X_te, y_te = make_xy(f["test"])

    models = ["logreg", "logreg_C0.3", "rf_tuned", "gbm_tuned", "hist_gbm"]
    print(f"{'model':<16} {'val_cap@25':>10} {'val_PRAUC':>10} "
          f"{'te_ROC':>7} {'te_PRAUC':>8} {'te_F1@cap':>10} {'te_F1@best':>11}")
    print("-" * 80)

    rows = []
    for name in models + ["hist_gbm+cal", "logreg+cal"]:
        if name.endswith("+cal"):
            pipe = calibrated(name.replace("+cal", ""))
        else:
            pipe = make(name)
        pipe.fit(X_tr, y_tr)
        va = pipe.predict_proba(X_va)[:, 1]
        te = pipe.predict_proba(X_te)[:, 1]

        val_cap = capture_at_topk(y_va, va, 0.25)
        cap_thr = val_cap["threshold"]
        f1_cap = f1_score(y_te, (te >= cap_thr).astype(int), zero_division=0)
        best_t, _ = best_f1_threshold(y_va, va)  # pick threshold on VAL, apply to TEST
        f1_best = f1_score(y_te, (te >= best_t).astype(int), zero_division=0)

        te_cap25 = capture_at_topk(y_te, te, 0.25)["denial_capture"]
        row = dict(name=name, val_cap=val_cap["denial_capture"],
                   val_prauc=average_precision_score(y_va, va),
                   te_roc=roc_auc_score(y_te, te),
                   te_prauc=average_precision_score(y_te, te),
                   te_cap25=te_cap25, f1_cap=f1_cap, f1_best=f1_best)
        rows.append(row)
        print(f"{name:<16} {row['val_cap']:>10.3f} {row['val_prauc']:>10.3f} "
              f"{row['te_roc']:>7.3f} {row['te_prauc']:>8.3f} "
              f"{f1_cap:>10.3f} {f1_best:>11.3f}")

    print("\nNote: te_cap25 (denial capture @ top25 on TEST) per model:")
    for r in rows:
        print(f"  {r['name']:<16} {r['te_cap25']:.3f}")


if __name__ == "__main__":
    main()
