"""Model factory: a preprocessing + estimator pipeline per model name.

Keeping the preprocessing *inside* the sklearn Pipeline guarantees the exact
same transforms are applied at train, evaluate, and score time.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import feature_columns

# Numeric columns fed to the model = raw numerics + engineered numerics + binaries.
_NUMERIC = config.NUMERIC_COLS + config.BINARY_COLS + config.ENGINEERED_NUMERIC
_CATEGORICAL = config.CATEGORICAL_COLS


def _preprocessor(scale: bool) -> ColumnTransformer:
    """One-hot encode categoricals; optionally scale numerics (for linear models)."""
    numeric_steps = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_steps, _NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=1),
                _CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def build_model(name: str) -> Pipeline:
    """Return an untrained pipeline for the requested model name."""
    name = name.lower()
    if name == "logreg":
        est = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            C=1.0,
            random_state=config.SEED,
        )
        scale = True
    elif name in ("rf", "randomforest"):
        est = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=config.SEED,
        )
        scale = False
    elif name in ("gbm", "gradientboosting"):
        est = GradientBoostingClassifier(random_state=config.SEED)
        scale = False
    else:
        raise ValueError(f"Unknown model '{name}'. Use logreg | rf | gbm.")

    return Pipeline(
        steps=[
            ("preprocess", _preprocessor(scale=scale)),
            ("classifier", est),
        ]
    )


AVAILABLE_MODELS = ["logreg", "rf", "gbm"]


def input_columns() -> list[str]:
    """Ordered list of columns the pipeline consumes (for reference/tests)."""
    return feature_columns()
