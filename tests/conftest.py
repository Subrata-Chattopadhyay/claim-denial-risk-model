"""Shared pytest fixtures.

A session-scoped fixture trains the model once (logreg only, for speed) so the
prediction / explanation tests can reuse the artifact.

An autouse fixture redirects all training artifacts to a temporary directory so
running the test suite never overwrites the committed files in ``outputs/``.
"""
from __future__ import annotations

import pickle

import pandas as pd
import pytest

from src import config
from src.data import add_engineered_features, load_claims
from src.train import train


@pytest.fixture(scope="session", autouse=True)
def _isolate_outputs(tmp_path_factory):
    """Point all generated artifacts at a temp dir (protects committed outputs/)."""
    tmp = tmp_path_factory.mktemp("artifacts")
    config.OUTPUT_DIR = tmp
    config.PLOTS_DIR = tmp / "plots"
    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    config.DEFAULT_MODEL = tmp / "model.pkl"
    config.DEFAULT_METRICS = tmp / "metrics.json"
    config.DEFAULT_PREDICTIONS = tmp / "predictions_current_claims.csv"
    yield


@pytest.fixture(scope="session")
def history_df() -> pd.DataFrame:
    return load_claims(config.DEFAULT_HISTORY)


@pytest.fixture(scope="session")
def current_df() -> pd.DataFrame:
    return load_claims(config.DEFAULT_CURRENT)


@pytest.fixture(scope="session")
def engineered_history(history_df) -> pd.DataFrame:
    return add_engineered_features(history_df)


@pytest.fixture(scope="session")
def trained_bundle(_isolate_outputs):
    """Train once (logreg) and return the persisted model bundle."""
    train(config.DEFAULT_HISTORY, models=["logreg"], seed=config.SEED)
    with open(config.DEFAULT_MODEL, "rb") as fh:
        return pickle.load(fh)
