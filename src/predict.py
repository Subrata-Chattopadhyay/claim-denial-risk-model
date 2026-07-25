"""Score current_claims.csv with the trained model and write predictions.

Produces ``predictions_current_claims.csv`` sorted highest -> lowest denial
probability, with the columns required by the assessment:
claim_id, denial_probability, predicted_denial, risk_tier, top_risk_factors,
explanation (explanation filled in by ``explain.py``).

Example:
    python -m src.predict --model_path outputs/model.pkl \
        --data_path data/current_claims.csv \
        --output outputs/predictions_current_claims.csv
"""
from __future__ import annotations

import argparse
import pickle

import pandas as pd

from . import config
from .data import add_engineered_features, load_claims
from .risk_drivers import format_factors


def load_bundle(model_path):
    with open(model_path, "rb") as fh:
        return pickle.load(fh)


def assign_tier(prob: float, thresholds: dict) -> str:
    if prob >= thresholds["high"]:
        return "High"
    if prob >= thresholds["medium"]:
        return "Medium"
    return "Low"


def score_claims(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Return a predictions frame (unsorted explanation column left blank)."""
    eng = add_engineered_features(df)
    X = eng[bundle["feature_columns"]]
    probs = bundle["pipeline"].predict_proba(X)[:, 1]

    drivers = bundle["risk_drivers"]
    thresholds = bundle["tier_thresholds"]
    op_threshold = bundle["operating_threshold"]

    records = []
    for i, (_, row) in enumerate(eng.iterrows()):
        prob = float(probs[i])
        factors = drivers.top_factors_for_row(row, n=3)
        records.append(
            {
                "claim_id": row["claim_id"],
                "denial_probability": round(prob, 4),
                "predicted_denial": int(prob >= op_threshold),
                "risk_tier": assign_tier(prob, thresholds),
                "top_risk_factors": format_factors(factors),
                "explanation": "",
            }
        )
    out = pd.DataFrame(records).sort_values(
        "denial_probability", ascending=False
    ).reset_index(drop=True)
    return out


def run(model_path, data_path, output):
    bundle = load_bundle(model_path)
    df = load_claims(data_path)
    preds = score_claims(bundle, df)
    preds.to_csv(output, index=False)
    n_high = (preds["risk_tier"] == "High").sum()
    n_med = (preds["risk_tier"] == "Medium").sum()
    print(
        f"Scored {len(preds)} claims -> {output}\n"
        f"High={n_high}  Medium={n_med}  Low={len(preds)-n_high-n_med}"
    )
    return preds


def parse_args():
    p = argparse.ArgumentParser(description="Score current claims.")
    p.add_argument("--model_path", default=str(config.DEFAULT_MODEL))
    p.add_argument("--data_path", default=str(config.DEFAULT_CURRENT))
    p.add_argument("--output", default=str(config.DEFAULT_PREDICTIONS))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model_path, args.data_path, args.output)
