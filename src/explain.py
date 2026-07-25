"""Generate GenAI explanations for the top-N highest-risk current claims.

Reads the predictions CSV, re-derives each claim's grounded risk drivers, calls
the LLM layer (real API if configured, else the grounded offline template),
writes the explanations back into the predictions CSV, and saves a companion
``top10_explanations.json`` with the exact prompt used for each claim.

Also demonstrates the low-risk sanity check requested by the assessment: it
runs the same prompt on the lowest-risk claim and prints the result.

Example:
    python -m src.explain --predictions outputs/predictions_current_claims.csv \
        --data_path data/current_claims.csv --top_n 10
"""
from __future__ import annotations

import argparse
import json
import pickle

import pandas as pd

from . import config, llm
from .data import add_engineered_features, load_claims


def _claim_context(row: pd.Series, pred_row: pd.Series) -> dict:
    return {
        "claim_id": row["claim_id"],
        "payer_id": row["payer_id"],
        "payer_type": row["payer_type"],
        "visit_type": row["visit_type"],
        "total_billed": float(row["total_billed"]),
        "expected_payment": float(row["expected_payment"]),
        "days_to_submit": int(row["days_to_submit"]),
        "denial_probability": float(pred_row["denial_probability"]),
        "risk_tier": pred_row["risk_tier"],
    }


def run(model_path, predictions_path, data_path, top_n=10):
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)
    drivers = bundle["risk_drivers"]

    preds = pd.read_csv(predictions_path)
    preds["explanation"] = preds["explanation"].astype("object")
    current = add_engineered_features(load_claims(data_path))
    current_by_id = current.set_index("claim_id", drop=False)

    top = preds.head(top_n).copy()
    explanations = []
    source_used = None

    for _, pred_row in top.iterrows():
        claim_id = pred_row["claim_id"]
        eng_row = current_by_id.loc[claim_id]
        claim_ctx = _claim_context(eng_row, pred_row)
        drv = drivers.top_factors_for_row(eng_row, n=3)
        prompt = llm.build_prompt(claim_ctx, drv)
        text, source = llm.generate_explanation(claim_ctx, drv)
        source_used = source
        preds.loc[preds["claim_id"] == claim_id, "explanation"] = text
        explanations.append(
            {
                "claim_id": claim_id,
                "denial_probability": claim_ctx["denial_probability"],
                "risk_tier": claim_ctx["risk_tier"],
                "top_risk_factors": [d["label"] for d in drv],
                "prompt": prompt,
                "explanation": text,
                "source": source,
            }
        )

    preds.to_csv(predictions_path, index=False)

    # Low-risk sanity check: run the same prompt on the lowest-risk claim.
    low_row = preds.iloc[-1]
    low_eng = current_by_id.loc[low_row["claim_id"]]
    low_ctx = _claim_context(low_eng, low_row)
    low_drv = drivers.top_factors_for_row(low_eng, n=3)
    low_prompt = llm.build_prompt(low_ctx, low_drv)
    low_text, _ = llm.generate_explanation(low_ctx, low_drv)
    low_check = {
        "claim_id": low_row["claim_id"],
        "denial_probability": low_ctx["denial_probability"],
        "risk_tier": low_ctx["risk_tier"],
        "top_risk_factors": [d["label"] for d in low_drv],
        "prompt": low_prompt,
        "explanation": low_text,
    }

    payload = {
        "generator": source_used,
        "system_prompt": llm.SYSTEM_PROMPT,
        "prompt_template": llm.PROMPT_TEMPLATE,
        "top_explanations": explanations,
        "low_risk_sanity_check": low_check,
    }
    out_json = config.OUTPUT_DIR / "top10_explanations.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote {len(explanations)} explanations (source={source_used}).")
    print(f"Updated {predictions_path}")
    print(f"Saved prompts + outputs -> {out_json}")
    print("\n--- Example (highest-risk) ---")
    print(explanations[0]["explanation"])
    print("\n--- Low-risk sanity check ---")
    print(low_text)
    return payload


def parse_args():
    p = argparse.ArgumentParser(description="LLM explanations for top-risk claims.")
    p.add_argument("--model_path", default=str(config.DEFAULT_MODEL))
    p.add_argument("--predictions", default=str(config.DEFAULT_PREDICTIONS))
    p.add_argument("--data_path", default=str(config.DEFAULT_CURRENT))
    p.add_argument("--top_n", type=int, default=10)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.model_path, args.predictions, args.data_path, top_n=args.top_n)
