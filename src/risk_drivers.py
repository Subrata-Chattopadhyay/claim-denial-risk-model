"""Transparent, per-claim risk drivers.

Rather than expose opaque model internals to analysts, we define a small set of
**interpretable risk flags** (each a plain-English condition on real field
values) and learn how much each flag lifts the denial rate above the base rate
from the training data. For any claim we surface the active flags with the
highest learned lift as its ``top_risk_factors``.

This keeps the analyst-facing explanation grounded strictly in the claim's own
field values, which is exactly what the GenAI step needs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Each driver: key -> (predicate over an engineered-feature row, analyst label).
# Predicates assume ``add_engineered_features`` has already been applied.
DRIVER_DEFS = {
    "auth_gap": (
        lambda r: r["auth_gap"] == 1,
        "Prior authorization required but not on file",
    ),
    "missing_documentation": (
        lambda r: r["missing_documentation_flag"] == 1,
        "Required supporting documentation appears to be missing",
    ),
    "eligibility_unverified": (
        lambda r: r["eligibility_verified"] == 0,
        "Patient eligibility was not verified before the visit",
    ),
    "referral_gap": (
        lambda r: r["referral_gap"] == 1,
        "Referral required but not present",
    ),
    "out_of_network": (
        lambda r: r["is_in_network"] == 0,
        "Provider is out of network for this payer",
    ),
    "late_submission": (
        lambda r: r["late_submission"] == 1,
        "Claim is being submitted late (timely-filing risk)",
    ),
    "inpatient_visit": (
        lambda r: r.get("visit_type") == "Inpatient",
        "Inpatient visit (historically higher denial rate)",
    ),
    "high_denial_payer_type": (
        lambda r: r.get("payer_type") in ("Medicaid MCO", "Medicare Advantage"),
        "Payer type with an above-average denial rate",
    ),
}


class RiskDrivers:
    """Learns per-flag denial-rate lift from training data."""

    def __init__(self) -> None:
        self.lift_: dict[str, float] = {}
        self.rate_: dict[str, float] = {}
        self.base_rate_: float = 0.0

    def fit(self, df: pd.DataFrame, target: str = "is_denied") -> "RiskDrivers":
        self.base_rate_ = float(df[target].mean())
        for key, (pred, _label) in DRIVER_DEFS.items():
            mask = df.apply(pred, axis=1)
            if mask.sum() == 0:
                self.rate_[key] = self.base_rate_
                self.lift_[key] = 1.0
                continue
            rate = float(df.loc[mask, target].mean())
            self.rate_[key] = rate
            self.lift_[key] = rate / self.base_rate_ if self.base_rate_ else 1.0
        return self

    def top_factors_for_row(self, row: pd.Series, n: int = 3) -> list[dict]:
        """Return up to ``n`` active drivers for a claim, highest lift first."""
        active = []
        for key, (pred, label) in DRIVER_DEFS.items():
            try:
                on = bool(pred(row))
            except Exception:
                on = False
            if on:
                active.append(
                    {
                        "key": key,
                        "label": label,
                        "lift": round(self.lift_.get(key, 1.0), 2),
                        "denial_rate": round(self.rate_.get(key, 0.0), 3),
                    }
                )
        active.sort(key=lambda d: d["lift"], reverse=True)
        return active[:n]

    def global_table(self) -> pd.DataFrame:
        rows = [
            {
                "driver": key,
                "label": DRIVER_DEFS[key][1],
                "denial_rate": round(self.rate_.get(key, 0.0), 3),
                "lift_vs_base": round(self.lift_.get(key, 1.0), 2),
            }
            for key in DRIVER_DEFS
        ]
        df = pd.DataFrame(rows).sort_values("lift_vs_base", ascending=False)
        return df.reset_index(drop=True)


def format_factors(factors: list[dict]) -> str:
    """Compact human-readable string for the predictions CSV."""
    return "; ".join(f["label"] for f in factors) if factors else "No major risk flags"
