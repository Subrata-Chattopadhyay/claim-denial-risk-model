"""Exploratory data analysis for the write-up.

Generates the plots and summary tables referenced by the PDF write-up:
  * denial rate by key front-end risk flags
  * denial rate by payer type and visit type
  * a global risk-driver lift table

Run:  python -m src.eda
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import add_engineered_features, load_claims, split_frames
from .risk_drivers import RiskDrivers


def _bar(ax, labels, values, title, ylabel, baseline=None):
    bars = ax.bar(labels, values, color="#4C72B0")
    if baseline is not None:
        ax.axhline(baseline, color="red", ls="--", label=f"base rate {baseline:.2f}")
        ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", fontsize=8)


def run(data_path=config.DEFAULT_HISTORY):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_claims(data_path)
    eng = add_engineered_features(raw)
    base = eng["is_denied"].mean()

    # 1) Front-end risk flags
    flags = {
        "Auth gap": eng["auth_gap"] == 1,
        "Missing docs": eng["missing_documentation_flag"] == 1,
        "Eligibility\nunverified": eng["eligibility_verified"] == 0,
        "Referral gap": eng["referral_gap"] == 1,
        "Out of network": eng["is_in_network"] == 0,
        "Late submit": eng["late_submission"] == 1,
    }
    labels = list(flags.keys())
    rates = [eng.loc[m, "is_denied"].mean() for m in flags.values()]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar(ax, labels, rates, "Denial rate when a front-end flag is present", "Denial rate", base)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_risk_flags.png", dpi=130)
    plt.close(fig)

    # 2) Payer type and visit type
    pt = eng.groupby("payer_type")["is_denied"].mean().sort_values(ascending=False)
    vt = eng.groupby("visit_type")["is_denied"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    _bar(ax[0], list(pt.index), list(pt.values), "Denial rate by payer type", "Denial rate", base)
    _bar(ax[1], list(vt.index), list(vt.values), "Denial rate by visit type", "Denial rate", base)
    fig.tight_layout()
    fig.savefig(config.PLOTS_DIR / "eda_payer_visit.png", dpi=130)
    plt.close(fig)

    # 3) Global driver lift table (fit on train split only)
    frames = split_frames(eng)
    drivers = RiskDrivers().fit(frames["train"])
    table = drivers.global_table()
    table.to_csv(config.OUTPUT_DIR / "risk_driver_table.csv", index=False)

    print(f"Base denial rate: {base:.3f}")
    print("\nGlobal risk-driver lift (train split):")
    print(table.to_string(index=False))
    print(f"\nSaved EDA plots to {config.PLOTS_DIR}")


if __name__ == "__main__":
    run()
