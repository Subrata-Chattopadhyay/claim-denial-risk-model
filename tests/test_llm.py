"""Tests for the GenAI layer: prompt construction and grounded generation.

These assert the offline generator's key safety properties: it is grounded in
the given fields, includes a recommended action, hedges as a risk estimate, and
behaves sensibly on a low-risk claim (no invented problems).
"""
from __future__ import annotations

from src import llm


def _claim(prob=0.94, tier="High"):
    return {
        "claim_id": "CCLM-TEST1",
        "payer_id": "P009",
        "payer_type": "Medicaid MCO",
        "visit_type": "Inpatient",
        "total_billed": 12000.0,
        "expected_payment": 6000.0,
        "days_to_submit": 30,
        "denial_probability": prob,
        "risk_tier": tier,
    }


HIGH_DRIVERS = [
    {"key": "missing_documentation", "label": "Required supporting documentation appears to be missing", "lift": 1.9},
    {"key": "auth_gap", "label": "Prior authorization required but not on file", "lift": 2.1},
]


def test_prompt_contains_only_given_fields():
    prompt = llm.build_prompt(_claim(), HIGH_DRIVERS)
    assert "CCLM-TEST1" in prompt
    assert "P009" in prompt
    # Each driver label must appear in the prompt.
    for d in HIGH_DRIVERS:
        assert d["label"] in prompt
    # The task instructions must be present.
    assert "risk estimate" in prompt.lower()


def test_offline_explanation_is_grounded_and_hedged():
    text, source = llm.generate_explanation(_claim(), HIGH_DRIVERS)
    assert source == "offline_template"  # no API key in test env
    # Grounded in the claim id and probability.
    assert "CCLM-TEST1" in text
    assert "94%" in text
    # Mentions the top driver.
    assert "documentation" in text.lower()
    # Includes a recommended action and a hedge.
    assert "recommended action" in text.lower()
    assert "risk estimate" in text.lower()
    assert "not a guaranteed denial" in text.lower()
    # Length sanity: 2-3 sentences (allow 2-4 to be safe).
    n_sentences = text.count(". ")
    assert 2 <= n_sentences <= 5


def test_offline_explanation_no_invented_numbers():
    # The generator must not introduce dollar amounts not present in the claim
    # facts. Our template never emits '$', so assert that stays true.
    text, _ = llm.generate_explanation(_claim(), HIGH_DRIVERS)
    assert "$" not in text


def test_low_risk_claim_behaves_sensibly():
    claim = _claim(prob=0.09, tier="Low")
    text, _ = llm.generate_explanation(claim, drivers=[])
    assert "low risk" in text.lower()
    # Should not fabricate a specific denial reason.
    assert "risk estimate" in text.lower()
    # Still offers a light, sensible next step.
    assert "check" in text.lower() or "review" in text.lower()


def test_recommended_action_maps_to_top_driver():
    # Auth gap as the top (highest-lift) driver -> action should mention auth.
    drivers = [
        {"key": "auth_gap", "label": "Prior authorization required but not on file", "lift": 2.1},
    ]
    text, _ = llm.generate_explanation(_claim(), drivers)
    assert "authorization" in text.lower()
