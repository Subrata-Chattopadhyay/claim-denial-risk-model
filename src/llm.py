"""GenAI explanation layer.

Design goals (from the assessment):
  * Grounded ONLY in the claim's real field values and the model's risk drivers.
  * Plain language, one specific recommended action, hedged as a risk estimate.
  * 2-3 sentences.

The prompt template lives in ``PROMPT_TEMPLATE`` and is provider-agnostic.
``generate_explanation`` will call a real LLM if an API key + SDK is available
(OpenAI or Anthropic), otherwise it falls back to a deterministic, fully
grounded template generator so the pipeline is reproducible offline. Either way
the *prompt* is the artifact under evaluation and is written to disk.
"""
from __future__ import annotations

import os
import textwrap

SYSTEM_PROMPT = (
    "You are a claims-denial risk assistant for a hospital's front-end review "
    "team. You help analysts quickly triage claims before submission. You must "
    "ground every statement strictly in the structured claim facts you are "
    "given. Never invent clinical details, dollar amounts, or reasons that are "
    "not in the facts. Always describe the score as a risk estimate, not a "
    "certainty."
)

PROMPT_TEMPLATE = textwrap.dedent(
    """\
    A machine-learning model scored the following claim for denial risk.
    Use ONLY the facts below. Do not add any information that is not present.

    CLAIM FACTS
    - Claim ID: {claim_id}
    - Payer: {payer_id} ({payer_type})
    - Visit type: {visit_type}
    - Total billed: ${total_billed:,.0f}; expected payment: ${expected_payment:,.0f}
    - Days from care to submission: {days_to_submit}
    - Model denial probability: {denial_probability:.0%}  (risk tier: {risk_tier})

    TOP RISK DRIVERS FOR THIS CLAIM (already validated against the data)
    {driver_bullets}

    TASK
    Write a 2-3 sentence explanation for a busy analyst that:
      1. States the risk level in plain language.
      2. Names the main driver(s) above, using no jargon.
      3. Gives ONE specific, concrete recommended action tied to the top driver.
      4. Makes clear this is a risk estimate, not a guaranteed denial.
    Do not use bullet points. Return only the explanation text.
    """
)


def build_prompt(claim: dict, drivers: list[dict]) -> str:
    if drivers:
        driver_bullets = "\n".join(f"    - {d['label']}" for d in drivers)
    else:
        driver_bullets = "    - No major front-end risk flags; score is driven by softer signals."
    return PROMPT_TEMPLATE.format(driver_bullets=driver_bullets, **claim)


# ---------------------------------------------------------------------------
# Provider calls (best-effort; used only when a key is configured)
# ---------------------------------------------------------------------------
def _call_openai(prompt: str) -> str | None:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - network/SDK dependent
        print(f"[llm] OpenAI call failed, using offline fallback: {exc}")
        return None


def _call_anthropic(prompt: str) -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            max_tokens=180,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as exc:  # pragma: no cover
        print(f"[llm] Anthropic call failed, using offline fallback: {exc}")
        return None


# ---------------------------------------------------------------------------
# Deterministic, grounded fallback (no network needed)
# ---------------------------------------------------------------------------
def _offline_explanation(claim: dict, drivers: list[dict]) -> str:
    prob = claim["denial_probability"]
    tier = claim["risk_tier"]
    level = {
        "High": "a high risk of denial",
        "Medium": "a moderate risk of denial",
        "Low": "a low risk of denial",
    }.get(tier, "an elevated risk of denial")

    if drivers:
        top = drivers[0]["label"].lower()
        others = [d["label"].lower() for d in drivers[1:]]
        driver_sentence = f"The main concern is that {top}"
        if others:
            driver_sentence += ", and also " + " and ".join(others)
        driver_sentence += "."
        action = _recommended_action(drivers[0]["key"], claim)
    else:
        driver_sentence = (
            "No single front-end problem stands out; the score reflects a "
            "combination of softer payer and visit signals."
        )
        action = (
            "Do a quick completeness check on authorization, documentation, and "
            "eligibility before submitting."
        )

    return (
        f"This claim ({claim['claim_id']}, payer {claim['payer_id']}) shows "
        f"{level} at about {prob:.0%}. {driver_sentence} Recommended action: "
        f"{action} This is a model risk estimate, not a guaranteed denial."
    )


def _recommended_action(driver_key: str, claim: dict) -> str:
    actions = {
        "auth_gap": "obtain and attach the prior authorization before submitting.",
        "missing_documentation": "gather the missing supporting documents and attach them before submitting.",
        "eligibility_unverified": "verify the patient's coverage with the payer before submitting.",
        "referral_gap": "secure the required referral and add it to the claim.",
        "out_of_network": "confirm the provider's network status or route the claim through the correct payer plan.",
        "late_submission": "expedite this claim now to stay within the payer's timely-filing window.",
        "inpatient_visit": "have a reviewer double-check authorization and documentation for this inpatient stay.",
        "high_denial_payer_type": "apply extra front-end review given this payer's higher denial rate.",
    }
    return actions.get(driver_key, "review the claim's front-end requirements before submitting.")


def generate_explanation(claim: dict, drivers: list[dict]) -> tuple[str, str]:
    """Return (explanation_text, source) where source is the generator used."""
    prompt = build_prompt(claim, drivers)
    for fn, name in ((_call_openai, "openai"), (_call_anthropic, "anthropic")):
        text = fn(prompt)
        if text:
            return text, name
    return _offline_explanation(claim, drivers), "offline_template"
