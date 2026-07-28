"""
tier_logic.py — Layer 4 (rule-based clinical override) + Layer 5 (tier
decision engine).

These two layers are kept in one file because Layer 5's only job is to
combine the ML probability (Layer 2) with the Layer 4 red-flag verdict —
they are tightly coupled by design, per spec.
"""

from __future__ import annotations


def check_red_flags(payload: dict) -> tuple[bool, list[str]]:
    """
    Layer 4. Returns (triggered: bool, reasons: list[str]).

    Rules:
      - Ultrasound PCOD = Yes  -> red flag
      - Irregular cycle AND period_gap > 45 days AND excessive hair growth -> red flag
    """
    reasons = []

    if str(payload.get("pcod_ultrasound", "")).strip().lower() == "yes":
        reasons.append("Ultrasound scan indicated PCOD")

    irregular = str(payload.get("period_cycle", "")).strip().lower() == "irregular"
    gap = payload.get("period_gap_days")
    try:
        gap = float(gap) if gap not in (None, "") else 0
    except (TypeError, ValueError):
        gap = 0
    excessive_hair = str(payload.get("hair_growth_excessive", "")).strip().lower() == "yes"

    if irregular and gap > 45 and excessive_hair:
        reasons.append("Irregular cycles with >45 day gaps combined with excessive hair growth")

    return (len(reasons) > 0, reasons)


TIER_META = {
    1: {"label": "Low Risk"},
    2: {"label": "Borderline - Monitor"},
    3: {"label": "High Risk - Action Needed"},
}


def decide_tier(probability: float, payload: dict) -> dict:
    """
    Layer 5. Combines Layer 2 (probability) + Layer 4 (red flags) into the
    final tier. Also applies the doctor-diagnosed-already exception, which
    bypasses ML entirely.

    Returns {tier, label, reason, red_flag_triggered, red_flag_reasons}
    """
    consulted = str(payload.get("consulted_doctor", "No")).strip().lower() == "yes"
    diagnosed = str(payload.get("diagnosed_condition", "None")).strip().lower()

    if consulted and diagnosed in ("pcos", "pcod"):
        return {
            "tier": 3,
            "label": TIER_META[3]["label"],
            "reason": f"Existing doctor diagnosis on record ({diagnosed.upper()}) — ML tier calculation skipped per protocol.",
            "red_flag_triggered": False,
            "red_flag_reasons": [],
        }

    triggered, reasons = check_red_flags(payload)
    if triggered:
        return {
            "tier": 3,
            "label": TIER_META[3]["label"],
            "reason": "Clinical red-flag rule triggered, overriding ML probability: " + "; ".join(reasons),
            "red_flag_triggered": True,
            "red_flag_reasons": reasons,
        }

    if probability < 0.35:
        tier = 1
    elif probability < 0.65:
        tier = 2
    else:
        tier = 3

    return {
        "tier": tier,
        "label": TIER_META[tier]["label"],
        "reason": f"Based on model probability {probability:.2f}",
        "red_flag_triggered": False,
        "red_flag_reasons": [],
    }


# ---- Laboratory test suggestions for users who have NOT consulted a doctor ----

LAB_TEST_SUGGESTIONS_BY_TIER = {
    1: [
        "Basic hormonal panel (FSH, LH) at your next routine checkup — optional at this stage",
        "Fasting blood glucose as a general wellness baseline",
    ],
    2: [
        "LH and FSH levels (with LH:FSH ratio)",
        "Thyroid panel (TSH)",
        "Fasting insulin and fasting glucose (screens for insulin resistance)",
        "Pelvic/transvaginal ultrasound to check ovarian follicle count and volume",
    ],
    3: [
        "Pelvic/transvaginal ultrasound (ovarian volume + antral follicle count)",
        "LH, FSH, and LH:FSH ratio",
        "Free and total testosterone",
        "AMH (Anti-Müllerian Hormone)",
        "TSH and Prolactin (to rule out thyroid/pituitary causes of symptoms)",
        "Fasting insulin, fasting glucose, and lipid profile",
    ],
}


def lab_suggestions_for(tier: int) -> list[str]:
    return LAB_TEST_SUGGESTIONS_BY_TIER.get(tier, LAB_TEST_SUGGESTIONS_BY_TIER[2])
