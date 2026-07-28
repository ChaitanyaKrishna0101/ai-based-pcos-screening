"""
llm_service.py — Layer 6 (LLM reasoning) + Layer 7 (personalization).

Calls Groq (llama-3.3-70b-versatile) to turn tier + SHAP factors +
medication/diagnosis history into structured, safe lifestyle guidance.

Safety rules enforced in the system prompt AND re-validated after the
response (defense in depth):
  - Never diagnoses.
  - Never gives specific medication dosages.
  - Always tells Tier 2 / Tier 3 users to see a professional.
  - Ayurvedic/natural tips are explicitly marked as supportive only.

If GROQ_API_KEY is missing or the API call fails, a deterministic
template-based fallback is used so the app never breaks in a demo or
in an offline HF Space without a key configured. Failures are logged
server-side, not surfaced to the user — the user only ever sees
normal-looking guidance text.
"""

from __future__ import annotations
import os
import json
import re
import logging

try:
    from groq import Groq
    HAS_GROQ_SDK = True
except ImportError:
    HAS_GROQ_SDK = False

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"

BANNED_DOSAGE_PATTERN = re.compile(
    r"\b\d+\s?(mg|mcg|g|ml|iu)\b", re.IGNORECASE
)


def _client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not HAS_GROQ_SDK:
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        logger.warning("Failed to initialize Groq client: %s", e)
        return None


def _build_system_prompt() -> str:
    return (
        "You are a lifestyle guidance assistant embedded in a PCOS SCREENING tool "
        "(not a diagnostic tool). You are NOT a doctor and must NEVER state or imply "
        "a medical diagnosis. Follow these rules strictly:\n"
        "1. Never diagnose PCOS/PCOD. Only discuss 'risk indicators' and 'screening results'.\n"
        "2. Output MUST be valid JSON only, no markdown fences, matching exactly this shape:\n"
        '   {"summary": str, "lifestyle": [str,...2-3 items], "ayurvedic": [str,...1-2 items], '
        '"avoid": [str,...1-2 items], "doctor_action": str, "mental_wellbeing": [str,...1-2 items]}\n'
        "3. Keep total content under 200 words.\n"
        "4. Mark every ayurvedic/natural suggestion as supportive only, never a replacement for treatment.\n"
        "5. For Tier 2 and Tier 3, doctor_action MUST clearly recommend professional consultation.\n"
        "6. NEVER include specific medication names with dosages (no 'mg', 'mcg', numeric doses). "
        "You may mention general categories only if the user already reports being on medication, "
        "and only to say advice is compatible with it — never suggest starting/stopping/changing dose.\n"
        "7. Reference the given top contributing factors in plain, non-alarming language.\n"
        "8. If the user is already on a doctor-prescribed medication or has a formal diagnosis, "
        "your advice must be medication-aware and must NOT contradict their existing treatment plan; "
        "explicitly say to keep following their doctor's plan.\n"
    )


def _build_user_prompt(tier: int, label: str, probability: float, top_factors: list[dict],
                        personalization: dict) -> str:
    factors_text = "; ".join(
        f"{f['label']} ({'raises' if f['direction']=='increases_risk' else 'lowers'} risk)"
        for f in top_factors
    ) or "no dominant single factor"

    consulted = personalization.get("consulted_doctor", False)
    diagnosis = personalization.get("diagnosed_condition", "None")
    medication = personalization.get("medication", "None")

    context = (
        f"Tier: {tier} ({label})\n"
        f"Model probability: {probability:.2f}\n"
        f"Top contributing factors for this specific person: {factors_text}\n"
        f"Previously consulted a doctor: {'Yes' if consulted else 'No'}\n"
        f"Existing diagnosis on file: {diagnosis}\n"
        f"Current medication on file: {medication}\n"
    )
    if not consulted:
        context += (
            "This user has NOT consulted a doctor yet. In doctor_action, besides "
            "recommending consultation appropriate to their tier, mention that lab/ultrasound "
            "screening tests would help clarify the picture (do not name exact panels, the app "
            "shows those separately).\n"
        )
    return context + "\nGenerate the JSON now."


def _describe_factors(top_factors: list[dict], max_items: int = 2) -> tuple[str, str, bool, bool]:
    """Turn SHAP top_factors into two simple-language phrases: things working
    against the person, and things working in their favor. Also returns
    whether each phrase is plural, for correct verb agreement."""
    raising = [f["label"] for f in top_factors if f.get("direction") == "increases_risk"][:max_items]
    lowering = [f["label"] for f in top_factors if f.get("direction") != "increases_risk"][:max_items]

    def join(items):
        items = [i[0].lower() + i[1:] if i else i for i in items]
        if not items:
            return "", False
        if len(items) == 1:
            return items[0], False
        return f"{items[0]} and {items[1]}", True

    raising_txt, _ = join(raising)
    lowering_txt, lowering_plural = join(lowering)
    return raising_txt, lowering_txt, lowering_plural, len(raising) > 1


def _build_fallback_summary(tier: int, top_factors: list[dict]) -> str:
    """Detailed, plain-language summary built from this person's actual top
    factors — not a single static sentence reused for everyone at a tier."""
    raising_txt, lowering_txt, lowering_plural, _ = _describe_factors(top_factors)
    lowering_verb = "are" if lowering_plural else "is"

    if tier == 1:
        base = "Your results show a low likelihood of PCOS right now."
        if raising_txt:
            base += f" One thing worth keeping an eye on is {raising_txt}."
        if lowering_txt:
            base += f" On the positive side, {lowering_txt} {lowering_verb} working in your favor."
        base += " Keep up your current habits and re-check in about 6 months."
        return base

    if tier == 2:
        base = "Your results show some indicators linked to PCOS, though nothing conclusive yet."
        if raising_txt:
            base += f" The main things pushing your risk up are {raising_txt}."
        if lowering_txt:
            base += f" {lowering_txt.capitalize()} {lowering_verb} helping balance that out."
        base += (
            " This is a good time to make some lifestyle changes and get checked by a doctor "
            "in the next few weeks so you have a clearer picture."
        )
        return base

    base = "Your results show several strong indicators that align with PCOS."
    if raising_txt:
        base += f" In particular, {raising_txt} stood out as the biggest contributors."
    base += (
        " This isn't a diagnosis — only a doctor can confirm that — but it does mean you "
        "shouldn't wait. Please book a consultation soon so you can get proper testing and support."
    )
    return base


FALLBACK_ADVICE = {
    1: {
        "summary": "Indicators are low. No signs of PCOS currently.",
        "lifestyle": ["Maintain a balanced diet with whole grains and vegetables",
                      "Keep up regular physical activity (~150 min/week)"],
        "ayurvedic": ["Herbal teas like spearmint or cinnamon may support hormonal balance — supportive only"],
        "avoid": ["Excess processed/junk food", "Prolonged periods of inactivity"],
        "doctor_action": "Not needed now. Re-screen in 6 months.",
        "mental_wellbeing": ["Practice short daily mindfulness or breathing breaks to manage stress"],
    },
    2: {
        "summary": "Some indicators present. Not conclusive but worth monitoring.",
        "lifestyle": ["Reduce refined sugar and junk food intake", "Aim for consistent sleep and moderate exercise 4-5x/week",
                      "Track your cycle length for the next 2-3 months"],
        "ayurvedic": ["Ashwagandha or fenugreek are sometimes used for hormonal support — consult a practitioner before use"],
        "avoid": ["Skipping meals or crash dieting", "Ignoring persistent irregular cycles"],
        "doctor_action": "Recommended within 4-6 weeks if symptoms persist.",
        "mental_wellbeing": ["Stress can affect hormonal cycles — consider journaling or light yoga", "Talk to someone you trust about how you're feeling"],
    },
    3: {
        "summary": "Multiple strong indicators align with PCOS.",
        "lifestyle": ["Adopt a low-glycemic, anti-inflammatory diet pattern", "Prioritize strength + cardio exercise most days of the week"],
        "ayurvedic": ["Supportive only — not a substitute for treatment"],
        "avoid": ["Self-medicating", "Delaying consultation"],
        "doctor_action": "Consult a gynecologist within 1-2 weeks. This is not optional.",
        "mental_wellbeing": ["A high-risk result can feel overwhelming — that reaction is normal; support from a professional can help with both the physical and emotional side"],
    },
}


def _sanitize(advice: dict, tier: int) -> dict:
    """Defense-in-depth: strip dosage-like patterns, enforce doctor_action wording for tier 2/3."""
    for key in ("summary", "doctor_action"):
        if key in advice and isinstance(advice[key], str):
            advice[key] = BANNED_DOSAGE_PATTERN.sub("[dose omitted]", advice[key])
    for key in ("lifestyle", "ayurvedic", "avoid", "mental_wellbeing"):
        if key in advice and isinstance(advice[key], list):
            advice[key] = [BANNED_DOSAGE_PATTERN.sub("[dose omitted]", str(x)) for x in advice[key]]

    if tier in (2, 3):
        da = advice.get("doctor_action", "")
        if "consult" not in da.lower() and "doctor" not in da.lower() and "gynecologist" not in da.lower():
            advice["doctor_action"] = FALLBACK_ADVICE[tier]["doctor_action"]
    return advice


def generate_advice(tier: int, label: str, probability: float, top_factors: list[dict],
                     personalization: dict) -> dict:
    """
    Layers 6+7 combined entry point. Returns the `advice` sub-object of the
    final response JSON.
    """
    client = _client()
    if client is None:
        logger.info("Groq client unavailable (no API key or SDK missing); using fallback advice for tier %s.", tier)
        advice = dict(FALLBACK_ADVICE[tier])
        advice["summary"] = _build_fallback_summary(tier, top_factors)
        return _sanitize(advice, tier)

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.4,
            max_tokens=600,
            messages=[
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": _build_user_prompt(tier, label, probability, top_factors, personalization)},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        advice = json.loads(raw)
        for key in ("summary", "lifestyle", "ayurvedic", "avoid", "doctor_action"):
            if key not in advice:
                advice[key] = FALLBACK_ADVICE[tier][key]
        advice.setdefault("mental_wellbeing", FALLBACK_ADVICE[tier]["mental_wellbeing"])
        return _sanitize(advice, tier)
    except Exception as e:
        # Log the real reason server-side (rate limit, timeout, bad JSON, etc.)
        # but never leak internal service status into user-facing text.
        logger.warning("LLM guidance generation failed, using fallback advice for tier %s: %s", tier, e)
        advice = dict(FALLBACK_ADVICE[tier])
        advice["summary"] = _build_fallback_summary(tier, top_factors)
        return _sanitize(advice, tier)