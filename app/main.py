"""
main.py — Flask entrypoint. Wires Layers 1-7 together and serves the
single-page frontend (templates/index.html + static/).

Endpoints:
    GET  /                -> frontend
    POST /api/predict     -> runs the full 7-layer pipeline, returns JSON
    POST /api/pdf         -> takes a previous result JSON, returns a PDF
    GET  /api/health      -> liveness check
"""

from __future__ import annotations
import os
import pickle
import logging

from flask import Flask, request, jsonify, render_template, send_file, Response
from dotenv import load_dotenv
import io

from app.preprocessing import raw_input_to_row
from app.explainability import ShapExplainer
from app.tier_logic import decide_tier, lab_suggestions_for
from app.llm_service import generate_advice
from app.pdf_report import build_pdf

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pcos-app")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "app", "model", "pcos_model.pkl")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

_ARTIFACT = None
_EXPLAINER = None


def get_artifact():
    """Lazy-load the trained model artifact (Layer 2) once per process."""
    global _ARTIFACT, _EXPLAINER
    if _ARTIFACT is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run `python app/model/train.py` first."
            )
        with open(MODEL_PATH, "rb") as f:
            _ARTIFACT = pickle.load(f)
        _EXPLAINER = ShapExplainer(_ARTIFACT["model"], _ARTIFACT["feature_order"])
        logger.info("Loaded model artifact: %s", _ARTIFACT.get("model_type"))
    return _ARTIFACT, _EXPLAINER


def _fmt(value, suffix=""):
    """Render a form value for display; keep 'not provided' explicit rather than blank."""
    if value in (None, "", "null"):
        return "Not provided"
    return f"{value}{suffix}"


def _build_user_details(payload: dict) -> dict:
    """
    Turn the raw /api/predict request payload into the friendly label -> value
    pairs shown in the 'Your Entered Details' table, both on the results page
    and in the PDF. Keys here MUST match the field names actually sent by
    templates/index.html's JS — verify against your frontend's fetch() body
    and adjust the payload.get(...) keys below if they differ.
    """
    height_cm = payload.get("height_cm")
    weight_kg = payload.get("weight_kg")
    bmi = None
    try:
        if height_cm and weight_kg:
            h_m = float(height_cm) / 100.0
            bmi = round(float(weight_kg) / (h_m * h_m), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        bmi = None

    return {
        "Age": _fmt(payload.get("age")),
        "Height (cm)": _fmt(height_cm),
        "Weight (kg)": _fmt(weight_kg),
        "BMI (auto)": _fmt(bmi),
        "Systolic BP": _fmt(payload.get("systolic_bp")),
        "Diastolic BP": _fmt(payload.get("diastolic_bp")),
        "Pulse rate": _fmt(payload.get("pulse_rate")),
        "Marital status": _fmt(payload.get("marital_status")),
        "Ultrasound: PCOD detected?": _fmt(payload.get("ultrasound_pcod")),
        "Period cycle": _fmt(payload.get("period_cycle")),
        "Avg. gap between periods (days)": _fmt(payload.get("avg_gap_days")),
        "Weight change >5kg recently?": _fmt(payload.get("weight_change_recent")),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    try:
        artifact, _ = get_artifact()
        return jsonify({"status": "ok", "model_type": artifact.get("model_type"),
                         "metrics": artifact.get("metrics")})
    except Exception as e:
        return jsonify({"status": "model_not_loaded", "detail": str(e)}), 503


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True, silent=True) or {}

    required = ["age", "height_cm", "weight_kg"]
    missing = [f for f in required if not payload.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        artifact, explainer = get_artifact()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    consulted = str(payload.get("consulted_doctor", "No")).strip().lower() == "yes"

    # ---- Layer 1: preprocessing ----
    try:
        row_df = raw_input_to_row(payload, medians=artifact["medians"])
    except Exception as e:
        return jsonify({"error": f"Preprocessing failed: {e}"}), 400

    # ---- Layer 2: ML classification ----
    model = artifact["model"]
    try:
        proba = float(model.predict_proba(row_df)[0][1])
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    # ---- Layer 3: explainability ----
    try:
        top_factors = explainer.top_factors(row_df, top_n=3)
    except Exception as e:
        logger.warning("SHAP explanation failed, continuing without it: %s", e)
        top_factors = []

    # ---- Layers 4+5: red-flag override + tier decision ----
    tier_result = decide_tier(proba, payload)
    tier = tier_result["tier"]

    # ---- Layers 6+7: LLM reasoning + personalization ----
    personalization = {
        "consulted_doctor": consulted,
        "diagnosed_condition": payload.get("diagnosed_condition", "None"),
        "medication": payload.get("current_medication", "None"),
    }
    advice = generate_advice(
        tier=tier,
        label=tier_result["label"],
        probability=proba,
        top_factors=top_factors,
        personalization=personalization,
    )

    response = {
        "tier": tier,
        "label": tier_result["label"],
        "probability": round(proba, 4),
        "top_factors": top_factors,
        "advice": advice,
        "red_flag_triggered": tier_result["red_flag_triggered"],
        "red_flag_reasons": tier_result["red_flag_reasons"],
        "tier_reason": tier_result["reason"],
        "consulted_doctor": consulted,
        # Echo back what the user entered, in display-ready form, so the
        # frontend can show it and forward this same object to /api/pdf.
        "user_details": _build_user_details(payload),
    }

    if not consulted:
        response["lab_suggestions"] = lab_suggestions_for(tier)

    return jsonify(response)


@app.route("/api/pdf", methods=["POST"])
def pdf_report():
    result = request.get_json(force=True, silent=True) or {}
    if not result:
        return jsonify({"error": "No result payload provided"}), 400
    try:
        pdf_bytes = build_pdf(result)
    except Exception as e:
        return jsonify({"error": f"PDF generation failed: {e}"}), 500
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="pcos_screening_report.pdf",
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG", "0") == "1")