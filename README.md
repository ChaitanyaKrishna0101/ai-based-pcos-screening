# URS PCOS — PCOS Early Screening

A full-stack PCOS **screening** web app (not a diagnostic tool). One page,
guided flow: intro → basic details → doctor-consultation branch → animated
result with risk gauge, SHAP-based explanation, AI-generated lifestyle
guidance, and a downloadable PDF report.

⚠️ **This app estimates risk from self-reported symptoms. It is not a
medical diagnosis and does not replace a licensed clinician.**

## The 7-layer pipeline

| Layer | File | Job |
|---|---|---|
| 1. Preprocessing | `app/preprocessing.py` | Encode categoricals, scale/impute numerics — shared identically by training and inference |
| 2. ML classification | `app/model/train.py` → `pcos_model.pkl` | XGBoost (falls back to RandomForest) outputs a 0–1 probability |
| 3. Explainability | `app/explainability.py` | SHAP computes the top-3 features driving *this specific* prediction |
| 4. Clinical override | `app/tier_logic.py` | Red-flag rules force Tier 3 when clinically obvious, regardless of ML score |
| 5. Tier decision | `app/tier_logic.py` | Combines probability + red flags (+ existing-diagnosis exception) into the final tier |
| 6. LLM reasoning | `app/llm_service.py` | Groq (`llama-3.3-70b-versatile`) turns tier + SHAP factors into structured advice |
| 7. Personalization | `app/llm_service.py` | Injects doctor-consultation/medication history so advice never contradicts existing treatment |

If `GROQ_API_KEY` isn't set, Layer 6/7 fall back to safe, deterministic
template advice — the app still fully works, just without AI-generated
phrasing.

## Run locally (2 commands)

```bash
git clone <this-repo> && cd pcos-predictor
./run_local.sh
```
That's it — the script creates a virtualenv, installs dependencies, trains
the model on first run, and starts the app at **http://localhost:7860**.

To enable AI-generated advice, add your free key from
[console.groq.com/keys](https://console.groq.com/keys) to `.env`
(created automatically from `.env.example` on first run):
```
GROQ_API_KEY=gsk_xxxxxxxxxxxx
```

### Manual setup (equivalent, if you don't want to use the script)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app/model/train.py
python -m app.main
```

## Deploy to Hugging Face Spaces (Docker SDK)

1. Create a new Space → SDK: **Docker** → hardware: CPU basic is enough.
2. Push this entire folder to the Space's repo (or connect this GitHub repo).
3. In the Space **Settings → Repository secrets**, add:
   - `GROQ_API_KEY` = your Groq key (optional but recommended)
4. That's it. The Dockerfile trains the model at build time and serves the
   app on port `7860`, which Spaces expects automatically.

No other configuration is needed — Spaces will build the image and the app
will be live at your Space's URL.

## Using the real Kaggle dataset instead of synthetic data

The repo ships with a synthetic dataset generator
(`data/generate_synthetic_data.py`) so the whole pipeline runs out of the
box. For a clinically-grounded model:

1. Download `PCOS_data_without_infertility.csv` from Kaggle
   ("PCOS Dataset" by Prasoon Kottarathil).
2. Save it as `data/pcos_dataset.csv` (matching the column names in
   `app/preprocessing.py` — rename columns if needed).
3. Re-run `python app/model/train.py`.
4. Re-build/redeploy.

## Project structure

```
pcos-predictor/
├── Dockerfile
├── requirements.txt
├── run_local.sh
├── .env.example
├── app/
│   ├── main.py              # Flask entrypoint, /api/predict, /api/pdf
│   ├── preprocessing.py     # Layer 1
│   ├── explainability.py    # Layer 3 (SHAP)
│   ├── tier_logic.py        # Layers 4 + 5
│   ├── llm_service.py       # Layers 6 + 7 (Groq)
│   ├── pdf_report.py        # PDF report builder
│   └── model/
│       ├── train.py         # Layer 2 training script
│       └── pcos_model.pkl   # generated on first run/build
├── templates/index.html     # single-page wizard UI
├── static/style.css
├── static/app.js
└── data/
    ├── generate_synthetic_data.py
    └── pcos_dataset.csv     # generated on first run (or replace with real Kaggle CSV)
```

## API

- `GET /api/health` — liveness + model metrics
- `POST /api/predict` — runs all 7 layers, returns tier/probability/SHAP factors/advice
- `POST /api/pdf` — takes a previous `/api/predict` response, returns a PDF report

## Notes on safety guardrails baked into the LLM layer

- Never states or implies a diagnosis
- Never outputs medication names with dosages (regex-filtered as a second safety net)
- Always recommends professional consultation for Tier 2/3
- Ayurvedic/natural suggestions are always labeled supportive-only
- Advice is medication-aware when the user reports existing treatment, and is instructed never to contradict it
